#!/usr/bin/env python3
"""
Generate scene descriptions for every image of the CSS dataset with your own model.

Loads ``starsky77/counterfactual-semantic-saliency`` (307 factual scenes + 1,306
counterfactual variants), queries the chosen backend with the fixed CSS prompt
``--n-samples`` times per image (default 5, as in the paper), and appends one
JSON line per image to ``<out-dir>/raw_descriptions.jsonl``. Re-running the same
command resumes: images already present in the JSONL are skipped.

Then run ``score_descriptions.py`` to produce the ``scored_descriptions_*.csv``
consumed by ``reproduce.sh``.

Examples:
  # OpenAI (or any OpenAI-compatible API)
  python Src/custom_model/generate_descriptions.py --model-name gpt-4o \
      --backend openai --model gpt-4o --workers 8

  # Local model served with vLLM (`vllm serve Qwen/Qwen3-VL-8B-Instruct`)
  python Src/custom_model/generate_descriptions.py --model-name Qwen_Qwen3-VL-8B-Instruct \
      --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct

  # Local HuggingFace transformers model
  python Src/custom_model/generate_descriptions.py --model-name Qwen_Qwen3-VL-4B-Instruct \
      --backend hf --model Qwen/Qwen3-VL-4B-Instruct

  # Your own function: def generate(image, prompt, n) -> list[str]
  python Src/custom_model/generate_descriptions.py --model-name my_model \
      --backend custom --custom path/to/my_model.py:generate
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from backends import CustomBackend, HFBackend, OpenAIBackend
from prompt import PROMPT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "starsky77/counterfactual-semantic-saliency"
DEFAULT_RUNS_ROOT = REPO_ROOT / "CSS_result" / "custom_runs"


def load_css_dataset(dataset: str):
    """Load the CSS dataset from the Hub id or a local copy of its ``image/`` folder.

    Returns a list of dicts with file_name, condition, original_file_name and a
    lazily-decoded image reference.
    """
    from datasets import Image as HFImage
    from datasets import load_dataset

    local = Path(dataset)
    if local.is_dir():
        data_dir = local / "image" if (local / "image").is_dir() else local
        ds = load_dataset("imagefolder", data_dir=str(data_dir), split="train")
    else:
        ds = load_dataset(dataset, data_dir="image", split="train")
    # imagefolder turns every ``*file_name`` metadata field into an image column:
    # ``file_name`` -> ``image`` and ``original_file_name`` -> ``original``.
    ds = ds.cast_column("image", HFImage(decode=False))
    if "original" in ds.column_names:
        ds = ds.cast_column("original", HFImage(decode=False))

    def basename(ref: dict) -> str:
        if not ref or not ref.get("path"):
            raise RuntimeError("Could not recover a file name from the dataset image path.")
        return Path(ref["path"]).name

    records = []
    for row in ds:
        if "original_file_name" in row:
            original_file_name = row["original_file_name"]
        else:
            original_file_name = basename(row["original"])
        records.append(
            {
                "file_name": basename(row["image"]),
                "condition": row["condition"],
                "original_file_name": original_file_name,
                "image_ref": row["image"],
            }
        )
    return records


def open_image(ref: dict) -> Image.Image:
    if ref.get("bytes"):
        return Image.open(io.BytesIO(ref["bytes"])).convert("RGB")
    return Image.open(ref["path"]).convert("RGB")


def load_done(jsonl_path: Path, n_samples: int) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.is_file():
        return done
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if len(rec.get("descriptions", [])) >= n_samples:
                done.add(rec["file_name"])
    return done


def build_backend(args: argparse.Namespace):
    if args.backend == "openai":
        if not args.model:
            raise SystemExit("--model is required for --backend openai")
        return OpenAIBackend(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    if args.backend == "hf":
        if not args.model:
            raise SystemExit("--model is required for --backend hf")
        return HFBackend(
            model=args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens or 256,
            trust_remote_code=args.trust_remote_code,
        )
    if args.backend == "custom":
        if not args.custom:
            raise SystemExit("--custom is required for --backend custom")
        return CustomBackend(args.custom, thread_safe=args.workers > 1)
    raise SystemExit(f"Unknown backend: {args.backend}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-name", required=True,
                   help="Name of this run (used for output folders and figure labels). Must not contain 'human'.")
    p.add_argument("--backend", choices=["openai", "hf", "custom"], required=True)
    p.add_argument("--model", help="Model id for the openai / hf backends.")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1 (default: OpenAI).")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="Environment variable holding the API key (default: OPENAI_API_KEY).")
    p.add_argument("--custom", help="User function 'module:func' or 'file.py:func' for --backend custom.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature (default: backend/provider default). Do not use 0; "
                        "the paper uses stochastic sampling.")
    p.add_argument("--top-p", type=float, default=None, help="top_p for --backend hf.")
    p.add_argument("--max-tokens", type=int, default=None, help="Max generated tokens.")
    p.add_argument("--device-map", default="auto", help="device_map for --backend hf.")
    p.add_argument("--dtype", default="auto", help="torch dtype for --backend hf (auto, bfloat16, float16).")
    p.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code to transformers.")
    p.add_argument("--n-samples", type=int, default=5, help="Descriptions per image (default 5, as in the paper).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel images (only for thread-safe backends such as openai).")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HF dataset id or local path to the downloaded dataset (default: %(default)s).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output folder (default: CSS_result/custom_runs/<model-name>/).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N scenes (factual + its counterfactuals), for smoke tests.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "human" in args.model_name.lower():
        raise SystemExit("--model-name must not contain 'human' (such runs are skipped by the analysis).")

    out_dir = args.out_dir or (DEFAULT_RUNS_ROOT / args.model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "raw_descriptions.jsonl"

    records = load_css_dataset(args.dataset)
    if args.limit is not None:
        scenes = sorted({r["original_file_name"] for r in records})[: args.limit]
        keep = set(scenes)
        records = [r for r in records if r["original_file_name"] in keep]

    done = load_done(jsonl_path, args.n_samples)
    todo = [r for r in records if r["file_name"] not in done]
    print(f"{len(records)} images, {len(done)} already done, {len(todo)} to generate -> {jsonl_path}")
    if not todo:
        return

    with (out_dir / "generation_config.json").open("w", encoding="utf-8") as f:
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        json.dump({**cfg, "prompt": PROMPT}, f, indent=2)

    backend = build_backend(args)
    workers = args.workers if getattr(backend, "thread_safe", False) else 1
    if workers != args.workers:
        print(f"Backend '{args.backend}' is not thread-safe; using --workers 1.")

    lock = threading.Lock()
    failures = 0

    def run_one(rec: dict) -> dict:
        descs = backend.generate(open_image(rec["image_ref"]), PROMPT, args.n_samples)
        if len(descs) != args.n_samples:
            raise RuntimeError(f"expected {args.n_samples} descriptions, got {len(descs)}")
        return {
            "file_name": rec["file_name"],
            "condition": rec["condition"],
            "original_file_name": rec["original_file_name"],
            "descriptions": descs,
        }

    with jsonl_path.open("a", encoding="utf-8") as out_f, tqdm(total=len(todo)) as bar:
        def write(result: dict) -> None:
            with lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

        if workers == 1:
            for rec in todo:
                try:
                    write(run_one(rec))
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"[fail] {rec['file_name']}: {e}", file=sys.stderr)
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(run_one, rec): rec for rec in todo}
                for fut in as_completed(futs):
                    rec = futs[fut]
                    try:
                        write(fut.result())
                    except Exception as e:  # noqa: BLE001
                        failures += 1
                        print(f"[fail] {rec['file_name']}: {e}", file=sys.stderr)
                    bar.update(1)

    if failures:
        print(f"{failures} image(s) failed; re-run the same command to retry them.", file=sys.stderr)
        sys.exit(1)
    print(f"Done. Descriptions in {jsonl_path}")


if __name__ == "__main__":
    main()
