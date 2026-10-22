#!/usr/bin/env python3
"""
Score generated descriptions into a ``scored_descriptions_*.csv`` for the CSS analysis.

For every counterfactual image, ``mean_similarity_score`` is the mean pairwise
cosine similarity between the sentence embeddings of the counterfactual
descriptions (``Caption``) and the factual-scene descriptions
(``Original_Caption``), using Jasper (``infgrad/jasper_en_vision_language_v1``).
CSS = 1 - mean_similarity_score.

Input (one of):
  --raw       raw_descriptions.jsonl written by generate_descriptions.py
  --from-csv  a CSV you built yourself with columns img_fn, Original_Caption,
              Caption (Python-list strings of descriptions)

Output: CSS_result/Cache/<YYYYMMDD_HHMMSS><model-name>/scored_descriptions_<YYYYMMDD_HHMMSS>.csv
with columns img_fn, Original_Caption, Caption, mean_similarity_score, which
``bash reproduce.sh`` picks up automatically.

Examples:
  python Src/custom_model/score_descriptions.py --model-name my_model \
      --raw CSS_result/custom_runs/my_model/raw_descriptions.jsonl
  python Src/custom_model/score_descriptions.py --model-name my_model --from-csv my_captions.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "CSS_result" / "Cache"
HUMAN_BASELINE = (
    CACHE_ROOT / "human_filtered_10" / "scene_description_human_filtered_all_10_with_mean_similarity_score.csv"
)
JASPER_MODEL = "infgrad/jasper_en_vision_language_v1"
OUTPUT_COLUMNS = ["img_fn", "Original_Caption", "Caption", "mean_similarity_score"]


class JasperScorer:
    """Mean pairwise cosine similarity with Jasper, matching the paper's scoring code."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading {JASPER_MODEL} on {device} ...")
            self._model = SentenceTransformer(JASPER_MODEL, trust_remote_code=True, device=device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        emb = np.asarray(self.model.encode(texts))
        return emb.reshape(1, -1) if emb.ndim == 1 else emb

    def mean_pairwise_similarity(self, caps: list[str], origs: list[str]) -> float | None:
        if not isinstance(caps, list) or not isinstance(origs, list):
            return None
        if len(caps) == 0 or len(origs) == 0:
            return None
        emb_c = self.embed(caps)
        emb_o = self.embed(origs)
        norm_c = np.linalg.norm(emb_c, axis=1, keepdims=True)
        norm_o = np.linalg.norm(emb_o, axis=1, keepdims=True)
        if np.any(norm_c == 0) or np.any(norm_o == 0):
            return None
        sim_matrix = (emb_c @ emb_o.T) / (norm_c @ norm_o.T)
        return float(sim_matrix.mean())


def table_from_raw(jsonl_path: Path) -> pd.DataFrame:
    """Pair each counterfactual image with the descriptions of its factual scene."""
    by_file: dict[str, dict] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_file[rec["file_name"]] = rec

    rows, missing_orig = [], []
    for fn, rec in sorted(by_file.items()):
        if rec["condition"] != "counterfactual":
            continue
        orig = by_file.get(rec["original_file_name"])
        if orig is None:
            missing_orig.append(fn)
            continue
        rows.append(
            {
                "img_fn": fn,
                "Original_Caption": list(orig["descriptions"]),
                "Caption": list(rec["descriptions"]),
            }
        )
    if missing_orig:
        print(
            f"Warning: {len(missing_orig)} counterfactual image(s) skipped because their factual "
            f"scene has no descriptions (e.g. {missing_orig[0]}).",
            file=sys.stderr,
        )
    return pd.DataFrame(rows, columns=["img_fn", "Original_Caption", "Caption"])


def table_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"img_fn", "Original_Caption", "Caption"} - set(df.columns)
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {sorted(missing)}")
    df = df[["img_fn", "Original_Caption", "Caption"]].copy()
    for col in ("Original_Caption", "Caption"):
        df[col] = df[col].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    return df


def validate(df: pd.DataFrame, n_samples: int | None) -> list[str]:
    problems: list[str] = []
    if HUMAN_BASELINE.is_file():
        expected = set(pd.read_csv(HUMAN_BASELINE, usecols=["img_fn"])["img_fn"].astype(str))
        got = set(df["img_fn"].astype(str))
        if got != expected:
            problems.append(
                f"img_fn set differs from the human baseline: {len(expected - got)} missing, "
                f"{len(got - expected)} unexpected (expected {len(expected)} counterfactual images)."
            )
    if df["img_fn"].duplicated().any():
        problems.append(f"{int(df['img_fn'].duplicated().sum())} duplicated img_fn rows.")
    if n_samples is not None:
        for col in ("Original_Caption", "Caption"):
            bad = int((df[col].map(len) != n_samples).sum())
            if bad:
                problems.append(f"{bad} rows where {col} does not have {n_samples} descriptions.")
    n_nan = int(df["mean_similarity_score"].isna().sum())
    if n_nan:
        problems.append(f"{n_nan} rows with missing mean_similarity_score.")
    return problems


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", type=Path, help="raw_descriptions.jsonl from generate_descriptions.py")
    src.add_argument("--from-csv", type=Path, help="CSV with img_fn, Original_Caption, Caption columns")
    p.add_argument("--model-name", help="Run name; output goes to CSS_result/Cache/<timestamp><model-name>/.")
    p.add_argument("--output", type=Path, default=None, help="Explicit output CSV path (overrides --model-name).")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Expected descriptions per image, for validation (default 5; 0 disables the check).")
    p.add_argument("--device", default=None, help="Device for Jasper (default: cuda if available, else cpu).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        if not args.model_name:
            raise SystemExit("Pass --model-name (or --output).")
        if "human" in args.model_name.lower():
            raise SystemExit("--model-name must not contain 'human' (such runs are skipped by the analysis).")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = CACHE_ROOT / f"{ts}{args.model_name}" / f"scored_descriptions_{ts}.csv"
    else:
        output = args.output

    df = table_from_raw(args.raw) if args.raw else table_from_csv(args.from_csv)
    if df.empty:
        raise SystemExit("No counterfactual rows to score.")
    df = df.sort_values("img_fn").reset_index(drop=True)

    scorer = JasperScorer(device=args.device)
    tqdm.pandas()
    df["mean_similarity_score"] = df.progress_apply(
        lambda row: scorer.mean_pairwise_similarity(row["Caption"], row["Original_Caption"]), axis=1
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")

    problems = validate(df, args.n_samples or None)
    if problems:
        print("Validation warnings:", file=sys.stderr)
        for msg in problems:
            print(f"  - {msg}", file=sys.stderr)
        print("The analysis only uses masks present in both the human baseline and this file.", file=sys.stderr)
    else:
        print("Validation passed: matches the human baseline image set.")


if __name__ == "__main__":
    main()
