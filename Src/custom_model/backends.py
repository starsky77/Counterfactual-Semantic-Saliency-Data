"""Model backends for generating scene descriptions.

Every backend exposes ``generate(image, prompt, n) -> list[str]`` which returns
``n`` independently sampled descriptions for one PIL image.

Backends:
  * ``openai``  OpenAI-compatible chat-completions API (OpenAI, vLLM, SGLang,
                Ollama, Gemini's OpenAI endpoint, ...).
  * ``hf``      Local HuggingFace transformers vision-language model.
  * ``custom``  A user function ``module.path:func`` or ``path/to/file.py:func``
                with the same signature.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import os
import random
import time
from pathlib import Path
from typing import Callable

from PIL import Image


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


class OpenAIBackend:
    """OpenAI-compatible chat-completions backend. Each of the ``n`` samples is
    a separate request, since many providers ignore or reject ``n > 1``."""

    thread_safe = True

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 6,
        image_format: str = "PNG",
    ) -> None:
        from openai import OpenAI

        api_key = os.environ.get(api_key_env) or "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.image_format = image_format

    def _one(self, data_url: str, prompt: str) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - provider errors vary widely
                if attempt == self.max_retries:
                    raise
                wait = min(60.0, 2.0**attempt) + random.random()
                print(f"[openai] {type(e).__name__}: {e}; retry in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def generate(self, image: Image.Image, prompt: str, n: int) -> list[str]:
        data_url = image_to_data_url(image, self.image_format)
        return [self._one(data_url, prompt) for _ in range(n)]


class HFBackend:
    """Local transformers VLM loaded with ``AutoModelForImageTextToText``.
    Uses stochastic sampling (``do_sample=True``) as in the paper."""

    thread_safe = False

    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        dtype: str = "auto",
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int = 256,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        ).eval()
        self.gen_kwargs: dict = {"do_sample": True, "max_new_tokens": max_new_tokens}
        if temperature is not None:
            self.gen_kwargs["temperature"] = temperature
        if top_p is not None:
            self.gen_kwargs["top_p"] = top_p

    def generate(self, image: Image.Image, prompt: str, n: int) -> list[str]:
        import torch

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image.convert("RGB")], return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, num_return_sequences=n, **self.gen_kwargs)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.strip() for t in texts]


class CustomBackend:
    """Wraps a user function ``fn(image: PIL.Image, prompt: str, n: int) -> list[str]``."""

    def __init__(self, spec: str, thread_safe: bool = False) -> None:
        self.fn = load_callable(spec)
        self.thread_safe = thread_safe

    def generate(self, image: Image.Image, prompt: str, n: int) -> list[str]:
        out = self.fn(image, prompt, n)
        if isinstance(out, str):
            out = [out]
        return [str(t).strip() for t in out]


def load_callable(spec: str) -> Callable:
    """Load ``module.path:func`` or ``path/to/file.py:func``."""
    if ":" not in spec:
        raise ValueError(f"--custom must look like 'module:func' or 'file.py:func', got {spec!r}")
    target, func_name = spec.rsplit(":", 1)
    if target.endswith(".py") or os.sep in target:
        path = Path(target).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn
