#!/usr/bin/env python3
"""Qwen3-VL-Embedding adapter.

Qwen/Qwen3-VL-Embedding-8B is exposed through SentenceTransformers in the model
card. This adapter intentionally uses that public embedding API rather than the
plain Transformers AutoModel API, because AutoModel does not expose image/text
embedding helpers for this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass
class QwenEmbConfig:
    model_id: str = "Qwen/Qwen3-VL-Embedding-8B"
    device: str = "cuda"
    device_map: str | None = None  # kept for CLI compatibility; ignored by SentenceTransformer
    dtype: str = "bfloat16"        # kept for CLI compatibility; model card path handles dtype internally
    trust_remote_code: bool = True # kept for compatibility
    dim: int | None = None
    normalize: bool = True
    query_prompt: str | None = "Retrieve relevant fashion product images for the query."


def _to_numpy(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype("float32", copy=False)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (x / denom).astype("float32", copy=False)


class QwenEmbEncoder:
    def __init__(self, cfg: QwenEmbConfig):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        # The official model card uses SentenceTransformer("Qwen/Qwen3-VL-Embedding-8B").
        # Keep construction minimal for compatibility across sentence-transformers releases.
        self.model = SentenceTransformer(cfg.model_id, device=cfg.device)

    @property
    def device(self):
        return getattr(self.model, "device", self.cfg.device)

    def _postprocess(self, arr: Any) -> np.ndarray:
        arr = _to_numpy(arr)
        if self.cfg.dim is not None:
            if self.cfg.dim <= 0 or self.cfg.dim > arr.shape[1]:
                raise ValueError(f"Invalid dim={self.cfg.dim} for embedding shape={arr.shape}")
            arr = arr[:, : self.cfg.dim]
        if self.cfg.normalize:
            arr = _l2_normalize(arr)
        return arr.astype("float32", copy=False)

    def _encode(self, items, *, prompt: str | None = None) -> np.ndarray:
        kwargs = {
            "convert_to_numpy": True,
            "show_progress_bar": False,
            "normalize_embeddings": False,  # postprocess handles dim truncation before normalization
        }
        if prompt:
            kwargs["prompt"] = prompt
        return self._postprocess(self.model.encode(items, **kwargs))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype="float32")
        try:
            return self._encode(texts, prompt=self.cfg.query_prompt)
        except TypeError:
            # Older sentence-transformers releases may not accept `prompt`.
            return self._encode(texts, prompt=None)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        images = [img.convert("RGB") for img in images]
        if not images:
            return np.empty((0, 0), dtype="float32")

        # First try direct PIL inputs. Some SentenceTransformer multimodal models accept this.
        try:
            return self._encode([{"image": img} for img in images], prompt=None)
        except Exception as direct_error:
            # Fallback: the Qwen model card demonstrates images as paths/URLs inside {"image": ...}.
            # Use a per-batch temp directory to avoid writing into the corpus.
            try:
                with TemporaryDirectory(prefix="qwenemb_img_") as tmpdir:
                    items = []
                    for i, img in enumerate(images):
                        p = Path(tmpdir) / f"{i:06d}.jpg"
                        img.save(p, "JPEG", quality=95)
                        items.append({"image": str(p)})
                    return self._encode(items, prompt=None)
            except Exception as path_error:
                raise RuntimeError(
                    "Could not obtain image embeddings through SentenceTransformer. "
                    f"Direct PIL error: {direct_error!r}; temp-path error: {path_error!r}"
                ) from path_error
