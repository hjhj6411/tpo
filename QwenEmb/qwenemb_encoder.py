#!/usr/bin/env python3
"""Small adapter for Qwen multimodal embedding models.

The Qwen3-VL-Embedding model family is expected to expose an embedding-oriented
API through its Hugging Face / remote-code implementation. Since the exact method
names can differ across releases, this adapter tries common embedding methods
first and fails loudly rather than silently using invalid causal-LM hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image


@dataclass
class QwenEmbConfig:
    model_id: str = "Qwen/Qwen3-VL-Embedding-8B"
    device: str = "cuda"
    device_map: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    dim: int | None = None
    normalize: bool = True


def _dtype(name: str):
    name = (name or "").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif torch.is_tensor(x):
        arr = x.detach().float().cpu().numpy()
    elif isinstance(x, (list, tuple)) and x and torch.is_tensor(x[0]):
        arr = torch.stack(list(x)).detach().float().cpu().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype("float32", copy=False)


def _extract_embedding_from_output(out: Any) -> np.ndarray | None:
    """Extract a 2D embedding tensor from common model output shapes."""
    if out is None:
        return None

    # Common dataclass / ModelOutput fields used by retrieval models.
    for name in (
        "embeddings", "embedding", "sentence_embeddings", "text_embeds",
        "image_embeds", "pooler_output",
    ):
        if hasattr(out, name):
            val = getattr(out, name)
            if val is not None:
                arr = _to_numpy(val)
                if arr.ndim == 2:
                    return arr

    if isinstance(out, dict):
        for name in (
            "embeddings", "embedding", "sentence_embeddings", "text_embeds",
            "image_embeds", "pooler_output",
        ):
            if name in out and out[name] is not None:
                arr = _to_numpy(out[name])
                if arr.ndim == 2:
                    return arr

    if torch.is_tensor(out) or isinstance(out, np.ndarray):
        arr = _to_numpy(out)
        if arr.ndim == 2:
            return arr

    if isinstance(out, (list, tuple)):
        for item in out:
            arr = _extract_embedding_from_output(item)
            if arr is not None:
                return arr

    return None


class QwenEmbEncoder:
    def __init__(self, cfg: QwenEmbConfig):
        from transformers import AutoModel, AutoProcessor

        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        model_kwargs = {
            "trust_remote_code": cfg.trust_remote_code,
            "torch_dtype": _dtype(cfg.dtype),
        }
        if cfg.device_map:
            model_kwargs["device_map"] = cfg.device_map
            self.model = AutoModel.from_pretrained(cfg.model_id, **model_kwargs)
        else:
            self.model = AutoModel.from_pretrained(cfg.model_id, **model_kwargs)
            self.model.to(cfg.device)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self.cfg.device)

    def _postprocess(self, arr: np.ndarray) -> np.ndarray:
        if self.cfg.dim is not None:
            if self.cfg.dim <= 0 or self.cfg.dim > arr.shape[1]:
                raise ValueError(f"Invalid dim={self.cfg.dim} for embedding shape={arr.shape}")
            arr = arr[:, : self.cfg.dim]
        if self.cfg.normalize:
            denom = np.linalg.norm(arr, axis=1, keepdims=True)
            denom = np.maximum(denom, 1e-12)
            arr = arr / denom
        return arr.astype("float32", copy=False)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype="float32")

        with torch.no_grad():
            # Preferred retrieval-model APIs.
            for method_name in ("encode_text", "get_text_features"):
                method = getattr(self.model, method_name, None)
                if method is None:
                    continue
                try:
                    inputs = self.processor(text=texts, padding=True, return_tensors="pt")
                    inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    out = method(**inputs)
                    arr = _extract_embedding_from_output(out)
                    if arr is not None:
                        return self._postprocess(arr)
                except Exception:
                    pass

            # Some remote-code embedding models expose one generic encode method.
            encode = getattr(self.model, "encode", None)
            if encode is not None:
                for kwargs in ({"texts": texts}, {"text": texts}, {"inputs": texts}):
                    try:
                        out = encode(**kwargs)
                        arr = _extract_embedding_from_output(out)
                        if arr is not None:
                            return self._postprocess(arr)
                    except Exception:
                        pass

        raise RuntimeError(
            "Could not obtain text embeddings from this model. Check the model card "
            "for its embedding API and adapt QwenEmb/qwenemb_encoder.py."
        )

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        images = [img.convert("RGB") for img in images]
        if not images:
            return np.empty((0, 0), dtype="float32")

        with torch.no_grad():
            for method_name in ("encode_image", "get_image_features"):
                method = getattr(self.model, method_name, None)
                if method is None:
                    continue
                try:
                    inputs = self.processor(images=images, return_tensors="pt")
                    inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    out = method(**inputs)
                    arr = _extract_embedding_from_output(out)
                    if arr is not None:
                        return self._postprocess(arr)
                except Exception:
                    pass

            encode = getattr(self.model, "encode", None)
            if encode is not None:
                for kwargs in ({"images": images}, {"image": images}, {"inputs": images}):
                    try:
                        out = encode(**kwargs)
                        arr = _extract_embedding_from_output(out)
                        if arr is not None:
                            return self._postprocess(arr)
                    except Exception:
                        pass

        raise RuntimeError(
            "Could not obtain image embeddings from this model. Check the model card "
            "for its embedding API and adapt QwenEmb/qwenemb_encoder.py."
        )
