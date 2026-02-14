"""
Watty ONNX Embedding Engine
Same interface as embeddings.py but uses onnxruntime instead of PyTorch.
~80MB instead of ~2GB. No GPU dependency.
"""

import sys
import numpy as np
import optimum.onnxruntime  # fail fast if not installed
from watty.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

_session = None
_tokenizer = None


def _load():
    global _session, _tokenizer
    if _session is not None:
        return
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    print(f"[Watty] Loading ONNX model: {EMBEDDING_MODEL}", file=sys.stderr, flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    _session = ORTModelForFeatureExtraction.from_pretrained(EMBEDDING_MODEL, export=True)
    print(f"[Watty] ONNX model loaded. Dimension: {EMBEDDING_DIMENSION}", file=sys.stderr, flush=True)


def embed_text(text: str) -> np.ndarray:
    _load()
    inputs = _tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
    outputs = _session(**inputs)
    # Mean pooling over token embeddings, then normalize
    mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state
    pooled = (embeddings * mask[..., np.newaxis]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    vec = pooled[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
