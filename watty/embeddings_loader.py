"""
Watty Embedding Loader
Selects the best available backend: onnx (light) or torch (heavy).
The rest of the codebase imports from here.
"""

import sys
import numpy as np

from watty.config import EMBEDDING_BACKEND

_embed_fn = None
_cosine_fn = None


def _load_onnx():
    from watty.embeddings_onnx import embed_text, cosine_similarity
    return embed_text, cosine_similarity


def _load_torch():
    from watty.embeddings import embed_text, cosine_similarity
    return embed_text, cosine_similarity


def _resolve():
    global _embed_fn, _cosine_fn
    if _embed_fn is not None:
        return

    if EMBEDDING_BACKEND == "onnx":
        _embed_fn, _cosine_fn = _load_onnx()
    elif EMBEDDING_BACKEND == "torch":
        _embed_fn, _cosine_fn = _load_torch()
    elif EMBEDDING_BACKEND == "auto":
        try:
            _embed_fn, _cosine_fn = _load_onnx()
            print("[Watty] Using ONNX backend (~100MB)", file=sys.stderr, flush=True)
        except ImportError:
            try:
                _embed_fn, _cosine_fn = _load_torch()
                print("[Watty] Using PyTorch backend (~2GB)", file=sys.stderr, flush=True)
            except ImportError:
                raise ImportError(
                    "No embedding backend available. Install one:\n"
                    "  pip install watty[onnx]    # Recommended: ~100MB\n"
                    "  pip install watty[torch]   # Full: ~2GB, includes PyTorch"
                )
    else:
        raise ValueError(f"Unknown WATTY_EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")


def embed_text(text: str) -> np.ndarray:
    _resolve()
    return _embed_fn(text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    _resolve()
    return _cosine_fn(a, b)
