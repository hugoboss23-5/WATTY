"""
Watty Embedding Engine
Converts text into semantic vectors.
Understands MEANING, not keywords.
"""

import os
import sys
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sentence_transformers import SentenceTransformer
from watty.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Watty] Loading embedding model: {EMBEDDING_MODEL}", file=sys.stderr, flush=True)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[Watty] Model loaded. Dimension: {EMBEDDING_DIMENSION}", file=sys.stderr, flush=True)
    return _model


def embed_text(text: str) -> np.ndarray:
    model = get_model()
    return np.array(model.encode(text, normalize_embeddings=True), dtype=np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    model = get_model()
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
