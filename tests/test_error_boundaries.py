"""
Tests for error boundaries and graceful degradation.
"""

import sys
import types
import os
import tempfile
import sqlite3
import numpy as np

if "sentence_transformers" not in sys.modules:
    _mock_st = types.ModuleType("sentence_transformers")
    class _MockModel:
        def encode(self, text, **kwargs):
            words = text.lower().split()
            vec = np.zeros(384, dtype=np.float32)
            for w in words:
                np.random.seed(hash(w) % (2**31))
                vec += np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            return (vec / norm) if norm > 0 else vec
    _mock_st.SentenceTransformer = lambda *a, **k: _MockModel()
    sys.modules["sentence_transformers"] = _mock_st

from watty.brain import Brain  # noqa: E402


def _fresh():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "test.db")
    return db, tmp


def test_store_with_broken_embedding():
    """If embedding fails for a chunk, store it with null vector."""
    db, _ = _fresh()
    brain = Brain(db_path=db)

    # Temporarily break embeddings
    import watty.embeddings_loader as loader
    old_fn = loader._embed_fn
    def broken_embed(text):
        raise RuntimeError("GPU on fire")
    loader._embed_fn = broken_embed

    try:
        chunks = brain.store_memory("This should still be stored despite embedding failure")
        assert chunks == 1  # stored, just without vector
        assert brain.stats()["total_memories"] == 1
    finally:
        loader._embed_fn = old_fn


def test_recall_with_corrupt_vector():
    """Recall skips corrupt vectors and returns remaining results."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    brain.store_memory("Valid memory with good vector")

    # Insert a corrupt vector directly
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO conversations (provider, created_at) VALUES (?, ?)",
        ("test", "2025-01-01T00:00:00+00:00"),
    )
    # Wrong dimension vector (192 instead of 384)
    bad_vec = np.zeros(192, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO chunks (conversation_id, role, content, chunk_index, embedding, created_at, provider, content_hash, source_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (2, "user", "Corrupt vector memory", 0, bad_vec, "2025-01-01T00:00:00+00:00", "test", "corrupt123", "conversation"),
    )
    conn.commit()
    conn.close()

    brain._index_dirty = True
    # Should not crash — skips corrupt vector, returns valid results
    results = brain.recall("Valid memory")
    assert isinstance(results, list)


def test_fts_fallback_search():
    """When vectors are unavailable, FTS/LIKE search still works."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    brain.store_memory("Python machine learning deep learning neural networks")

    # Clear all vectors to simulate no-embedding scenario
    conn = sqlite3.connect(db)
    conn.execute("UPDATE chunks SET embedding = NULL")
    conn.commit()
    conn.close()
    brain._index_dirty = True

    results = brain.recall("Python machine learning")
    # Should find via FTS or LIKE fallback
    assert isinstance(results, list)
    assert len(results) > 0


def test_re_embed_null_chunks():
    """re_embed() processes chunks with null vectors."""
    db, _ = _fresh()
    brain = Brain(db_path=db)

    # Insert a chunk without embedding
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO conversations (provider, created_at) VALUES (?, ?)",
        ("test", "2025-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO chunks (conversation_id, role, content, chunk_index, embedding, created_at, provider, content_hash, source_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "user", "Needs re-embedding", 0, None, "2025-01-01T00:00:00+00:00", "test", "reembed123", "conversation"),
    )
    conn.commit()
    conn.close()

    count = brain.re_embed()
    assert count == 1

    # Now it should have an embedding
    vr = brain.validate_vectors()
    assert vr["null"] == 0
    assert vr["valid"] >= 1


def test_validate_vectors():
    """validate_vectors() detects corrupt and null vectors."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    brain.store_memory("Valid memory one")
    brain.store_memory("Valid memory two")

    result = brain.validate_vectors()
    assert result["valid"] == 2
    assert result["corrupt"] == 0
    assert result["null"] == 0
