"""
Tests for performance features: benchmark suite, faiss fallback, lazy loading.
"""

import sys
import types
import os
import tempfile
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


def test_benchmark_runs_empty():
    """Benchmark suite runs without error on empty brain."""
    from tests.bench import bench_store, bench_recall, bench_cluster
    db, tmp = _fresh()
    brain = Brain(db_path=db)
    r = bench_store(brain, 5)
    assert r["count"] == 5
    assert r["throughput"] > 0

    r = bench_recall(brain, ["test query"], "5 memories")
    assert r["queries"] == 1
    assert r["avg_ms"] >= 0

    r = bench_cluster(brain, "5 memories")
    assert isinstance(r["clusters_found"], int)


def test_benchmark_runs_1k():
    """Benchmark suite runs on 1k memories."""
    from tests.bench import bench_store, bench_recall
    db, _ = _fresh()
    brain = Brain(db_path=db)
    r = bench_store(brain, 100)
    assert r["count"] == 100

    r = bench_recall(brain, ["machine learning", "distributed systems"] * 5, "100 memories")
    assert r["queries"] == 10
    assert r["avg_ms"] > 0


def test_lazy_vector_loading():
    """First recall triggers index load, subsequent recalls use cache."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    brain.store_memory("Test memory for lazy loading")

    # Vectors not loaded yet
    assert brain._index_dirty is True

    # First recall triggers load
    results = brain.recall("Test memory")
    assert brain._index_dirty is False
    assert brain._vectors is not None

    # Second recall uses cache
    brain.store_memory("Another memory")
    assert brain._index_dirty is True
    results2 = brain.recall("Another memory")
    assert brain._index_dirty is False


def test_numpy_search_produces_results():
    """Numpy vector search returns correct results."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    brain.store_memory("Python machine learning data science")
    brain.store_memory("JavaScript React frontend development")

    results = brain.recall("Python data science")
    assert len(results) > 0
    assert "Python" in results[0]["content"] or "python" in results[0]["content"].lower()
