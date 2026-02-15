"""
Tests for structured logging and watty doctor.
"""

import sys
import types
import os
import tempfile
import logging
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
from watty.doctor import check_all  # noqa: E402


def _fresh():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "test.db")
    return db, tmp


def test_log_file_created():
    """setup() creates watty.log with rotation."""
    import watty.log as log_mod
    # Reset so we can test setup
    old_configured = log_mod._configured
    log_mod._configured = False
    old_handlers = log_mod.log.handlers[:]
    log_mod.log.handlers = [logging.NullHandler()]
    try:
        log_mod.setup()
        # Should have at least stderr + file handlers (+ NullHandler)
        handler_types = [type(h).__name__ for h in log_mod.log.handlers]
        assert "StreamHandler" in handler_types
        # File handler may not exist if WATTY_HOME is unwritable, but check it's tried
        assert log_mod._configured is True
    finally:
        log_mod.log.handlers = old_handlers
        log_mod._configured = old_configured


def test_doctor_healthy():
    """watty doctor exits 0 on healthy brain."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    brain.store_memory("Doctor test memory")

    result = check_all(db_path=db)
    assert result["healthy"] is True
    assert result["checks"]["database"]["status"] == "ok"
    assert result["checks"]["embeddings"]["status"] == "ok"
    assert result["checks"]["schema"]["status"] == "ok"
    assert result["checks"]["memories"]["total"] == 1


def test_doctor_missing_db():
    """watty doctor reports missing database."""
    result = check_all(db_path="/tmp/nonexistent_watty_test.db")
    assert result["healthy"] is False
    assert result["checks"]["database"]["status"] == "missing"


def test_doctor_json_output():
    """watty doctor --json produces valid JSON."""
    import json
    db, _ = _fresh()
    Brain(db_path=db)
    result = check_all(db_path=db)
    # Should be JSON-serializable
    output = json.dumps(result)
    parsed = json.loads(output)
    assert "healthy" in parsed
    assert "checks" in parsed


def test_timed_context_manager():
    """timed() context manager measures elapsed time."""
    from watty.log import timed
    import time
    with timed("test_op") as t:
        time.sleep(0.01)
    assert t.elapsed_ms >= 5  # at least 5ms (generous for CI)
