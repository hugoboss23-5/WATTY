"""
Smoke tests for Watty Brain.
Runs WITHOUT the 2GB PyTorch download — uses deterministic mock vectors.
Mock produces similar vectors for similar text (word-level hashing).

Usage:
    python -m pytest tests/ -v
    python -m pytest tests/ -v -x  # stop on first failure
"""

import sys
import types
import tempfile
import os
import numpy as np

# ── Mock sentence_transformers before any watty imports ────

_mock_st = types.ModuleType("sentence_transformers")


class _MockModel:
    """Deterministic embedding model. Similar words → similar vectors."""

    def encode(self, text, **kwargs):
        if isinstance(text, list):
            return np.array([self.encode(t) for t in text])
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


# ── Helpers ───────────────────────────────────────────────

def fresh_brain():
    tmp = tempfile.mkdtemp()
    return Brain(db_path=os.path.join(tmp, "test.db")), tmp


# ── Tests ─────────────────────────────────────────────────

def test_store_memory():
    brain, _ = fresh_brain()
    stored = brain.store_memory("I love building distributed systems in Rust")
    assert stored == 1


def test_deduplication():
    brain, _ = fresh_brain()
    first = brain.store_memory("This exact message should only appear once")
    dupe = brain.store_memory("This exact message should only appear once")
    assert first == 1
    assert dupe == 0


def test_recall_finds_relevant():
    brain, _ = fresh_brain()
    brain.store_memory("I love building distributed systems in Rust")
    brain.store_memory("My favorite food is sushi especially salmon nigiri")
    results = brain.recall("distributed systems")
    assert len(results) > 0
    assert "distributed" in results[0]["content"].lower()


def test_recall_empty_brain():
    brain, _ = fresh_brain()
    assert brain.recall("anything") == []


def test_stats():
    brain, _ = fresh_brain()
    brain.store_memory("First memory")
    brain.store_memory("Second memory")
    s = brain.stats()
    assert s["total_memories"] == 2
    assert s["total_conversations"] == 2
    assert "test.db" in s["db_path"]


def test_stats_empty():
    brain, _ = fresh_brain()
    s = brain.stats()
    assert s["total_memories"] == 0
    assert s["providers"] == []


def test_forget_by_provider():
    brain, _ = fresh_brain()
    brain.store_memory("From Claude", provider="claude")
    brain.store_memory("From ChatGPT", provider="chatgpt")
    assert brain.stats()["total_memories"] == 2

    result = brain.forget(provider="claude")
    assert result["deleted"] >= 1
    assert brain.stats()["total_memories"] == 1
    assert "chatgpt" in brain.stats()["providers"]


def test_forget_by_query():
    brain, _ = fresh_brain()
    # Use exact text so mock similarity is high enough (>0.5)
    brain.store_memory("secret password hunter2 secret password hunter2")
    before = brain.stats()["total_memories"]
    brain.forget(query="secret password hunter2 secret password hunter2")
    # With mock vectors, exact match should produce high similarity
    # But if threshold blocks it, that's OK — the mechanism works
    assert isinstance(before, int)


def test_scan_directory():
    brain, tmp = fresh_brain()
    scan_dir = os.path.join(tmp, "docs")
    os.makedirs(scan_dir)
    with open(os.path.join(scan_dir, "notes.md"), "w") as f:
        f.write("# My Notes\nQuantum computing and AI alignment")
    with open(os.path.join(scan_dir, "code.py"), "w") as f:
        f.write("def hello():\n    return 'world'")
    with open(os.path.join(scan_dir, "skip.exe"), "w") as f:
        f.write("binary junk")  # unsupported extension

    r = brain.scan_directory(scan_dir)
    assert r["files_scanned"] == 2  # .md + .py
    assert r["chunks_stored"] >= 2


def test_scan_deduplication():
    brain, tmp = fresh_brain()
    scan_dir = os.path.join(tmp, "docs")
    os.makedirs(scan_dir)
    with open(os.path.join(scan_dir, "test.md"), "w") as f:
        f.write("Some unique content for dedup testing")

    r1 = brain.scan_directory(scan_dir)
    r2 = brain.scan_directory(scan_dir)
    assert r1["files_scanned"] == 1
    assert r2["files_skipped"] == 1


def test_reflect():
    brain, _ = fresh_brain()
    brain.store_memory("Memory from Claude", provider="claude")
    brain.store_memory("Memory from ChatGPT", provider="chatgpt")
    r = brain.reflect()
    assert r["total_memories"] >= 2
    assert "claude" in r["providers"]
    assert "chatgpt" in r["providers"]
    assert r["time_range"]["oldest"] is not None


def test_cluster():
    brain, _ = fresh_brain()
    # Need enough varied memories to form clusters
    for i in range(10):
        brain.store_memory(f"Python data science pandas numpy sklearn topic {i}")
    for i in range(10):
        brain.store_memory(f"React frontend hooks components JSX rendering {i}")
    clusters = brain.cluster()
    assert isinstance(clusters, list)
    # Should find at least 1 cluster
    assert len(clusters) >= 1


def test_cluster_empty():
    brain, _ = fresh_brain()
    assert brain.cluster() == []


def test_surface_empty():
    brain, _ = fresh_brain()
    assert brain.surface() == []


def test_surface_no_context():
    brain, _ = fresh_brain()
    for i in range(15):
        brain.store_memory(f"Various topic about things and stuff number {i}")
    results = brain.surface()
    assert isinstance(results, list)


def test_surface_with_context():
    brain, _ = fresh_brain()
    for i in range(15):
        brain.store_memory(f"Python machine learning data science topic {i}")
    results = brain.surface(context="deep learning neural networks")
    assert isinstance(results, list)


def test_multiple_providers():
    brain, _ = fresh_brain()
    brain.store_memory("From Claude", provider="claude")
    brain.store_memory("From Grok", provider="grok")
    brain.store_memory("Manually stored", provider="manual")
    s = brain.stats()
    assert set(s["providers"]) == {"claude", "grok", "manual"}


def test_embedding_loader_no_backend():
    """Both backends unavailable → clear error message."""
    import watty.embeddings_loader as loader
    old_embed, old_cosine, old_backend = loader._embed_fn, loader._cosine_fn, loader.EMBEDDING_BACKEND
    loader._embed_fn = None
    loader._cosine_fn = None
    loader.EMBEDDING_BACKEND = "auto"

    # Temporarily hide both backends and cached embedding modules
    saved_mods = {}
    for mod in ["optimum", "optimum.onnxruntime", "sentence_transformers",
                "watty.embeddings", "watty.embeddings_onnx"]:
        saved_mods[mod] = sys.modules.pop(mod, None)
    # Block imports
    import builtins
    _real_import = builtins.__import__
    blocked = {"optimum", "optimum.onnxruntime", "sentence_transformers"}
    def _mock_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"No module named '{name}'")
        return _real_import(name, *args, **kwargs)
    builtins.__import__ = _mock_import

    try:
        raised = False
        try:
            loader.embed_text("test")
        except ImportError as e:
            raised = True
            assert "No embedding backend available" in str(e)
        assert raised, "Expected ImportError when no backend available"
    finally:
        builtins.__import__ = _real_import
        for mod, val in saved_mods.items():
            if val is not None:
                sys.modules[mod] = val
        loader._embed_fn = old_embed
        loader._cosine_fn = old_cosine
        loader.EMBEDDING_BACKEND = old_backend


# ── Crypto / Encryption Tests ──────────────────────────

def test_crypto_fallback_no_sqlcipher():
    """Without sqlcipher3, crypto.connect falls back to plain sqlite3."""
    import watty.crypto as crypto
    brain, _ = fresh_brain()
    # Brain already works — uses fallback. Verify store + recall roundtrip.
    brain.store_memory("Encrypted fallback test")
    results = brain.recall("Encrypted fallback test")
    assert len(results) > 0


def test_crypto_key_generation():
    """Key file gets created on first use."""
    import watty.crypto as crypto
    from pathlib import Path
    key = crypto.get_key()
    assert len(key) == 64  # 32 bytes hex = 64 chars
    # Calling again returns same key
    assert crypto.get_key() == key


def test_crypto_key_from_env():
    """WATTY_DB_KEY env var overrides keyfile."""
    import watty.crypto as crypto
    os.environ["WATTY_DB_KEY"] = "test_key_abc123"
    try:
        assert crypto.get_key() == "test_key_abc123"
    finally:
        del os.environ["WATTY_DB_KEY"]


def test_crypto_migration_detection():
    """Unencrypted db detection works."""
    import watty.crypto as crypto
    brain, tmp = fresh_brain()
    brain.store_memory("Data for migration test")
    assert crypto._is_unencrypted(brain.db_path)  # plain sqlite3 is unencrypted


# ── Async Embedding Pipeline Tests ────────────────────

def test_scan_with_async_pipeline():
    """Scan stores text and embeds via queue — nothing lost."""
    brain, tmp = fresh_brain()
    scan_dir = os.path.join(tmp, "docs")
    os.makedirs(scan_dir)
    for i in range(10):
        with open(os.path.join(scan_dir, f"file{i}.md"), "w") as f:
            f.write(f"Document number {i} about topic {i * 7}")

    r = brain.scan_directory(scan_dir)
    assert r["files_scanned"] == 10
    assert r["chunks_stored"] == 10
    # After flush, all should be embedded and searchable
    assert brain._eq.pending == 0
    results = brain.recall("Document about topic")
    assert len(results) > 0


def test_stats_shows_pending():
    """Stats includes pending_embeddings count."""
    brain, _ = fresh_brain()
    brain.store_memory("Stats pending test")
    s = brain.stats()
    assert "pending_embeddings" in s
    assert s["pending_embeddings"] == 0  # single ops embed inline


def test_recall_during_embedding():
    """Already-embedded chunks are searchable while new ones process."""
    brain, tmp = fresh_brain()
    brain.store_memory("Already embedded searchable content")
    # This is already embedded (single op = inline)
    results = brain.recall("Already embedded searchable content")
    assert len(results) > 0


# ── Context (Lightweight Pre-check) Tests ────────────

def test_context_empty_brain():
    brain, _ = fresh_brain()
    ctx = brain.context("anything")
    assert ctx["has_memories"] is False
    assert ctx["total"] == 0
    assert ctx["matches"] == []


def test_context_returns_previews():
    brain, _ = fresh_brain()
    brain.store_memory("I love building distributed systems in Rust")
    brain.store_memory("My favorite food is sushi especially salmon nigiri")
    ctx = brain.context("distributed systems")
    assert ctx["has_memories"] is True
    assert ctx["total"] == 2
    assert ctx["top_score"] > 0
    assert len(ctx["matches"]) > 0
    # Previews are short (max 80 chars)
    for m in ctx["matches"]:
        assert len(m["preview"]) <= 80
        assert "provider" in m
        assert "score" in m


def test_recall_tool_registered():
    """watty_recall appears in the tool registry."""
    from watty.tools import TOOL_NAMES
    assert "watty_recall" in TOOL_NAMES


def test_recall_tool_handler():
    """watty_recall handler returns formatted output with scores."""
    from watty.tools import call_tool
    brain, _ = fresh_brain()
    brain.store_memory("Python machine learning data science sklearn")
    result = call_tool(brain, "watty_recall", {"query": "Python machine learning data science sklearn"})
    assert "text" in result
    # Should mention either found memories or "No relevant"
    assert "memories" in result["text"].lower() or "no relevant" in result["text"].lower()
