"""
Tests for conversation importers.
Verifies ChatGPT, Claude, and generic JSON formats parse and store correctly.
"""

import sys
import types
import json
import tempfile
import os
import zipfile
import numpy as np

# ── Mock sentence_transformers before any watty imports ────

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
from watty.importers.chatgpt import import_chatgpt  # noqa: E402
from watty.importers.claude import import_claude  # noqa: E402
from watty.importers.generic import import_json  # noqa: E402


def _tmp_brain():
    """Patch Brain to use temp db."""
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "test.db")
    import watty.importers.chatgpt as cg
    import watty.importers.claude as cl
    import watty.importers.generic as gn
    original_brain_cls = Brain.__init__.__code__
    # Monkey-patch Brain default to use temp db
    old_init = Brain.__init__
    def patched_init(self, db_path=None):
        old_init(self, db_path=db_path or db)
    Brain.__init__ = patched_init
    return tmp, db, old_init


def _restore_brain(old_init):
    Brain.__init__ = old_init


# ── ChatGPT Import Tests ─────────────────────────────────

def test_chatgpt_import():
    tmp, db, old_init = _tmp_brain()
    try:
        conversations = [{
            "id": "conv-1",
            "title": "Test Chat",
            "mapping": {
                "node-1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Hello ChatGPT"]}}},
                "node-2": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Hi there!"]}}},
                "node-3": {"message": {"author": {"role": "system"}, "content": {"parts": ["System prompt"]}}},
                "node-4": {"message": None},
            }
        }]
        json_path = os.path.join(tmp, "conversations.json")
        with open(json_path, "w") as f:
            json.dump(conversations, f)

        # Create ZIP
        zip_path = os.path.join(tmp, "export.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", json.dumps(conversations))

        result = import_chatgpt(zip_path)
        assert result["conversations"] == 1
        assert result["chunks"] >= 2  # user + assistant (system skipped)

        brain = Brain(db_path=db)
        assert brain.stats()["total_memories"] >= 2
        assert "chatgpt" in brain.stats()["providers"]
    finally:
        _restore_brain(old_init)


def test_chatgpt_dedup():
    tmp, db, old_init = _tmp_brain()
    try:
        conversations = [{
            "id": "conv-1", "title": "Dedup test",
            "mapping": {
                "n1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Same message twice"]}}},
            }
        }]
        zip_path = os.path.join(tmp, "export.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", json.dumps(conversations))

        r1 = import_chatgpt(zip_path)
        r2 = import_chatgpt(zip_path)
        assert r1["chunks"] == 1
        assert r2["chunks"] == 0  # deduped
    finally:
        _restore_brain(old_init)


# ── Claude Import Tests ──────────────────────────────────

def test_claude_import():
    tmp, db, old_init = _tmp_brain()
    try:
        conversations = [{
            "uuid": "conv-uuid-1",
            "name": "Test Claude Chat",
            "chat_messages": [
                {"sender": "human", "text": "What is Rust?"},
                {"sender": "assistant", "text": "Rust is a systems programming language."},
            ]
        }]
        path = os.path.join(tmp, "claude_export.json")
        with open(path, "w") as f:
            json.dump(conversations, f)

        result = import_claude(path)
        assert result["conversations"] == 1
        assert result["chunks"] >= 2

        brain = Brain(db_path=db)
        assert "claude" in brain.stats()["providers"]
    finally:
        _restore_brain(old_init)


# ── Generic JSON Import Tests ────────────────────────────

def test_generic_flat_messages():
    tmp, db, old_init = _tmp_brain()
    try:
        messages = [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language"},
        ]
        path = os.path.join(tmp, "messages.json")
        with open(path, "w") as f:
            json.dump(messages, f)

        result = import_json(path, provider="test")
        assert result["conversations"] == 1
        assert result["chunks"] >= 2
    finally:
        _restore_brain(old_init)


def test_generic_conversation_array():
    tmp, db, old_init = _tmp_brain()
    try:
        data = [
            {"id": "c1", "title": "Chat 1", "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi back"},
            ]},
            {"id": "c2", "title": "Chat 2", "messages": [
                {"role": "user", "content": "Goodbye"},
            ]},
        ]
        path = os.path.join(tmp, "convos.json")
        with open(path, "w") as f:
            json.dump(data, f)

        result = import_json(path)
        assert result["conversations"] == 2
        assert result["chunks"] >= 3
    finally:
        _restore_brain(old_init)


def test_import_nonexistent_file():
    result = import_chatgpt("/nonexistent/path/file.zip")
    assert "error" in result
    result = import_claude("/nonexistent/path/file.json")
    assert "error" in result
    result = import_json("/nonexistent/path/file.json")
    assert "error" in result


def test_chatgpt_empty_messages():
    tmp, db, old_init = _tmp_brain()
    try:
        conversations = [{
            "id": "empty", "title": "Empty",
            "mapping": {
                "n1": {"message": {"author": {"role": "user"}, "content": {"parts": [""]}}},
                "n2": {"message": {"author": {"role": "user"}, "content": {"parts": []}}},
            }
        }]
        zip_path = os.path.join(tmp, "empty.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", json.dumps(conversations))

        result = import_chatgpt(zip_path)
        assert result["chunks"] == 0
    finally:
        _restore_brain(old_init)
