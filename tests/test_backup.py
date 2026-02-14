"""
Tests for backup and restore CLI.
"""

import sys
import types
import json
import os
import tempfile
import tarfile
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
from watty.backup import backup, restore  # noqa: E402


def _setup():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "test.db")
    brain = Brain(db_path=db)
    return brain, tmp, db


def test_backup_creates_archive():
    brain, tmp, db = _setup()
    brain.store_memory("Backup test memory one")
    brain.store_memory("Backup test memory two")

    out = os.path.join(tmp, "backup.tar.gz")
    result = backup(output=out, db_path=db)
    assert os.path.exists(result)

    with tarfile.open(result, "r:gz") as tar:
        names = tar.getnames()
        assert "brain.db" in names
        assert "manifest.json" in names
        manifest = json.loads(tar.extractfile("manifest.json").read())
        assert manifest["memories"] == 2


def test_backup_restore_roundtrip():
    brain, tmp, db = _setup()
    brain.store_memory("Important data that must survive")
    brain.store_memory("Second important memory")

    out = os.path.join(tmp, "backup.tar.gz")
    backup(output=out, db_path=db)

    # Corrupt the db
    with open(db, "w") as f:
        f.write("corrupted")

    # Restore
    manifest = restore(out, force=True, db_path=db)
    assert manifest["memories"] == 2

    # Verify data survived
    restored = Brain(db_path=db)
    assert restored.stats()["total_memories"] == 2


def test_manifest_validation():
    brain, tmp, db = _setup()
    brain.store_memory("Manifest test")

    out = os.path.join(tmp, "backup.tar.gz")
    backup(output=out, db_path=db)

    with tarfile.open(out, "r:gz") as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read())
        assert "version" in manifest
        assert "memories" in manifest
        assert "created" in manifest
        assert isinstance(manifest["providers"], list)
