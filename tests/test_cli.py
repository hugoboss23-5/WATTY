"""
Tests for unified CLI.
"""

import sys
import types
import subprocess
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


def test_version():
    """watty --version prints version."""
    from watty.config import SERVER_VERSION
    result = subprocess.run(
        [sys.executable, "-m", "watty.cli", "--version"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert SERVER_VERSION in result.stdout


def test_help_fits_screen():
    """watty --help returns help text."""
    result = subprocess.run(
        [sys.executable, "-m", "watty.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "import" in result.stdout
    assert "doctor" in result.stdout
    assert "stats" in result.stdout
    assert "backup" in result.stdout
    assert "restore" in result.stdout


def test_subcommand_help():
    """Each subcommand shows help without error."""
    for cmd in ["serve", "import", "doctor", "stats", "backup", "restore"]:
        result = subprocess.run(
            [sys.executable, "-m", "watty.cli", cmd, "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{cmd} --help failed: {result.stderr}"


def test_unknown_subcommand():
    """Unknown subcommand exits with error."""
    result = subprocess.run(
        [sys.executable, "-m", "watty.cli", "nonexistent"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
