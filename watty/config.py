"""
Watty Layer 1 Configuration
Every setting in one place. The brain's parameters.
"""

import os
from pathlib import Path

# ── Identity ─────────────────────────────────────────────
SERVER_NAME = "watty"
SERVER_VERSION = "1.2.0"

# ── Paths ────────────────────────────────────────────────
WATTY_HOME = Path(os.environ.get("WATTY_HOME", Path.home() / ".watty"))
DB_PATH = WATTY_HOME / "brain.db"

# ── Embedding Model ──────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("WATTY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384
EMBEDDING_BACKEND = os.environ.get("WATTY_EMBEDDING_BACKEND", "auto")  # auto | onnx | torch

# ── Search Tuning ────────────────────────────────────────
TOP_K = int(os.environ.get("WATTY_TOP_K", "10"))
RELEVANCE_THRESHOLD = float(os.environ.get("WATTY_RELEVANCE_THRESHOLD", "0.35"))
RECENCY_WEIGHT = float(os.environ.get("WATTY_RECENCY_WEIGHT", "0.15"))

# ── Chunking ─────────────────────────────────────────────
CHUNK_SIZE = int(os.environ.get("WATTY_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("WATTY_CHUNK_OVERLAP", "200"))

# ── Scanner ──────────────────────────────────────────────
SCAN_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".log",
    ".py", ".js", ".ts", ".swift", ".rs",
    ".html", ".css", ".yaml", ".yml", ".toml",
    ".sh", ".bat", ".ps1",
}
SCAN_MAX_FILE_SIZE = 1_000_000  # 1MB per file
SCAN_IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".watty", ".cache", "dist", "build", ".egg-info",
}

# ── Clustering ───────────────────────────────────────────
CLUSTER_MIN_MEMORIES = 5       # minimum memories before clustering kicks in
CLUSTER_SIMILARITY_THRESHOLD = 0.65  # how similar memories must be to group

# ── Surface (proactive) ─────────────────────────────────
SURFACE_TOP_K = 3              # how many proactive insights to return
SURFACE_NOVELTY_WEIGHT = 0.3   # favor surprising connections

def ensure_home():
    """Create the Watty home directory if it doesn't exist."""
    WATTY_HOME.mkdir(parents=True, exist_ok=True)
