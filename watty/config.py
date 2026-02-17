"""
Watty Layer 1 Configuration
Every setting in one place. The brain's parameters.
"""

import os
from pathlib import Path

# ── Identity ─────────────────────────────────────────────
SERVER_NAME = "watty"
SERVER_VERSION = "2.3.0"

# ── Paths ────────────────────────────────────────────────
WATTY_HOME = Path(os.environ.get("WATTY_HOME", Path.home() / ".watty"))
DB_PATH = WATTY_HOME / "brain.db"

# ── Embedding Model ──────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("WATTY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

# ── Search Tuning ────────────────────────────────────────
TOP_K = int(os.environ.get("WATTY_TOP_K", "10"))
RELEVANCE_THRESHOLD = float(os.environ.get("WATTY_RELEVANCE_THRESHOLD", "0.35"))
RECENCY_WEIGHT = float(os.environ.get("WATTY_RECENCY_WEIGHT", "0.15"))

# ── Chunking ─────────────────────────────────────────────
CHUNK_SIZE = int(os.environ.get("WATTY_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("WATTY_CHUNK_OVERLAP", "200"))

# ── Prompt Genome ────────────────────────────────────────
PROMPT_WEIGHT_THRESHOLD = float(os.environ.get("WATTY_PROMPT_WEIGHT", "0.4"))

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

# ── Hippocampus ──────────────────────────────────────────
# Dentate Gyrus — pattern separation
DG_SPARSITY = float(os.environ.get("WATTY_DG_SPARSITY", "0.02"))           # 2% activation (biological)
DG_SEPARATION_THRESHOLD = float(os.environ.get("WATTY_DG_SEP_THRESH", "0.85"))  # above this = too similar, orthogonalize
DG_PROJECTION_DIM = int(os.environ.get("WATTY_DG_PROJ_DIM", "2048"))       # expanded sparse space

# CA3 — associative storage + pattern completion
CA3_MAX_ASSOCIATIONS = int(os.environ.get("WATTY_CA3_MAX_ASSOC", "10"))     # recurrent connections per memory
CA3_COMPLETION_THRESHOLD = float(os.environ.get("WATTY_CA3_COMPLETION", "0.4"))  # min similarity to trigger completion
CA3_ASSOCIATION_DECAY = float(os.environ.get("WATTY_CA3_DECAY", "0.95"))    # association strength decay per cycle

# CA1 — mismatch detection
CA1_NOVELTY_THRESHOLD = float(os.environ.get("WATTY_CA1_NOVELTY", "0.3"))   # below this = novel
CA1_CONTRADICTION_THRESHOLD = float(os.environ.get("WATTY_CA1_CONTRA", "0.15"))  # below this + high keyword overlap = contradiction

# Consolidation engine
CONSOLIDATION_INTERVAL = int(os.environ.get("WATTY_CONSOL_INTERVAL", "3600"))  # seconds between consolidation cycles
CONSOLIDATION_REPLAY_COUNT = int(os.environ.get("WATTY_CONSOL_REPLAY", "20"))  # memories replayed per cycle
CONSOLIDATION_PROMOTION_THRESHOLD = int(os.environ.get("WATTY_CONSOL_PROMOTE", "5"))  # access count to promote to consolidated
CONSOLIDATION_DECAY_DAYS = int(os.environ.get("WATTY_CONSOL_DECAY", "30"))  # days before unaccessed episodic memories decay

# Memory tiers
TIER_EPISODIC = "episodic"     # fast capture, temporary, hippocampus-dependent
TIER_CONSOLIDATED = "consolidated"  # slow-learned, permanent, hippocampus-independent
TIER_SCHEMA = "schema"         # abstract patterns extracted across episodes

# ── Chestahedron ────────────────────────────────────────
CHESTAHEDRON_SIGNAL_DIM = 48
CHESTAHEDRON_RESERVOIR_SIZE = 64
CHESTAHEDRON_CIRCULATIONS = 3
CHESTAHEDRON_GEO_WEIGHT = float(os.environ.get("WATTY_CHESTA_GEO_WEIGHT", "0.15"))
CHESTAHEDRON_COHERENCE_CONTRA = float(os.environ.get("WATTY_CHESTA_CONTRA", "0.1"))
CHESTAHEDRON_MIGRATION_BATCH = 100

# ── Chestahedron Plasticity ───────────────────────────────
CHESTAHEDRON_PLASTICITY_LR = float(os.environ.get("WATTY_CHESTA_LR", "0.001"))
CHESTAHEDRON_HOMEOSTATIC_RATE = float(os.environ.get("WATTY_CHESTA_HOMEO", "0.0005"))
CHESTAHEDRON_INTRINSIC_LR = float(os.environ.get("WATTY_CHESTA_INTRINSIC_LR", "0.0001"))
CHESTAHEDRON_WOUT_SPECTRAL_MAX = float(os.environ.get("WATTY_CHESTA_WOUT_MAX", "2.0"))

# ── Navigator (Layer 2) ────────────────────────────────────
NAVIGATOR_MAX_CIRCULATIONS = int(os.environ.get("WATTY_NAV_MAX_CIRC", "3"))
NAVIGATOR_DECAY = float(os.environ.get("WATTY_NAV_DECAY", "0.7"))
NAVIGATOR_GEO_EDGE_WEIGHT = float(os.environ.get("WATTY_NAV_GEO_WEIGHT", "1.0"))
NAVIGATOR_FELT_MODULATION = os.environ.get("WATTY_NAV_FELT_MOD", "1") == "1"
NAVIGATOR_MIN_ACTIVATION = float(os.environ.get("WATTY_NAV_MIN_ACT", "0.05"))
NAVIGATOR_SEED_TOP_N = int(os.environ.get("WATTY_NAV_SEED_N", "20"))
NAVIGATOR_SEED_THRESHOLD = float(os.environ.get("WATTY_NAV_SEED_THRESH", "0.35"))

# ── Knowledge Graph ────────────────────────────────────────
KG_OLLAMA_URL = os.environ.get("WATTY_KG_OLLAMA_URL", "http://localhost:11434")
KG_OLLAMA_MODEL = os.environ.get("WATTY_KG_OLLAMA_MODEL", "qwen2.5:7b")
KG_EXTRACTION_TIMEOUT = int(os.environ.get("WATTY_KG_EXTRACT_TIMEOUT", "60"))
KG_MAX_ENTITIES_PER_CHUNK = int(os.environ.get("WATTY_KG_MAX_ENTITIES", "15"))
KG_MERGE_SIMILARITY_THRESHOLD = float(os.environ.get("WATTY_KG_MERGE_SIM", "0.92"))
KG_TRAVERSAL_MAX_HOPS = int(os.environ.get("WATTY_KG_MAX_HOPS", "2"))
KG_RRF_K = int(os.environ.get("WATTY_KG_RRF_K", "60"))
KG_ENABLED = os.environ.get("WATTY_KG_ENABLED", "1") == "1"

# ── Reflection Engine ──────────────────────────────────────
REFLECTION_AUTO_PROMOTE_THRESHOLD = int(os.environ.get("WATTY_REFLECT_PROMOTE", "3"))
REFLECTION_MAX_LENGTH = int(os.environ.get("WATTY_REFLECT_MAX_LEN", "200"))
REFLECTION_TOP_K = int(os.environ.get("WATTY_REFLECT_TOP_K", "5"))
REFLECTION_SIMILARITY_THRESHOLD = float(os.environ.get("WATTY_REFLECT_SIM_THRESH", "0.35"))
REFLECTION_ENABLED = os.environ.get("WATTY_REFLECT_ENABLED", "1") == "1"

# ── Evaluation Framework ──────────────────────────────────
EVAL_ENABLED = os.environ.get("WATTY_EVAL_ENABLED", "1") == "1"
EVAL_ALERT_WINDOW_DAYS = int(os.environ.get("WATTY_EVAL_ALERT_WINDOW", "7"))
EVAL_TREND_SHORT_DAYS = 30
EVAL_TREND_LONG_DAYS = 90
EVAL_ALERT_PRECISION_DROP = float(os.environ.get("WATTY_EVAL_PRECISION_DROP", "0.15"))
EVAL_ALERT_CONTRADICTION_RISE = float(os.environ.get("WATTY_EVAL_CONTRA_RISE", "0.20"))
EVAL_ALERT_SUCCESS_DROP = float(os.environ.get("WATTY_EVAL_SUCCESS_DROP", "0.10"))

# ── A2A Protocol ──────────────────────────────────────────
A2A_ENABLED = os.environ.get("WATTY_A2A_ENABLED", "1") == "1"
A2A_MAX_CONCURRENT = int(os.environ.get("WATTY_A2A_MAX_CONCURRENT", "5"))
A2A_TASK_TTL_HOURS = int(os.environ.get("WATTY_A2A_TASK_TTL", "72"))
A2A_AUTH_TOKEN = os.environ.get("WATTY_A2A_AUTH_TOKEN", "")
A2A_RATE_LIMIT_PER_MINUTE = int(os.environ.get("WATTY_A2A_RATE_LIMIT", "30"))

# ── Voice Engine ──────────────────────────────────────────
VOICE_MODELS_DIR = WATTY_HOME / "voice" / "models"

def ensure_home():
    """Create the Watty home directory if it doesn't exist."""
    WATTY_HOME.mkdir(parents=True, exist_ok=True)
