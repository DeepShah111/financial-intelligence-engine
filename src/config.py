"""
Configuration settings for the Financial Intelligence Engine (RAG).

UPGRADES vs previous version:
- Removed import-time side effects (os.makedirs, logging.basicConfig no longer
  run on import). This prevents test-suite pollution and multi-process conflicts.
- setup_environment() must be called once explicitly at pipeline startup.
- get_logger() is idempotent: safe to call from any module without duplicating handlers.
- All path construction consolidated — no f-string path building scattered across modules.
"""

import os
import logging
from pathlib import Path

# ── Directory Layout ──────────────────────────────────────────────────────────
# Absolute paths via pathlib so os.chdir() in Colab never breaks resolution.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: str      = str(PROJECT_ROOT / "data" / "raw_pdfs")
ARTIFACTS_DIR: str = str(PROJECT_ROOT / "artifacts")
VECTOR_DB_DIR: str = str(PROJECT_ROOT / "artifacts" / "vector_db")
EVAL_REPORTS_DIR: str = str(PROJECT_ROOT / "artifacts" / "eval_reports")
VISUALS_DIR: str   = str(PROJECT_ROOT / "artifacts" / "visualizations")
LOG_FILE: str      = str(PROJECT_ROOT / "artifacts" / "pipeline_run.log")

_ALL_DIRS: list[str] = [
    ARTIFACTS_DIR,
    VECTOR_DB_DIR,
    EVAL_REPORTS_DIR,
    VISUALS_DIR,
]

# ── RAG Hyperparameters ───────────────────────────────────────────────────────
CHUNK_SIZE: int    = 1200
CHUNK_OVERLAP: int = 250
TOP_K_VECTORS: int = 7

# RRF constant — standard default is 60. Increase to flatten rank differences,
# decrease to make top ranks dominate more aggressively.
RRF_K: int = 60

# Maximum chunks returned per company during balanced retrieval.
# Set to TOP_K_VECTORS // number_of_companies. With 3 companies & TOP_K=7 → 3.
MAX_CHUNKS_PER_COMPANY: int = 3

# Embedding model — BAAI/bge-small-en-v1.5 is a top-ranked open-source model
# on the MTEB leaderboard; cost-free and production-grade.
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

# Generator & Evaluator model names (Groq-hosted)
GENERATOR_MODEL: str  = "llama-3.3-70b-versatile"
EVALUATOR_MODEL: str  = "qwen/qwen3-32b"
                                                 

# API call reliability
MAX_API_RETRIES: int       = 3
API_RETRY_MIN_WAIT: int    = 2   # seconds
API_RETRY_MAX_WAIT: int    = 10  # seconds


# ── Environment Setup ─────────────────────────────────────────────────────────
def setup_environment() -> None:
    """
    Create all required artifact directories.

    Call this ONCE at the start of the pipeline (e.g., Cell 1 of the notebook).
    NOT called at import time — doing so would cause side effects in tests and
    multi-process/multi-worker deployments.
    """
    for directory in _ALL_DIRS:
        os.makedirs(directory, exist_ok=True)


# ── Logger Factory ────────────────────────────────────────────────────────────
def get_logger(name: str = "financial_rag") -> logging.Logger:
    """
    Return a configured logger. Idempotent — safe to call multiple times.

    Adds handlers only once regardless of how many modules call this function,
    preventing duplicated log lines in long-running Colab sessions.

    Args:
        name: Logger name, visible in log output. Use __name__ in each module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Guard: if handlers already attached, return as-is (idempotent).
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always active
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler — written to artifacts dir.
    # We use the absolute LOG_FILE path so it is independent of cwd.
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)  # ensure dir exists for log file
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Non-fatal: if log file cannot be created (permissions, read-only FS),
        # continue with console-only logging.
        logger.warning("Could not create log file at %s. Logging to console only.", LOG_FILE)

    return logger


# ── Module-level logger ───────────────────────────────────────────────────────
# Other modules import this directly:  from src.config import logger
# It is created here so there is one canonical logger instance for the project.
logger: logging.Logger = get_logger("financial_rag")