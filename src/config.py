"""
Configuration settings for the Financial Intelligence Engine (RAG).
"""
import os
import logging
from pathlib import Path

# UPGRADE: Directory Setup using absolute paths via pathlib to prevent os.chdir breaks
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "data" / "raw_pdfs")
ARTIFACTS_DIR = str(PROJECT_ROOT / "artifacts")
VECTOR_DB_DIR = f"{ARTIFACTS_DIR}/vector_db"
EVAL_REPORTS_DIR = f"{ARTIFACTS_DIR}/eval_reports"

# Auto-create directories
sub_dirs = [ARTIFACTS_DIR, VECTOR_DB_DIR, EVAL_REPORTS_DIR]
for d in sub_dirs:
    os.makedirs(d, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(f"{ARTIFACTS_DIR}/ingestion_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# RAG Hyperparameters
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
TOP_K_VECTORS = 7