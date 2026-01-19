"""
Configuration file for Help Center QA Search system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_secret(key: str, default: str = None) -> str:
    """Get secret from environment variable or Streamlit secrets."""
    # 1. Try environment variable first (local .env)
    value = os.getenv(key)
    if value:
        return value

    # 2. Try Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default


# Project root directory
PROJECT_ROOT = Path(__file__).parent

# API Keys
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # For Phase 2
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # For Phase 2 reranking

# Zendesk API credentials (optional - for Help Center data sync)
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN", "socar-docs")
ZENDESK_LOCALE = os.getenv("ZENDESK_LOCALE", "ko")

# Validate required keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "help_center_carsharing_only_20260109_135511.csv"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "help_center_chroma"
VEHICLE_DATA_PATH = PROJECT_ROOT / "data" / "vehicle_manual_data.csv"
VEHICLE_VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "vehicle_manuals_chroma"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
COLLECTION_NAME = "help_center"
VEHICLE_COLLECTION_NAME = "vehicle_manuals"

# Data source mode: "csv" or "api"
DATA_SOURCE_MODE = os.getenv("DATA_SOURCE_MODE", "api")

# Model settings
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536  # Options: 1536 (cost-effective) or 3072 (max quality)

# Retrieval settings
BM25_TOP_K = 20
VECTOR_TOP_K = 20
FINAL_TOP_K = 10
RRF_K = 60  # Reciprocal Rank Fusion constant
ALPHA = 0.5  # BM25 weight (0: Vector only, 1: BM25 only, 0.5: balanced)

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Help Center Search API"
API_VERSION = "0.1.0"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Authentication settings
AUTH_ENABLED = get_secret("AUTH_ENABLED", "true").lower() == "true"
ALLOWED_EMAIL_DOMAINS = get_secret("ALLOWED_EMAIL_DOMAINS", "socar.kr").split(",")  # comma-separated

# Feature flags
# ENABLE_RERANKING: Auto-enabled if COHERE_API_KEY is set, or explicitly via env var
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true" if COHERE_API_KEY else "false").lower() == "true"
ENABLE_CACHING = False  # Phase 2
ENABLE_QUERY_EXPANSION = False  # Phase 2

print(f"✅ Config loaded successfully")
print(f"   - Data path: {DATA_PATH}")
print(f"   - Vector store: {VECTORSTORE_DIR}")
print(f"   - Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS}D)")
