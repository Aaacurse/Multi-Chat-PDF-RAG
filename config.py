"""Configuration and constants for the PDF RAG Chat application"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════════════════════════════

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found! Please add it to your .env file:\n"
        "GOOGLE_API_KEY=your_api_key_here"
    )

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Embedding Model
EMBEDDING_MODEL = "gemini-embedding-001"  # Gemini embedding model

# LLM Model
LLM_MODEL = "gemini-2.5-flash"  # Fast and cost-effective
LLM_TEMP = 0.5  # Lower = more factual, Higher = more creative

# ══════════════════════════════════════════════════════════════════════════════
# TEXT SPLITTING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 1000  # Characters per chunk (~250 words)
CHUNK_OVERLAP = 200  # Overlap between chunks to preserve context

# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TOP_K_RESULTS = 4  # Number of chunks to retrieve per query

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

# Base directory (project root)
BASE_DIR = Path(__file__).parent

# Data directory (for persistent storage)
DATA_DIR = BASE_DIR / "data"

# Vector stores directory (FAISS indices)
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"

# App state file (chat histories)
STATE_FILE = DATA_DIR / "app_state.json"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PAGE_TITLE = "PDF RAG Chat"
PAGE_ICON = ""
LAYOUT = "wide"  # "centered" or "wide"

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED SETTINGS (Optional)
# ══════════════════════════════════════════════════════════════════════════════

# Maximum file size for PDF upload (in MB)
MAX_FILE_SIZE_MB = 200

# Show debug info in sidebar
DEBUG_MODE = False

# Enable streaming responses (requires async implementation)
ENABLE_STREAMING = False

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_config():
    """Validate configuration on import"""
    
    # Check if API key is set
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
        raise ValueError(
            "⚠️ GOOGLE_API_KEY is not properly configured!\n\n"
            "Please create a .env file in the project root with:\n"
            "GOOGLE_API_KEY=your_actual_api_key\n\n"
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )
    
    # Check if directories are writable
    try:
        test_file = DATA_DIR / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        raise PermissionError(
            f"Cannot write to data directory: {DATA_DIR}\n"
            f"Error: {str(e)}"
        )

# Run validation on import
validate_config()

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT ALL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # API
    "GOOGLE_API_KEY",
    
    # Models
    "EMBEDDING_MODEL",
    "LLM_MODEL",
    "LLM_TEMP",
    
    # Text Processing
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    
    # Retrieval
    "TOP_K_RESULTS",
    
    # Paths
    "BASE_DIR",
    "DATA_DIR",
    "VECTOR_STORE_DIR",
    "STATE_FILE",
    
    # Streamlit
    "PAGE_TITLE",
    "PAGE_ICON",
    "LAYOUT",
    
    # Advanced
    "MAX_FILE_SIZE_MB",
    "DEBUG_MODE",
    "ENABLE_STREAMING",
]