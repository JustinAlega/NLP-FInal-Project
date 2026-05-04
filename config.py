"""
MicroKG Configuration
=====================
Central configuration for the Microplastics Knowledge Graph system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAPH_DATA_DIR = DATA_DIR / "graph"
DB_PATH = DATA_DIR / "documents.db"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, GRAPH_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama Configuration ──────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── PubMed Configuration ─────────────────────────────────────
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "microkg@example.com")
PUBMED_MAX_RESULTS = int(os.getenv("PUBMED_MAX_RESULTS", "50"))
PUBMED_QUERIES = [
    '"microplastic" AND "health"',
    '"microplastic" AND "detection"',
    '"microplastic" AND "drinking water"',
    '"nanoplastic" AND "human"',
    '"microplastic" AND "polymer" AND "environment"',
]

# ── Extraction Configuration ─────────────────────────────────
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 64        # token overlap between chunks
EXTRACTION_TEMPERATURE = 0.1
EXTRACTION_MAX_RETRIES = 3

# ── Graph Configuration ──────────────────────────────────────
GRAPH_EXPORT_PATH = GRAPH_DATA_DIR / "microkg_graph.json"
GRAPH_GRAPHML_PATH = GRAPH_DATA_DIR / "microkg_graph.graphml"

# ── API Configuration ────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Embedding Configuration ──────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model
EMBEDDING_DIM = 384
SIMILARITY_THRESHOLD = 0.85             # for entity resolution
