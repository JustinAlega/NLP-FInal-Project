# MicroKG — Microplastics Domain Knowledge Graph

A domain-specific knowledge graph system that consolidates microplastics research from peer-reviewed publications, enabling structured querying and RAG-powered question answering.

## 🏗️ Architecture

```
PubMed Papers → NLP Extraction (Ollama) → Entity Resolution → Knowledge Graph (NetworkX) → FastAPI + RAG QA
```

**Components:**
- **Ontology**: 8 entity types, 12 relationship types, canonical dictionaries
- **Ingestion**: PubMed E-Utilities API + PDF parser
- **Extraction**: LLM-based NER & relation extraction via Ollama
- **Graph**: NetworkX-based KG with JSON/GraphML export
- **API**: FastAPI REST endpoints for search, traversal, QA
- **RAG**: Hybrid graph + vector retrieval with Ollama-powered QA

## 📋 Prerequisites

1. **Python 3.10+**
2. **Ollama** — Install from [ollama.com](https://ollama.com)
   ```bash
   # Pull a model (llama3 recommended)
   ollama pull llama3
   ```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve

# 3. Run the full pipeline (fetches papers, extracts, builds graph)
python scripts/seed_kg.py --papers 50

# 4. Run demo queries
python scripts/demo_queries.py

# 5. Start the API server
python api/main.py
# → API available at http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stats` | Graph statistics |
| GET | `/entities?q=polyethylene&entity_type=Polymer` | Search entities |
| GET | `/entities/{id}` | Entity details |
| GET | `/entities/{id}/neighbors?hops=2` | N-hop neighborhood |
| GET | `/graph/path?source=...&target=...` | Shortest path |
| GET | `/graph/subgraph?entity_type=Polymer` | Type subgraph |
| POST | `/qa/ask` | RAG-powered QA |

## 🧪 Example QA

```bash
curl -X POST http://localhost:8000/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What polymers are found in drinking water?"}'
```

## 📁 Project Structure

```
NLP Final Project/
├── config.py                  # Central configuration
├── requirements.txt
├── ontology/
│   └── schema.py              # 8 entity types, 12 relation types
├── ingestion/
│   ├── pubmed_fetcher.py      # PubMed API client
│   ├── pdf_parser.py          # PDF text extraction
│   └── document_store.py      # SQLite document store
├── extraction/
│   ├── chunker.py             # Sentence-aware text chunking
│   ├── entity_extractor.py    # Ollama-based NER
│   ├── relation_extractor.py  # Ollama-based relation extraction
│   └── entity_resolver.py     # Deduplication & normalization
├── graph/
│   └── graph_manager.py       # NetworkX KG (search, traversal, export)
├── api/
│   ├── main.py                # FastAPI application
│   └── models.py              # Pydantic schemas
├── rag/
│   ├── embeddings.py          # Sentence-transformer embeddings
│   ├── retriever.py           # Hybrid graph+vector retriever
│   └── qa_chain.py            # Ollama-powered QA
└── scripts/
    ├── seed_kg.py             # End-to-end pipeline
    └── demo_queries.py        # Demo query examples
```

## 🧬 Ontology

**Entity Types**: Polymer, SizeClass, Source, DetectionMethod, EnvironmentalCompartment, HealthEffect, ExposurePathway, Publication

**Relationship Types**: DETECTED_BY, FOUND_IN, ORIGINATES_FROM, HAS_SIZE_CLASS, CAUSES, EXPOSURE_VIA, LEADS_TO, REPORTED_IN, CO_OCCURS_WITH, AFFECTS, REGULATED_BY, MEASURED_IN
