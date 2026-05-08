"""
MicroKG FastAPI Application
============================
REST API for querying the microplastics knowledge graph.
Provides entity search, graph traversal, and RAG-powered QA.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from api.models import (
    EntityResponse, NeighborhoodResponse, PathResponse,
    GraphStatsResponse, QARequest, QAResponse,
    UrlIngestRequest, UrlIngestResponse,
)
from api.url_ingestion import ingest_url_to_graph
from graph.graph_manager import get_graph
from rag.qa_chain import QAChain
from config import API_HOST, API_PORT

# ── App Setup ─────────────────────────────────────────────────

app = FastAPI(
    title="MicroKG API",
    description="Knowledge Graph API for Microplastics Research",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons
_qa_chain = None

def _get_qa():
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = QAChain(get_graph())
    return _qa_chain


# ── Health & Stats ────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "MicroKG API", "version": "1.0.0", "status": "running"}


@app.get("/stats", response_model=GraphStatsResponse)
def graph_stats():
    """Get knowledge graph statistics."""
    return get_graph().get_stats()


# ── Entity Endpoints ─────────────────────────────────────────

@app.get("/entities")
def search_entities(
    q: str = Query(default="", description="Search query"),
    entity_type: str = Query(default="", description="Filter by entity type"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Search entities by name and/or type."""
    results = get_graph().search_entities(query=q, entity_type=entity_type, limit=limit)
    return {"count": len(results), "entities": results}


@app.get("/entities/{entity_id:path}/neighbors")
def get_neighbors(
    entity_id: str,
    hops: int = Query(default=1, ge=1, le=3),
    direction: str = Query(default="both", pattern="^(in|out|both)$"),
):
    """Get N-hop neighborhood of an entity."""
    if not get_graph().get_entity(entity_id):
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")
    return get_graph().get_neighbors(entity_id, hops=hops, direction=direction)


@app.get("/entities/{entity_id:path}")
def get_entity(entity_id: str):
    """Get entity details by ID."""
    entity = get_graph().get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")
    return entity


# ── Graph Traversal ───────────────────────────────────────────

@app.get("/graph/path")
def find_path(
    source: str = Query(..., description="Source entity ID"),
    target: str = Query(..., description="Target entity ID"),
):
    """Find shortest path between two entities."""
    result = get_graph().get_shortest_path(source, target)
    if not result:
        raise HTTPException(status_code=404, detail="No path found between entities")
    return result


@app.get("/graph/subgraph")
def get_subgraph(
    entity_type: str = Query(..., description="Entity type to extract"),
):
    """Get subgraph containing nodes of a specific type."""
    return get_graph().get_subgraph_by_type(entity_type)


# ── QA Endpoint ───────────────────────────────────────────────

@app.post("/qa/ask", response_model=QAResponse)
def ask_question(req: QARequest):
    """Ask a question using RAG over the knowledge graph."""
    result = _get_qa().ask(
        question=req.question,
        top_k=req.top_k,
        hops=req.hops,
    )
    return QAResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result.get("sources", []),
        triples_used=result.get("triples_used", 0),
    )


# ── URL Ingestion ─────────────────────────────────────────────

@app.post("/ingest/url", response_model=UrlIngestResponse)
def ingest_url(req: UrlIngestRequest):
    """Build a small knowledge graph slice from a pasted URL."""
    try:
        return ingest_url_to_graph(
            url=req.url,
            graph=get_graph(),
            max_chunks=req.max_chunks,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
