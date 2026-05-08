"""
Pydantic Models
================
Request/response schemas for the MicroKG API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class EntityResponse(BaseModel):
    id: str
    entity_type: str
    name: str
    aliases: list[str] = []
    attributes: dict = {}
    confidence: float = 1.0
    source_doc: str = ""


class RelationResponse(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: float = 1.0
    evidence: str = ""


class NeighborhoodResponse(BaseModel):
    center: str
    nodes: list[dict] = []
    edges: list[dict] = []


class PathResponse(BaseModel):
    path: list[str] = []
    nodes: list[dict] = []
    edges: list[dict] = []
    length: int = 0


class GraphStatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    nodes_by_type: dict = {}
    edges_by_type: dict = {}
    density: float = 0
    connected_components: int = 0


class QARequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question")
    top_k: int = Field(default=10, ge=1, le=50)
    hops: int = Field(default=1, ge=1, le=3)


class QAResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict] = []
    triples_used: int = 0


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=8, description="Public URL to extract into the graph")
    max_chunks: int = Field(default=3, ge=1, le=8)


class UrlIngestResponse(BaseModel):
    url: str
    title: str = ""
    doc_id: str
    chunks_processed: int
    entities_added: int
    relations_added: int
    nodes: list[dict] = []
    edges: list[dict] = []
