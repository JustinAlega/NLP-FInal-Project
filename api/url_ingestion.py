"""
URL ingestion helpers for the API.
==================================
Fetches a public URL, extracts readable text, and builds a small graph slice.
"""

import hashlib
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests

from extraction.chunker import chunk_document
from extraction.entity_extractor import extract_entities_from_text
from extraction.entity_resolver import resolve_entities
from extraction.relation_extractor import extract_relations_from_text
from ingestion.pdf_parser import extract_text_from_pdf
from ingestion.pubmed_fetcher import fetch_paper_details
from ontology.schema import (
    CANONICAL_COMPARTMENTS,
    CANONICAL_EXPOSURE_PATHWAYS,
    CANONICAL_POLYMERS,
    Entity,
    EntityType,
    Relation,
    RelationType,
)


class ReadableHtmlParser(HTMLParser):
    """Small stdlib HTML-to-text parser for article-like pages."""

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._title = ""
        self._in_title = False
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        text = data.strip()
        if not text or self._skip_depth:
            return
        if self._in_title:
            self._title = f"{self._title} {text}".strip()
        self._parts.append(text)

    @property
    def title(self) -> str:
        return self._title

    @property
    def text(self) -> str:
        text = " ".join(self._parts)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def ingest_url_to_graph(url: str, graph, max_chunks: int = 3) -> dict:
    """Fetch a URL, extract entities/relations, add them to graph, and return a graph slice."""
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")

    doc = _document_from_pubmed_url(url)
    if doc is None:
        response = requests.get(url, timeout=20, headers={"User-Agent": "MicroKG/1.0"})
        response.raise_for_status()
        doc = _document_from_response(url, response)
    chunks = chunk_document(doc)[:max_chunks]
    if not chunks:
        raise ValueError("No readable text was found at that URL")

    chunk_entities = []
    all_entities = []
    for chunk in chunks:
        entities = extract_entities_from_text(chunk.text, doc_id=doc["id"])
        resolved = resolve_entities(entities, use_embeddings=False)
        chunk_entities.append((chunk, resolved))
        all_entities.extend(resolved)

    entities = resolve_entities(all_entities, use_embeddings=False)
    entity_by_id = {entity.id: entity for entity in entities}

    relations = []
    for chunk, entities_for_chunk in chunk_entities:
        relations.extend(
            extract_relations_from_text(chunk.text, entities_for_chunk, doc_id=chunk.doc_id)
        )

    if len(entity_by_id) < 3:
        fallback_entities = resolve_entities(
            _extract_entities_rule_based(" ".join(chunk.text for chunk in chunks), doc["id"]),
            use_embeddings=False,
        )
        entity_by_id = {entity.id: entity for entity in fallback_entities}
        relations = _generate_relations_rule_based(list(entity_by_id.values()), doc["id"])

    publication = _publication_entity(doc)
    graph.add_entity(publication)
    graph.add_entities(list(entity_by_id.values()))

    source_relations = [
        Relation(
            source_id=entity.id,
            relation_type=RelationType.REPORTED_IN.value,
            target_id=publication.id,
            confidence=1.0,
            evidence=f"Extracted from {url}",
            source_doc=doc["id"],
        )
        for entity in entity_by_id.values()
        if entity.entity_type != EntityType.PUBLICATION.value
    ]

    graph.add_relations(relations + source_relations)
    graph.export_json()

    node_ids = set(entity_by_id) | {publication.id}
    nodes = [graph.get_entity(node_id) for node_id in node_ids if graph.get_entity(node_id)]
    edges = [
        edge
        for edge in relations + source_relations
        if edge.source_id in node_ids and edge.target_id in node_ids
    ]

    return {
        "url": url,
        "title": doc.get("title", ""),
        "doc_id": doc["id"],
        "chunks_processed": len(chunks),
        "entities_added": len(entity_by_id) + 1,
        "relations_added": len(edges),
        "nodes": nodes,
        "edges": [edge.to_dict() for edge in edges],
    }


def _document_from_pubmed_url(url: str) -> dict | None:
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if not match:
        return None

    papers = fetch_paper_details([match.group(1)])
    if not papers:
        return None

    paper = papers[0]
    return {
        "id": f"pubmed:{paper.get('pmid', match.group(1))}",
        "title": paper.get("title", url),
        "abstract": paper.get("abstract", ""),
        "doi": paper.get("doi", url) or url,
        "year": paper.get("year", ""),
    }


def _document_from_response(url: str, response: requests.Response) -> dict:
    content_type = response.headers.get("content-type", "").lower()
    doc_id = f"url:{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}"

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_path = Path(temp_file.name)
        try:
            parsed = extract_text_from_pdf(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        return {
            "id": doc_id,
            "title": parsed.get("title", url),
            "full_text": parsed.get("text", ""),
            "doi": url,
        }

    parser = ReadableHtmlParser()
    parser.feed(response.text)
    title = parser.title or urlparse(url).netloc or url

    return {
        "id": doc_id,
        "title": title,
        "abstract": parser.text,
        "doi": url,
    }


def _publication_entity(doc: dict) -> Entity:
    title = doc.get("title") or doc["id"]
    normalized = re.sub(r"[^a-z0-9\s-]", "", title.lower())
    normalized = re.sub(r"\s+", "_", normalized).strip("_")[:80] or doc["id"]

    return Entity(
        id=f"{EntityType.PUBLICATION.value}:{normalized}",
        entity_type=EntityType.PUBLICATION.value,
        name=title,
        attributes={"url": doc.get("doi", "")},
        confidence=1.0,
        source_doc=doc["id"],
    )


def _extract_entities_rule_based(text: str, doc_id: str) -> list[Entity]:
    text_lower = text.lower()
    entities = []

    for canonical, aliases in CANONICAL_POLYMERS.items():
        if _contains_any(text_lower, [canonical, *aliases]):
            entities.append(_entity(EntityType.POLYMER.value, canonical, doc_id, 0.75))

    for canonical, aliases in CANONICAL_COMPARTMENTS.items():
        if _contains_any(text_lower, [canonical, *aliases]):
            entities.append(_entity(EntityType.ENVIRONMENTAL_COMPARTMENT.value, canonical, doc_id, 0.72))

    for canonical, aliases in CANONICAL_EXPOSURE_PATHWAYS.items():
        if _contains_any(text_lower, [canonical, *aliases]):
            entities.append(_entity(EntityType.EXPOSURE_PATHWAY.value, canonical, doc_id, 0.72))

    health_terms = {
        "Oxidative stress": ["oxidative stress"],
        "Inflammation": ["inflammation", "inflammatory"],
        "Cytotoxicity": ["cytotoxicity", "cytotoxic"],
        "Reproductive toxicity": ["fertility", "reproductive", "pregnancy"],
        "Respiratory effects": ["respiratory", "lung", "inhalation"],
        "Digestive effects": ["digestive", "intestinal", "gut"],
    }
    for canonical, aliases in health_terms.items():
        if _contains_any(text_lower, aliases):
            entities.append(_entity(EntityType.HEALTH_EFFECT.value, canonical, doc_id, 0.72))

    return entities


def _generate_relations_rule_based(entities: list[Entity], doc_id: str) -> list[Relation]:
    polymers = [entity for entity in entities if entity.entity_type == EntityType.POLYMER.value]
    compartments = [
        entity for entity in entities if entity.entity_type == EntityType.ENVIRONMENTAL_COMPARTMENT.value
    ]
    health_effects = [entity for entity in entities if entity.entity_type == EntityType.HEALTH_EFFECT.value]
    pathways = [entity for entity in entities if entity.entity_type == EntityType.EXPOSURE_PATHWAY.value]
    relations = []

    for polymer in polymers:
        for compartment in compartments[:4]:
            relations.append(
                Relation(polymer.id, RelationType.FOUND_IN.value, compartment.id, 0.7, source_doc=doc_id)
            )
        for effect in health_effects[:4]:
            relations.append(
                Relation(polymer.id, RelationType.CAUSES.value, effect.id, 0.65, source_doc=doc_id)
            )

    for effect in health_effects:
        for pathway in pathways:
            relations.append(
                Relation(effect.id, RelationType.EXPOSURE_VIA.value, pathway.id, 0.65, source_doc=doc_id)
            )

    return relations


def _entity(entity_type: str, name: str, doc_id: str, confidence: float) -> Entity:
    normalized = re.sub(r"[^a-z0-9\s-]", "", name.lower())
    normalized = re.sub(r"\s+", "_", normalized).strip("_")
    return Entity(
        id=f"{entity_type}:{normalized}",
        entity_type=entity_type,
        name=name,
        confidence=confidence,
        source_doc=doc_id,
    )


def _contains_any(text_lower: str, terms: list[str]) -> bool:
    return any(term.lower() in text_lower for term in terms)
