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
from urllib.parse import unquote, urlparse

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
        self._meta: dict[str, str] = {}
        self._json_ld: list[str] = []
        self._in_json_ld = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "script" and attrs_dict.get("type") == "application/ld+json":
            self._in_json_ld = True
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = (
                attrs_dict.get("name")
                or attrs_dict.get("property")
                or attrs_dict.get("itemprop")
                or ""
            ).lower()
            value = attrs_dict.get("content", "").strip()
            if key and value:
                self._meta[key] = value
        if tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "script":
            self._in_json_ld = False
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        text = data.strip()
        if self._in_json_ld and text:
            self._json_ld.append(text)
            return
        if not text or self._skip_depth:
            return
        if self._in_title:
            self._title = f"{self._title} {text}".strip()
        self._parts.append(text)

    @property
    def title(self) -> str:
        return (
            self._meta.get("citation_title")
            or self._meta.get("dc.title")
            or self._meta.get("og:title")
            or self._title
        )

    @property
    def abstract(self) -> str:
        for key in (
            "citation_abstract",
            "dc.description",
            "description",
            "og:description",
            "twitter:description",
        ):
            if self._meta.get(key):
                return self._meta[key]
        return ""

    @property
    def doi(self) -> str:
        doi = self._meta.get("citation_doi") or self._meta.get("dc.identifier")
        if doi:
            return doi.replace("doi:", "").strip()
        return ""

    @property
    def source(self) -> str:
        return self._meta.get("citation_journal_title") or self._meta.get("dc.source") or ""

    @property
    def text(self) -> str:
        text = " ".join(self._parts)
        text = re.sub(r"\s+", " ", text)
        return _clean_article_text(text)


def ingest_url_to_graph(url: str, graph, max_chunks: int = 3) -> dict:
    """Fetch a URL, extract entities/relations, add them to graph, and return a graph slice."""
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")

    doc = _document_from_pubmed_url(url)
    if doc is None:
        try:
            response = requests.get(url, timeout=20, headers=_request_headers())
            response.raise_for_status()
            doc = _document_from_response(url, response)
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code not in {401, 403, 429}:
                raise
            doc = _document_from_reader_url(url)
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
    stable_id = _doi_from_url(url) or url
    doc_id = f"url:{hashlib.sha1(stable_id.encode('utf-8')).hexdigest()[:12]}"

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
    readable_text = _best_readable_text(parser)
    doi = parser.doi or _doi_from_url(url) or url
    metadata = _crossref_metadata(doi) if doi and not doi.startswith(("http", "pii:")) else {}
    if _is_bad_title(title) and metadata.get("title"):
        title = metadata["title"]
    if len(readable_text.split()) < 40 and metadata.get("abstract"):
        readable_text = metadata["abstract"]
    source = _source_from_url(url, parser.source)

    return {
        "id": doc_id,
        "title": title,
        "abstract": readable_text,
        "doi": doi,
        "source": source,
    }


def _document_from_reader_url(url: str) -> dict:
    """Fetch pages blocked to direct requests through a public reader endpoint."""
    reader_url = f"https://r.jina.ai/{url}"
    response = requests.get(reader_url, timeout=30, headers=_request_headers())
    response.raise_for_status()

    text = _clean_article_text(response.text)
    title = _title_from_reader_text(text) or urlparse(url).netloc or url
    stable_id = _doi_from_url(url) or url
    metadata = _crossref_metadata(stable_id) if stable_id and not stable_id.startswith(("http", "pii:")) else {}
    if _is_bad_title(title) and metadata.get("title"):
        title = metadata["title"]
    if len(text.split()) < 40 and metadata.get("abstract"):
        text = metadata["abstract"]

    return {
        "id": f"url:{hashlib.sha1(stable_id.encode('utf-8')).hexdigest()[:12]}",
        "title": title,
        "abstract": text,
        "doi": _doi_from_url(url) or url,
        "source": _source_from_url(url),
    }


def _request_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 MicroKG/1.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _best_readable_text(parser: ReadableHtmlParser) -> str:
    if len(parser.abstract.split()) >= 25:
        return parser.abstract
    return parser.text


def _doi_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path)

    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", path)
    if doi_match:
        return doi_match.group(0).rstrip("/")

    pii_match = re.search(r"/pii/([A-Za-z0-9]+)", path)
    if pii_match:
        return f"pii:{pii_match.group(1)}"

    return ""


def _source_from_url(url: str, fallback: str = "") -> str:
    host = urlparse(url).netloc.lower()
    if "pubs.acs.org" in host:
        return "ACS Publications"
    if "sciencedirect.com" in host:
        return "ScienceDirect"
    if "elsevier" in host:
        return "Elsevier"
    if "pubmed.ncbi.nlm.nih.gov" in host:
        return "PubMed"
    return fallback or host


def _clean_article_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    repeated_noise = [
        "Skip to main content",
        "Access through your institution",
        "Sign in",
        "Register",
        "Purchase PDF",
        "View PDF",
        "Download PDF",
        "Cookie",
        "Copyright",
    ]
    for phrase in repeated_noise:
        text = text.replace(phrase, " ")

    abstract_match = re.search(
        r"(Abstract|Summary)\s+(.*?)(Introduction|Graphical abstract|Keywords|Section snippets|References)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if abstract_match and len(abstract_match.group(2).split()) >= 40:
        return _clean_article_text(abstract_match.group(2))

    return re.sub(r"\s+", " ", text).strip()


def _title_from_reader_text(text: str) -> str:
    match = re.search(r"(?:^|\s)Title:\s*(.*?)(?:\s+URL Source:|\s+Markdown Content:|$)", text)
    if match:
        return match.group(1).strip()
    heading = re.search(r"#\s+(.+)", text)
    return heading.group(1).strip() if heading else ""


def _is_bad_title(title: str) -> bool:
    lowered = title.lower().strip()
    return not lowered or lowered in {"just a moment...", "access denied", "attention required"}


def _crossref_metadata(doi: str) -> dict:
    try:
        response = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=15,
            headers=_request_headers(),
        )
        response.raise_for_status()
        message = response.json().get("message", {})
        titles = message.get("title", [])
        abstracts = message.get("abstract", "")
        return {
            "title": titles[0] if titles else "",
            "abstract": _clean_article_text(re.sub(r"<[^>]+>", " ", abstracts)) if abstracts else "",
        }
    except Exception:
        return {}


def _publication_entity(doc: dict) -> Entity:
    title = doc.get("title") or doc["id"]
    normalized = re.sub(r"[^a-z0-9\s-]", "", title.lower())
    normalized = re.sub(r"\s+", "_", normalized).strip("_")[:80] or doc["id"]

    return Entity(
        id=f"{EntityType.PUBLICATION.value}:{normalized}",
        entity_type=EntityType.PUBLICATION.value,
        name=title,
        attributes={
            "url": doc.get("doi", ""),
            "source": doc.get("source", ""),
        },
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
