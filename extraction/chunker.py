"""
Text Chunker
=============
Splits documents into overlapping chunks suitable for LLM extraction,
preserving sentence boundaries and source metadata.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata for traceability."""
    text: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.word_count = len(self.text.split())


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex-based heuristics.
    Handles abbreviations and decimal numbers.
    """
    # Protect common abbreviations
    text = re.sub(r"\b(et al|Fig|fig|Eq|eq|vs|Dr|Mr|Mrs|Prof)\.", r"\1<DOT>", text)
    # Protect decimal numbers
    text = re.sub(r"(\d)\.", r"\1<DOT>", text)

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Restore protected dots
    sentences = [s.replace("<DOT>", ".") for s in sentences]

    # Filter empty sentences
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    max_words: int = 400,
    overlap_words: int = 50,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """
    Split a document into overlapping chunks at sentence boundaries.

    Args:
        text: The full document text
        doc_id: Document identifier for traceability
        max_words: Maximum words per chunk (~512 tokens ≈ 400 words)
        overlap_words: Number of overlapping words between chunks
        metadata: Optional metadata to attach to each chunk

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_word_count = 0
    char_offset = 0

    for sentence in sentences:
        sent_words = len(sentence.split())

        # If adding this sentence exceeds the limit, finalize the current chunk
        if current_word_count + sent_words > max_words and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text_str,
                doc_id=doc_id,
                chunk_index=len(chunks),
                start_char=char_offset,
                end_char=char_offset + len(chunk_text_str),
                metadata=metadata or {},
            ))

            # Calculate overlap: keep last N words worth of sentences
            overlap_sents = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_words = len(s.split())
                if overlap_count + s_words > overlap_words:
                    break
                overlap_sents.insert(0, s)
                overlap_count += s_words

            # Update char offset
            char_offset += len(chunk_text_str) - len(" ".join(overlap_sents))

            current_sentences = overlap_sents
            current_word_count = overlap_count

        current_sentences.append(sentence)
        current_word_count += sent_words

    # Don't forget the last chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(Chunk(
            text=chunk_text_str,
            doc_id=doc_id,
            chunk_index=len(chunks),
            start_char=char_offset,
            end_char=char_offset + len(chunk_text_str),
            metadata=metadata or {},
        ))

    logger.info(
        f"Chunked document '{doc_id}': {len(text)} chars → {len(chunks)} chunks "
        f"(~{max_words} words each, {overlap_words} overlap)"
    )
    return chunks


def chunk_document(doc: dict) -> list[Chunk]:
    """
    Chunk a document dictionary (from document store or PubMed fetcher).
    Uses abstract for PubMed papers, full_text for PDFs.

    Args:
        doc: Document dictionary with 'id', 'abstract'/'full_text', 'title'

    Returns:
        List of Chunk objects
    """
    doc_id = doc.get("id", doc.get("pmid", "unknown"))

    # Determine text to chunk
    text = doc.get("abstract", "") or doc.get("full_text", "")
    if not text:
        logger.warning(f"No text content for document {doc_id}")
        return []

    # For short texts (abstracts), don't chunk — return as single chunk
    word_count = len(text.split())
    if word_count <= 500:
        return [Chunk(
            text=text,
            doc_id=doc_id,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            metadata={
                "title": doc.get("title", ""),
                "doi": doc.get("doi", ""),
                "year": doc.get("year", ""),
            },
        )]

    return chunk_text(
        text=text,
        doc_id=doc_id,
        metadata={
            "title": doc.get("title", ""),
            "doi": doc.get("doi", ""),
            "year": doc.get("year", ""),
        },
    )
