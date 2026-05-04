"""
PDF Parser
===========
Extracts text from uploaded PDF documents (policy papers, reports)
using PyMuPDF for section-aware parsing.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str | Path) -> dict:
    """
    Extract text and metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with title, text, pages, and metadata
    """
    if fitz is None:
        raise ImportError("PyMuPDF is required: pip install PyMuPDF")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))

    # Extract metadata
    metadata = doc.metadata or {}

    # Extract text page by page
    pages = []
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append({
            "page_number": page_num + 1,
            "text": text.strip(),
        })
        full_text.append(text)

    doc.close()

    result = {
        "source_file": pdf_path.name,
        "title": metadata.get("title", pdf_path.stem),
        "author": metadata.get("author", ""),
        "text": "\n\n".join(full_text).strip(),
        "pages": pages,
        "page_count": len(pages),
        "metadata": metadata,
    }

    logger.info(
        f"Extracted {len(result['text'])} chars from {pdf_path.name} "
        f"({len(pages)} pages)"
    )
    return result


def extract_sections(text: str) -> dict[str, str]:
    """
    Attempt to split a paper's text into common sections.

    Args:
        text: Full paper text

    Returns:
        Dictionary mapping section names to their text content
    """
    import re

    # Common section headers in scientific papers
    section_patterns = [
        r"(?i)\b(abstract)\b",
        r"(?i)\b(introduction)\b",
        r"(?i)\b(background)\b",
        r"(?i)\b(methods?|materials?\s+and\s+methods?|experimental)\b",
        r"(?i)\b(results?)\b",
        r"(?i)\b(discussion)\b",
        r"(?i)\b(results?\s+and\s+discussion)\b",
        r"(?i)\b(conclusion|conclusions|summary)\b",
        r"(?i)\b(references|bibliography)\b",
    ]

    sections = {"full_text": text}
    lines = text.split("\n")

    current_section = "preamble"
    section_texts: dict[str, list[str]] = {current_section: []}

    for line in lines:
        stripped = line.strip()
        matched = False

        for pattern in section_patterns:
            # Check if line is primarily a section header (short, matches pattern)
            if len(stripped) < 80 and re.search(pattern, stripped):
                current_section = re.sub(r"[^a-z\s]", "", stripped.lower()).strip()
                if current_section not in section_texts:
                    section_texts[current_section] = []
                matched = True
                break

        if not matched:
            section_texts.setdefault(current_section, []).append(line)

    for section_name, section_lines in section_texts.items():
        content = "\n".join(section_lines).strip()
        if content:
            sections[section_name] = content

    return sections


def process_pdf_directory(
    pdf_dir: str | Path,
    output_dir: Optional[Path] = None
) -> list[dict]:
    """
    Process all PDFs in a directory.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Optional output directory for extracted text

    Returns:
        List of extracted document dictionaries
    """
    import json

    pdf_dir = Path(pdf_dir)
    documents = []

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []

    logger.info(f"Processing {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        try:
            doc = extract_text_from_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "pdf_documents.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} documents to {output_path}")

    return documents
