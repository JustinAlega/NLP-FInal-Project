"""
Document Store
==============
SQLite-backed storage for ingested documents, tracking processing status
and deduplication.
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

logger = logging.getLogger(__name__)


class DocumentStore:
    """SQLite-backed document metadata and text store."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,           -- 'pubmed' or 'pdf'
                    pmid TEXT,
                    doi TEXT,
                    title TEXT,
                    abstract TEXT,
                    full_text TEXT,
                    authors TEXT,                    -- JSON list
                    journal TEXT,
                    year TEXT,
                    keywords TEXT,                   -- JSON list
                    mesh_terms TEXT,                 -- JSON list
                    status TEXT DEFAULT 'ingested',  -- ingested/extracted/resolved
                    ingested_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    entities TEXT,      -- JSON list of extracted entities
                    relations TEXT,     -- JSON list of extracted relations
                    chunk_index INTEGER,
                    extracted_at TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_pmid ON documents(pmid)
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def add_pubmed_paper(self, paper: dict) -> bool:
        """
        Add a PubMed paper to the store. Returns True if newly inserted.

        Args:
            paper: Paper metadata dict from pubmed_fetcher

        Returns:
            True if the paper was newly inserted, False if duplicate
        """
        doc_id = f"pubmed:{paper['pmid']}"
        now = datetime.utcnow().isoformat()

        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO documents
                       (id, source, pmid, doi, title, abstract, authors,
                        journal, year, keywords, mesh_terms, status,
                        ingested_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        doc_id,
                        "pubmed",
                        paper.get("pmid", ""),
                        paper.get("doi", ""),
                        paper.get("title", ""),
                        paper.get("abstract", ""),
                        json.dumps(paper.get("authors", [])),
                        paper.get("journal", ""),
                        paper.get("year", ""),
                        json.dumps(paper.get("keywords", [])),
                        json.dumps(paper.get("mesh_terms", [])),
                        "ingested",
                        now,
                        now,
                    ),
                )
                return conn.total_changes > 0

        except sqlite3.Error as e:
            logger.error(f"Failed to insert paper {doc_id}: {e}")
            return False

    def add_pdf_document(self, doc: dict) -> bool:
        """Add a parsed PDF document to the store."""
        doc_id = f"pdf:{doc.get('source_file', 'unknown')}"
        now = datetime.utcnow().isoformat()

        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO documents
                       (id, source, title, full_text, authors, status,
                        ingested_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        doc_id,
                        "pdf",
                        doc.get("title", ""),
                        doc.get("text", ""),
                        doc.get("author", ""),
                        "ingested",
                        now,
                        now,
                    ),
                )
                return conn.total_changes > 0

        except sqlite3.Error as e:
            logger.error(f"Failed to insert PDF document {doc_id}: {e}")
            return False

    def bulk_add_papers(self, papers: list[dict]) -> int:
        """Add multiple PubMed papers. Returns count of newly inserted."""
        inserted = sum(1 for p in papers if self.add_pubmed_paper(p))
        logger.info(f"Inserted {inserted}/{len(papers)} papers")
        return inserted

    def get_documents(
        self,
        status: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve documents with optional filtering."""
        query = "SELECT * FROM documents WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY ingested_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get a single document by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()

        return dict(row) if row else None

    def update_status(self, doc_id: str, status: str):
        """Update the processing status of a document."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, doc_id),
            )

    def save_extraction_results(
        self,
        doc_id: str,
        entities: list[dict],
        relations: list[dict],
        chunk_index: int = 0,
    ):
        """Save extraction results for a document chunk."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO extraction_results
                   (doc_id, entities, relations, chunk_index, extracted_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    json.dumps(entities),
                    json.dumps(relations),
                    chunk_index,
                    now,
                ),
            )

    def get_extraction_results(self, doc_id: str) -> list[dict]:
        """Get all extraction results for a document."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM extraction_results WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            r["entities"] = json.loads(r["entities"])
            r["relations"] = json.loads(r["relations"])
            results.append(r)

        return results

    def get_stats(self) -> dict:
        """Get document store statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            by_status = conn.execute(
                "SELECT status, COUNT(*) as count FROM documents GROUP BY status"
            ).fetchall()
            by_source = conn.execute(
                "SELECT source, COUNT(*) as count FROM documents GROUP BY source"
            ).fetchall()
            extractions = conn.execute(
                "SELECT COUNT(*) FROM extraction_results"
            ).fetchone()[0]

        return {
            "total_documents": total,
            "by_status": {row["status"]: row["count"] for row in by_status},
            "by_source": {row["source"]: row["count"] for row in by_source},
            "total_extractions": extractions,
        }

    def clear(self):
        """Clear all data (use with caution)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM extraction_results")
            conn.execute("DELETE FROM documents")
        logger.info("Document store cleared")
