"""
PubMed Fetcher
==============
Fetches microplastics research papers from PubMed using NCBI E-Utilities.
Retrieves title, abstract, authors, DOI, journal, and publication date.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

from Bio import Entrez, Medline
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PUBMED_EMAIL, PUBMED_MAX_RESULTS, PUBMED_QUERIES, RAW_DATA_DIR
)

logger = logging.getLogger(__name__)

# Required by NCBI
Entrez.email = PUBMED_EMAIL


def search_pubmed(query: str, max_results: int = 50) -> list[str]:
    """
    Search PubMed and return a list of PMIDs.

    Args:
        query: PubMed search query string
        max_results: Maximum number of results to return

    Returns:
        List of PubMed IDs (PMIDs)
    """
    logger.info(f"Searching PubMed: '{query}' (max {max_results})")
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
        usehistory="y",
    )
    results = Entrez.read(handle)
    handle.close()

    pmids = results.get("IdList", [])
    total = results.get("Count", "0")
    logger.info(f"Found {total} total results, retrieved {len(pmids)} PMIDs")

    return pmids


def fetch_paper_details(pmids: list[str]) -> list[dict]:
    """
    Fetch detailed metadata for a list of PMIDs.

    Args:
        pmids: List of PubMed IDs

    Returns:
        List of paper metadata dictionaries
    """
    if not pmids:
        return []

    papers = []

    # Fetch in batches of 20 to respect rate limits
    batch_size = 20
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        logger.info(f"Fetching details for PMIDs {i+1}-{i+len(batch)}")

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="medline",
                retmode="text",
            )
            records = list(Medline.parse(handle))
            handle.close()

            for record in records:
                paper = _parse_medline_record(record)
                if paper and paper.get("abstract"):  # Only keep papers with abstracts
                    papers.append(paper)

        except Exception as e:
            logger.error(f"Error fetching batch starting at {i}: {e}")

        # NCBI rate limit: max 3 requests per second
        time.sleep(0.4)

    return papers


def _parse_medline_record(record: dict) -> Optional[dict]:
    """Parse a MEDLINE record into a structured dictionary."""
    try:
        # Extract DOI from article identifiers
        doi = ""
        aid_list = record.get("AID", [])
        for aid in aid_list:
            if "[doi]" in aid:
                doi = aid.replace(" [doi]", "")
                break

        paper = {
            "pmid": record.get("PMID", ""),
            "title": record.get("TI", ""),
            "abstract": record.get("AB", ""),
            "authors": record.get("AU", []),
            "journal": record.get("JT", record.get("TA", "")),
            "year": _extract_year(record.get("DP", "")),
            "doi": doi,
            "mesh_terms": record.get("MH", []),
            "keywords": record.get("OT", []),
            "publication_type": record.get("PT", []),
        }
        return paper

    except Exception as e:
        logger.warning(f"Failed to parse record: {e}")
        return None


def _extract_year(date_str: str) -> str:
    """Extract year from PubMed date string (e.g., '2024 Jan 15')."""
    if date_str:
        parts = date_str.split()
        if parts and parts[0].isdigit():
            return parts[0]
    return ""


def fetch_microplastics_papers(
    queries: Optional[list[str]] = None,
    max_per_query: int = 15,
    total_target: int = 50,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Fetch microplastics papers from PubMed across multiple queries.

    Args:
        queries: List of search queries (defaults to config queries)
        max_per_query: Max results per individual query
        total_target: Target total number of unique papers
        output_dir: Directory to save raw results

    Returns:
        List of unique paper metadata dictionaries
    """
    if queries is None:
        queries = PUBMED_QUERIES
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    all_papers = {}  # pmid -> paper (for deduplication)

    print(f"\n🔬 Fetching microplastics papers from PubMed...")
    print(f"   Target: ~{total_target} papers across {len(queries)} queries\n")

    for query in tqdm(queries, desc="Search queries"):
        # Adjust per-query limit based on remaining papers needed
        remaining = total_target - len(all_papers)
        if remaining <= 0:
            break

        fetch_count = min(max_per_query, remaining + 5)  # Fetch extra to account for dupes

        pmids = search_pubmed(query, max_results=fetch_count)
        papers = fetch_paper_details(pmids)

        new_count = 0
        for paper in papers:
            pmid = paper["pmid"]
            if pmid not in all_papers:
                all_papers[pmid] = paper
                new_count += 1

        logger.info(f"Query '{query}': {len(papers)} fetched, {new_count} new")
        time.sleep(0.5)  # Rate limiting between queries

    # Convert to list and trim to target
    paper_list = list(all_papers.values())[:total_target]

    print(f"\n✅ Fetched {len(paper_list)} unique papers with abstracts")

    # Save raw data
    output_path = output_dir / "pubmed_papers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(paper_list, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to {output_path}")

    return paper_list


def load_cached_papers(path: Optional[Path] = None) -> list[dict]:
    """Load previously fetched papers from JSON cache."""
    if path is None:
        path = RAW_DATA_DIR / "pubmed_papers.json"

    if not path.exists():
        logger.warning(f"No cached papers found at {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} papers from cache")
    return papers


# ── CLI Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    papers = fetch_microplastics_papers(total_target=50)
    print(f"\nSample paper titles:")
    for p in papers[:5]:
        print(f"  • {p['title'][:80]}...")
