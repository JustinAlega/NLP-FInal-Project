"""
MicroKG Seed Pipeline
======================
End-to-end script that:
1. Fetches papers from PubMed
2. Extracts entities and relations using Ollama
3. Resolves entities
4. Builds the knowledge graph
5. Exports the graph

Usage:
    python scripts/seed_kg.py
    python scripts/seed_kg.py --papers 20 --skip-fetch  (use cached papers)
"""

import argparse, json, logging, sys, time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, GRAPH_EXPORT_PATH
from ontology.schema import Entity, Relation, EntityType, export_ontology_json
from ingestion.pubmed_fetcher import fetch_microplastics_papers, load_cached_papers
from ingestion.document_store import DocumentStore
from extraction.chunker import chunk_document
from extraction.entity_extractor import extract_entities_from_text
from extraction.relation_extractor import extract_relations_from_text
from extraction.entity_resolver import resolve_entities
from graph.graph_manager import KnowledgeGraph
from tqdm import tqdm


def run_pipeline(
    num_papers: int = 50,
    skip_fetch: bool = False,
    model: str = "",
):
    """Run the full MicroKG pipeline."""
    start = time.time()
    print("\n" + "=" * 60)
    print("  🧪 MicroKG — Microplastics Knowledge Graph Pipeline")
    print("=" * 60)

    # ── Step 0: Export ontology ────────────────────────────────
    print("\n📋 Step 0: Exporting ontology schema...")
    ontology_path = Path(__file__).parent.parent / "data" / "ontology.json"
    export_ontology_json(ontology_path)
    print(f"   Saved to {ontology_path}")

    # ── Step 1: Fetch papers ──────────────────────────────────
    print(f"\n📥 Step 1: Fetching papers from PubMed...")
    if skip_fetch:
        papers = load_cached_papers()
        if not papers:
            print("   No cached papers found. Fetching from PubMed...")
            papers = fetch_microplastics_papers(total_target=num_papers)
    else:
        papers = fetch_microplastics_papers(total_target=num_papers)

    if not papers:
        print("❌ No papers fetched. Exiting.")
        return

    # ── Step 2: Store documents ───────────────────────────────
    print(f"\n💾 Step 2: Storing {len(papers)} documents...")
    store = DocumentStore()
    inserted = store.bulk_add_papers(papers)
    print(f"   {inserted} new papers stored")

    # ── Step 3: Extract entities and relations ────────────────
    print(f"\n🔬 Step 3: Extracting entities & relations (Ollama)...")
    print("   This may take a while depending on your hardware...\n")

    all_entities = []
    all_relations = []

    for i, paper in enumerate(tqdm(papers, desc="Processing papers")):
        doc_id = f"pubmed:{paper['pmid']}"
        text = paper.get("abstract", "")
        if not text:
            continue

        # Extract entities
        try:
            entities = extract_entities_from_text(
                text=text, doc_id=doc_id, model=model
            )
        except Exception as e:
            logging.warning(f"Entity extraction failed for {doc_id}: {e}")
            entities = []

        # Add publication entity
        pub_entity = Entity(
            id=f"Publication:{paper['pmid']}",
            entity_type=EntityType.PUBLICATION,
            name=paper.get("title", "Unknown"),
            attributes={
                "doi": paper.get("doi", ""),
                "year": paper.get("year", ""),
                "journal": paper.get("journal", ""),
                "authors": ", ".join(paper.get("authors", [])[:3]),
            },
            source_doc=doc_id,
        )
        entities.append(pub_entity)

        # Create REPORTED_IN relations for all extracted entities
        reported_relations = []
        for entity in entities:
            if entity.entity_type != EntityType.PUBLICATION:
                reported_relations.append(Relation(
                    source_id=entity.id,
                    relation_type="REPORTED_IN",
                    target_id=pub_entity.id,
                    confidence=0.95,
                    evidence=f"Reported in: {paper.get('title', '')}",
                    source_doc=doc_id,
                ))

        # Extract inter-entity relations
        try:
            relations = extract_relations_from_text(
                text=text, entities=entities, doc_id=doc_id, model=model
            )
        except Exception as e:
            logging.warning(f"Relation extraction failed for {doc_id}: {e}")
            relations = []

        all_entities.extend(entities)
        all_relations.extend(relations + reported_relations)

        # Save extraction results
        store.save_extraction_results(
            doc_id=doc_id,
            entities=[e.to_dict() for e in entities],
            relations=[r.to_dict() for r in (relations + reported_relations)],
        )
        store.update_status(doc_id, "extracted")

    print(f"\n   Raw extraction: {len(all_entities)} entities, {len(all_relations)} relations")

    # ── Step 4: Resolve entities ──────────────────────────────
    print(f"\n🔗 Step 4: Resolving & deduplicating entities...")
    resolved_entities = resolve_entities(all_entities, use_embeddings=True)
    print(f"   {len(all_entities)} → {len(resolved_entities)} entities after resolution")

    # Remap relation IDs after resolution
    old_to_new = {}
    for e in all_entities:
        for r in resolved_entities:
            if e.name.lower() == r.name.lower() and e.entity_type == r.entity_type:
                old_to_new[e.id] = r.id
                break

    remapped_relations = []
    for rel in all_relations:
        new_src = old_to_new.get(rel.source_id, rel.source_id)
        new_tgt = old_to_new.get(rel.target_id, rel.target_id)
        remapped_relations.append(Relation(
            source_id=new_src,
            relation_type=rel.relation_type,
            target_id=new_tgt,
            confidence=rel.confidence,
            evidence=rel.evidence,
            source_doc=rel.source_doc,
        ))

    # ── Step 5: Build knowledge graph ─────────────────────────
    print(f"\n🕸️  Step 5: Building knowledge graph...")
    kg = KnowledgeGraph()
    kg.add_entities(resolved_entities)
    kg.add_relations(remapped_relations)

    stats = kg.get_stats()
    print(f"\n   📊 Graph Statistics:")
    print(f"      Nodes: {stats['total_nodes']}")
    print(f"      Edges: {stats['total_edges']}")
    print(f"      Components: {stats['connected_components']}")
    print(f"      Density: {stats['density']:.4f}")
    print(f"\n      Nodes by type:")
    for t, c in sorted(stats['nodes_by_type'].items()):
        print(f"        {t}: {c}")
    print(f"\n      Edges by type:")
    for t, c in sorted(stats['edges_by_type'].items()):
        print(f"        {t}: {c}")

    # ── Step 6: Export graph ──────────────────────────────────
    print(f"\n💾 Step 6: Exporting graph...")
    kg.export_json()
    kg.export_graphml()
    print(f"   JSON: {GRAPH_EXPORT_PATH}")

    elapsed = time.time() - start
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print("=" * 60)

    return kg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroKG Pipeline")
    parser.add_argument("--papers", type=int, default=50, help="Number of papers")
    parser.add_argument("--skip-fetch", action="store_true", help="Use cached papers")
    parser.add_argument("--model", type=str, default="", help="Ollama model name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_pipeline(
        num_papers=args.papers,
        skip_fetch=args.skip_fetch,
        model=args.model,
    )
