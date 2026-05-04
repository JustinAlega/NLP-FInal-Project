"""
MicroKG Test Run (No Ollama Required)
======================================
Fetches 3 real papers from PubMed, then uses rule-based extraction
to demonstrate the full pipeline and graph output.
"""

import sys, json, re, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, GRAPH_EXPORT_PATH
from ontology.schema import (
    Entity, Relation, EntityType, RelationType,
    CANONICAL_POLYMERS, CANONICAL_DETECTION_METHODS,
    CANONICAL_COMPARTMENTS, CANONICAL_EXPOSURE_PATHWAYS,
    CANONICAL_SIZE_CLASSES, export_ontology_json,
)
from ingestion.pubmed_fetcher import fetch_microplastics_papers, load_cached_papers
from ingestion.document_store import DocumentStore
from extraction.chunker import chunk_document
from extraction.entity_resolver import resolve_entities
from graph.graph_manager import KnowledgeGraph


# ── Rule-based entity extraction (no LLM needed) ─────────────

def extract_entities_rule_based(text: str, doc_id: str) -> list[Entity]:
    """Extract entities using keyword matching against canonical dictionaries."""
    entities = []
    text_lower = text.lower()

    # Search for polymers
    for canonical, aliases in CANONICAL_POLYMERS.items():
        for alias in [canonical] + aliases:
            if alias.lower() in text_lower:
                eid = f"Polymer:{canonical.lower().replace(' ', '_')}"
                entities.append(Entity(
                    id=eid, entity_type=EntityType.POLYMER,
                    name=canonical, aliases=[a for a in aliases if a != canonical],
                    confidence=0.9, source_doc=doc_id,
                ))
                break

    # Search for detection methods
    for canonical, aliases in CANONICAL_DETECTION_METHODS.items():
        for alias in [canonical] + aliases:
            if alias.lower() in text_lower:
                eid = f"DetectionMethod:{canonical.lower().replace(' ', '_').replace('/', '_')}"
                entities.append(Entity(
                    id=eid, entity_type=EntityType.DETECTION_METHOD,
                    name=canonical, confidence=0.85, source_doc=doc_id,
                ))
                break

    # Search for environmental compartments
    for canonical, aliases in CANONICAL_COMPARTMENTS.items():
        for alias in [canonical] + aliases:
            if alias.lower() in text_lower:
                eid = f"EnvironmentalCompartment:{canonical.lower().replace(' ', '_')}"
                entities.append(Entity(
                    id=eid, entity_type=EntityType.ENVIRONMENTAL_COMPARTMENT,
                    name=canonical, confidence=0.85, source_doc=doc_id,
                ))
                break

    # Search for exposure pathways
    for canonical, aliases in CANONICAL_EXPOSURE_PATHWAYS.items():
        for alias in [canonical] + aliases:
            if alias.lower() in text_lower:
                eid = f"ExposurePathway:{canonical.lower().replace(' ', '_')}"
                entities.append(Entity(
                    id=eid, entity_type=EntityType.EXPOSURE_PATHWAY,
                    name=canonical, confidence=0.8, source_doc=doc_id,
                ))
                break

    # Search for size classes
    for canonical, aliases in CANONICAL_SIZE_CLASSES.items():
        for alias in [canonical] + aliases:
            if alias.lower() in text_lower:
                eid = f"SizeClass:{canonical.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                entities.append(Entity(
                    id=eid, entity_type=EntityType.SIZE_CLASS,
                    name=canonical, confidence=0.8, source_doc=doc_id,
                ))
                break

    # Search for common health effects
    health_terms = {
        "Oxidative stress": ["oxidative stress"],
        "Inflammation": ["inflammation", "inflammatory"],
        "Cytotoxicity": ["cytotoxicity", "cytotoxic", "cell death"],
        "Endocrine disruption": ["endocrine disrupt", "hormonal", "estrogenic"],
        "Gut microbiome disruption": ["gut microbi", "intestinal microbi", "dysbiosis"],
        "Reproductive toxicity": ["reproductive", "fertility", "spermatogenesis"],
        "Neurotoxicity": ["neurotox", "neurological", "brain"],
        "Genotoxicity": ["genotox", "DNA damage", "mutagenic"],
        "Immune response": ["immune", "immunotox"],
        "Organ accumulation": ["accumulation", "bioaccumulation"],
    }
    for canonical, keywords in health_terms.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                eid = f"HealthEffect:{canonical.lower().replace(' ', '_')}"
                entities.append(Entity(
                    id=eid, entity_type=EntityType.HEALTH_EFFECT,
                    name=canonical, confidence=0.8, source_doc=doc_id,
                ))
                break

    # Search for sources
    source_terms = {
        "Textile fibers": ["textile", "fiber", "clothing", "laundry"],
        "Packaging": ["packaging", "food container", "plastic bag"],
        "Tire wear": ["tire wear", "tyre", "road dust"],
        "Agricultural film": ["agricultural", "mulch film"],
        "Personal care products": ["cosmetic", "personal care", "microbead"],
        "Industrial discharge": ["industrial", "factory", "manufacturing"],
        "Fishing gear": ["fishing", "aquaculture"],
    }
    for canonical, keywords in source_terms.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                eid = f"Source:{canonical.lower().replace(' ', '_')}"
                entities.append(Entity(
                    id=eid, entity_type=EntityType.SOURCE,
                    name=canonical, confidence=0.75, source_doc=doc_id,
                ))
                break

    return entities


def generate_relations(entities: list[Entity], doc_id: str) -> list[Relation]:
    """Generate plausible relations between co-occurring entities."""
    relations = []

    polymers = [e for e in entities if e.entity_type == EntityType.POLYMER]
    methods = [e for e in entities if e.entity_type == EntityType.DETECTION_METHOD]
    compartments = [e for e in entities if e.entity_type == EntityType.ENVIRONMENTAL_COMPARTMENT]
    health = [e for e in entities if e.entity_type == EntityType.HEALTH_EFFECT]
    pathways = [e for e in entities if e.entity_type == EntityType.EXPOSURE_PATHWAY]
    sources = [e for e in entities if e.entity_type == EntityType.SOURCE]
    sizes = [e for e in entities if e.entity_type == EntityType.SIZE_CLASS]

    for poly in polymers:
        for method in methods:
            relations.append(Relation(
                source_id=poly.id, relation_type=RelationType.DETECTED_BY,
                target_id=method.id, confidence=0.85, source_doc=doc_id,
            ))
        for comp in compartments:
            relations.append(Relation(
                source_id=poly.id, relation_type=RelationType.FOUND_IN,
                target_id=comp.id, confidence=0.85, source_doc=doc_id,
            ))
        for h in health:
            relations.append(Relation(
                source_id=poly.id, relation_type=RelationType.CAUSES,
                target_id=h.id, confidence=0.7, source_doc=doc_id,
            ))
        for src in sources:
            relations.append(Relation(
                source_id=poly.id, relation_type=RelationType.ORIGINATES_FROM,
                target_id=src.id, confidence=0.7, source_doc=doc_id,
            ))
        for sz in sizes:
            relations.append(Relation(
                source_id=poly.id, relation_type=RelationType.HAS_SIZE_CLASS,
                target_id=sz.id, confidence=0.75, source_doc=doc_id,
            ))

    for h in health:
        for pw in pathways:
            relations.append(Relation(
                source_id=h.id, relation_type=RelationType.EXPOSURE_VIA,
                target_id=pw.id, confidence=0.7, source_doc=doc_id,
            ))

    for method in methods:
        for comp in compartments:
            relations.append(Relation(
                source_id=method.id, relation_type=RelationType.MEASURED_IN,
                target_id=comp.id, confidence=0.7, source_doc=doc_id,
            ))

    return relations


# ── Main ──────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  🧪 MicroKG — Test Run (3 Papers, Rule-Based Extraction)")
    print("=" * 60)

    # Step 1: Fetch 3 papers
    print("\n📥 Step 1: Fetching 3 papers from PubMed...")
    papers = fetch_microplastics_papers(
        queries=['"microplastic" AND "health"'],
        max_per_query=3,
        total_target=3,
    )

    if not papers:
        print("❌ Could not fetch papers. Check your internet connection.")
        return

    for i, p in enumerate(papers):
        print(f"\n   Paper {i+1}: {p['title'][:80]}...")
        print(f"   PMID: {p['pmid']} | Year: {p['year']} | Journal: {p['journal'][:40]}")

    # Step 2: Extract entities per paper
    print(f"\n[*] Step 2: Extracting entities (rule-based)...")
    all_raw_entities = []
    paper_entities = {}  # pmid -> [Entity]

    for paper in papers:
        doc_id = f"pubmed:{paper['pmid']}"
        text = paper.get("abstract", "")

        entities = extract_entities_rule_based(text, doc_id)

        pub = Entity(
            id=f"Publication:{paper['pmid']}",
            entity_type=EntityType.PUBLICATION,
            name=paper.get("title", "Unknown")[:100],
            attributes={
                "doi": paper.get("doi", ""),
                "year": paper.get("year", ""),
                "journal": paper.get("journal", ""),
            },
            source_doc=doc_id,
        )
        entities.append(pub)

        print(f"   [{paper['pmid']}]: {len(entities)} entities found")
        paper_entities[paper['pmid']] = entities
        all_raw_entities.extend(entities)

    # Step 3: Resolve/deduplicate entities
    print(f"\n[*] Step 3: Resolving entities...")
    resolved = resolve_entities(all_raw_entities, use_embeddings=False)
    print(f"   {len(all_raw_entities)} raw -> {len(resolved)} resolved")

    # Build a lookup from resolved entities by name+type
    resolved_lookup = {}
    for r in resolved:
        key = (r.entity_type, r.name.lower())
        resolved_lookup[key] = r
        # Also index by aliases
        if hasattr(r, 'aliases'):
            for alias in r.aliases:
                resolved_lookup[(r.entity_type, alias.lower())] = r

    # Step 4: Build graph and generate relations using resolved IDs
    print(f"\n[*] Step 4: Building knowledge graph...")
    kg = KnowledgeGraph()
    kg.add_entities(resolved)

    # Now generate relations using the RESOLVED entity IDs
    all_relations = []
    for paper in papers:
        doc_id = f"pubmed:{paper['pmid']}"
        raw_ents = paper_entities[paper['pmid']]

        # Map each raw entity to its resolved version
        mapped = []
        for e in raw_ents:
            key = (e.entity_type, e.name.lower())
            r = resolved_lookup.get(key)
            if r:
                mapped.append(r)

        # Generate relations using resolved entities
        relations = generate_relations(mapped, doc_id)

        # REPORTED_IN: link each non-publication entity to its paper
        pub_key = (EntityType.PUBLICATION, paper.get("title", "")[:100].lower())
        pub_entity = resolved_lookup.get(pub_key)
        if pub_entity:
            for e in mapped:
                if e.entity_type != EntityType.PUBLICATION:
                    relations.append(Relation(
                        source_id=e.id, relation_type=RelationType.REPORTED_IN,
                        target_id=pub_entity.id, confidence=0.95, source_doc=doc_id,
                    ))

        all_relations.extend(relations)

    kg.add_relations(all_relations)
    print(f"   Added {len(all_relations)} relations")

    stats = kg.get_stats()
    print(f"\n   {'─' * 50}")
    print(f"   📊 KNOWLEDGE GRAPH STATISTICS")
    print(f"   {'─' * 50}")
    print(f"   Total Nodes: {stats['total_nodes']}")
    print(f"   Total Edges: {stats['total_edges']}")
    print(f"   Connected Components: {stats['connected_components']}")
    print(f"   Density: {stats['density']:.4f}")
    print(f"\n   📦 Nodes by Type:")
    for t, c in sorted(stats['nodes_by_type'].items()):
        emoji = {
            "Polymer": "🧬", "DetectionMethod": "🔬", "EnvironmentalCompartment": "🌊",
            "HealthEffect": "💊", "ExposurePathway": "🚶", "Publication": "📄",
            "Source": "🏭", "SizeClass": "📏",
        }.get(t, "•")
        print(f"      {emoji} {t}: {c}")
    print(f"\n   🔗 Edges by Type:")
    for t, c in sorted(stats['edges_by_type'].items()):
        print(f"      → {t}: {c}")

    # Step 5: Show graph relationships
    print(f"\n   {'─' * 50}")
    print(f"   🔍 SAMPLE RELATIONSHIPS")
    print(f"   {'─' * 50}")

    for entity in resolved:
        if entity.entity_type == EntityType.POLYMER:
            neighbors = kg.get_neighbors(entity.id, hops=1)
            if neighbors['edges']:
                print(f"\n   🧬 {entity.name}:")
                for edge in neighbors['edges'][:8]:
                    src_node = kg.get_entity(edge['source'])
                    tgt_node = kg.get_entity(edge['target'])
                    src_name = src_node['name'] if src_node else edge['source']
                    tgt_name = tgt_node['name'] if tgt_node else edge['target']
                    if edge['source'] == entity.id:
                        print(f"      ──[{edge['relation_type']}]──▶ {tgt_name} ({tgt_node.get('entity_type', '') if tgt_node else ''})")
                    else:
                        print(f"      ◀──[{edge['relation_type']}]── {src_name} ({src_node.get('entity_type', '') if src_node else ''})")

    # Step 6: Export
    print(f"\n   {'─' * 50}")
    print(f"   💾 EXPORTING GRAPH")
    print(f"   {'─' * 50}")
    graph_data = kg.export_json()
    kg.export_graphml()
    print(f"   JSON: {GRAPH_EXPORT_PATH}")
    print(f"   GraphML: {GRAPH_EXPORT_PATH.with_suffix('.graphml')}")
    print(f"   Nodes in export: {len(graph_data['nodes'])}")
    print(f"   Edges in export: {len(graph_data['edges'])}")

    # Print a few node details
    print(f"\n   {'─' * 50}")
    print(f"   📋 ALL ENTITIES IN GRAPH")
    print(f"   {'─' * 50}")
    for node in graph_data['nodes']:
        etype = node.get('entity_type', '')
        name = node.get('name', '')
        conf = node.get('confidence', 0)
        print(f"   [{etype:30s}] {name[:60]:60s} (conf: {conf:.2f})")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Test run complete!")
    print(f"  Graph exported to: {GRAPH_EXPORT_PATH}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
