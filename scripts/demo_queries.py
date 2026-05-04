"""
Demo Queries
=============
Demonstrates MicroKG capabilities with example queries.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph_manager import KnowledgeGraph, get_graph
from config import GRAPH_EXPORT_PATH


def run_demo():
    """Run demo queries against the knowledge graph."""
    print("\n" + "=" * 60)
    print("  🔬 MicroKG — Demo Queries")
    print("=" * 60)

    kg = get_graph()
    stats = kg.get_stats()

    if stats["total_nodes"] == 0:
        print("\n⚠️  Graph is empty. Run 'python scripts/seed_kg.py' first.")
        return

    print(f"\n📊 Graph loaded: {stats['total_nodes']} nodes, {stats['total_edges']} edges\n")

    # ── Demo 1: Search for polymers ───────────────────────────
    print("─" * 50)
    print("🔍 Demo 1: Search for all Polymer entities")
    print("─" * 50)
    polymers = kg.search_entities(entity_type="Polymer")
    for p in polymers[:10]:
        print(f"   • {p['name']} (confidence: {p.get('confidence', 0):.2f})")

    # ── Demo 2: Search for detection methods ──────────────────
    print(f"\n{'─' * 50}")
    print("🔍 Demo 2: Search for Detection Methods")
    print("─" * 50)
    methods = kg.search_entities(entity_type="DetectionMethod")
    for m in methods[:10]:
        print(f"   • {m['name']}")

    # ── Demo 3: Neighbors of Polyethylene ─────────────────────
    print(f"\n{'─' * 50}")
    print("🔍 Demo 3: What is connected to Polyethylene?")
    print("─" * 50)
    pe_results = kg.search_entities(query="polyethylene", entity_type="Polymer")
    if pe_results:
        pe_id = pe_results[0]["id"]
        neighbors = kg.get_neighbors(pe_id, hops=1)
        print(f"   Center: {pe_results[0]['name']}")
        for edge in neighbors.get("edges", [])[:10]:
            src_node = kg.get_entity(edge["source"])
            tgt_node = kg.get_entity(edge["target"])
            src_name = src_node["name"] if src_node else edge["source"]
            tgt_name = tgt_node["name"] if tgt_node else edge["target"]
            print(f"   {src_name} --[{edge.get('relation_type', '?')}]--> {tgt_name}")
    else:
        print("   Polyethylene not found in graph")

    # ── Demo 4: Health effects ────────────────────────────────
    print(f"\n{'─' * 50}")
    print("🔍 Demo 4: Health effects in the knowledge graph")
    print("─" * 50)
    effects = kg.search_entities(entity_type="HealthEffect")
    for e in effects[:10]:
        print(f"   • {e['name']}")

    # ── Demo 5: Environmental compartments ────────────────────
    print(f"\n{'─' * 50}")
    print("🔍 Demo 5: Environmental compartments")
    print("─" * 50)
    comps = kg.search_entities(entity_type="EnvironmentalCompartment")
    for c in comps[:10]:
        print(f"   • {c['name']}")

    # ── Demo 6: QA (if Ollama is available) ───────────────────
    print(f"\n{'─' * 50}")
    print("🤖 Demo 6: RAG-powered Question Answering")
    print("─" * 50)
    try:
        from rag.qa_chain import QAChain
        qa = QAChain(kg)

        demo_questions = [
            "What polymers are most commonly found in drinking water?",
            "What detection methods are used for microplastics?",
            "What health effects are associated with microplastic exposure?",
        ]

        for q in demo_questions:
            print(f"\n   ❓ {q}")
            result = qa.ask(q)
            answer_lines = result["answer"].split("\n")
            for line in answer_lines[:5]:
                print(f"   💡 {line}")
            print(f"   📚 ({result['triples_used']} triples used)")

    except Exception as e:
        print(f"   ⚠️  QA unavailable (Ollama not running?): {e}")

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
