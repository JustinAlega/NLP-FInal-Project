"""
Hybrid Retriever
=================
Combines graph traversal and vector similarity search for
knowledge-grounded retrieval over the microplastics KG.
"""

import logging
from typing import Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph_manager import KnowledgeGraph, get_graph
from rag.embeddings import VectorIndex
from ontology.schema import EntityType

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves context from both the KG (graph) and vector index."""

    def __init__(self, kg: Optional[KnowledgeGraph] = None):
        self.kg = kg or get_graph()
        self.vector_index = VectorIndex()
        self._indexed = False

    def build_index(self):
        """Build vector index from graph node descriptions."""
        texts, metas = [], []
        for node_id, data in self.kg.graph.nodes(data=True):
            name = data.get("name", "")
            etype = data.get("entity_type", "")
            attrs = data.get("attributes", {})
            attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items()) if isinstance(attrs, dict) else str(attrs)
            description = f"{etype}: {name}. {attr_str}".strip()
            texts.append(description)
            metas.append({"node_id": node_id, "name": name, "entity_type": etype, "text": description})

        if texts:
            self.vector_index.add_batch(texts, metas)
            self._indexed = True
            logger.info(f"Built vector index with {len(texts)} entries")

    def retrieve(self, query: str, top_k: int = 10, hops: int = 1) -> dict:
        """
        Hybrid retrieval: vector search + graph traversal.

        Args:
            query: Natural language query
            top_k: Number of top vector matches
            hops: Graph traversal depth from matched nodes

        Returns:
            Dict with matched_nodes, graph_context, and combined_context
        """
        if not self._indexed:
            self.build_index()

        # Step 1: Vector search for relevant nodes
        vector_results = self.vector_index.search(query, top_k=top_k)

        # Step 2: Graph traversal from top matches
        graph_context = []
        seen_nodes = set()
        seen_edges = set()

        for result in vector_results[:5]:  # Top 5 for graph expansion
            node_id = result.get("node_id")
            if not node_id or node_id in seen_nodes:
                continue

            neighborhood = self.kg.get_neighbors(node_id, hops=hops)
            for node in neighborhood.get("nodes", []):
                nid = node.get("id")
                if nid not in seen_nodes:
                    seen_nodes.add(nid)
                    graph_context.append(node)
            for edge in neighborhood.get("edges", []):
                edge_key = (edge["source"], edge.get("relation_type", ""), edge["target"])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)

        # Step 3: Format context as natural language triples
        triples = []
        for edge_key in seen_edges:
            src_name = self._get_node_name(edge_key[0])
            tgt_name = self._get_node_name(edge_key[2])
            triples.append(f"{src_name} --[{edge_key[1]}]--> {tgt_name}")

        return {
            "query": query,
            "vector_matches": vector_results,
            "graph_nodes": graph_context,
            "triples": triples,
            "combined_context": self._format_context(vector_results, triples),
        }

    def _get_node_name(self, node_id: str) -> str:
        node = self.kg.get_entity(node_id)
        return node.get("name", node_id) if node else node_id

    def _format_context(self, vector_results: list, triples: list) -> str:
        """Format retrieval results as context string for the LLM."""
        parts = []

        if triples:
            parts.append("KNOWLEDGE GRAPH RELATIONSHIPS:")
            for t in triples[:20]:
                parts.append(f"  • {t}")

        if vector_results:
            parts.append("\nRELEVANT ENTITIES:")
            for r in vector_results[:10]:
                parts.append(f"  • {r.get('text', '')} (relevance: {r.get('score', 0):.2f})")

        return "\n".join(parts)
