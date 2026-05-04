"""
Graph Manager (Pure Python)
=============================
Lightweight knowledge graph storage using pure Python dictionaries.
No external graph library dependency — works with Python 3.14+.
Supports building, querying, and exporting the microplastics KG.
"""

import json, logging
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.schema import Entity, Relation, EntityType, RelationType
from config import GRAPH_EXPORT_PATH, GRAPH_GRAPHML_PATH

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Pure Python knowledge graph for the microplastics domain."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}          # id -> node data
        self._out_edges: dict[str, list[dict]] = defaultdict(list)  # src -> [edge]
        self._in_edges: dict[str, list[dict]] = defaultdict(list)   # tgt -> [edge]

    # ── Build Operations ──────────────────────────────────────

    def add_entity(self, entity: Entity) -> str:
        """Add an entity as a node. Returns the node ID."""
        node_id = entity.id
        self._nodes[node_id] = {
            "id": node_id,
            "entity_type": entity.entity_type,
            "name": entity.name,
            "aliases": entity.aliases,
            "attributes": entity.attributes,
            "confidence": entity.confidence,
            "source_doc": entity.source_doc or "",
        }
        return node_id

    def add_relation(self, relation: Relation) -> bool:
        """Add a relation as a directed edge. Returns True if successful."""
        if relation.source_id not in self._nodes:
            logger.warning(f"Source node not found: {relation.source_id}")
            return False
        if relation.target_id not in self._nodes:
            logger.warning(f"Target node not found: {relation.target_id}")
            return False

        edge = {
            "source": relation.source_id,
            "target": relation.target_id,
            "relation_type": relation.relation_type,
            "confidence": relation.confidence,
            "evidence": relation.evidence,
            "source_doc": relation.source_doc or "",
        }
        self._out_edges[relation.source_id].append(edge)
        self._in_edges[relation.target_id].append(edge)
        return True

    def add_entities(self, entities: list[Entity]):
        """Bulk add entities."""
        for e in entities:
            self.add_entity(e)
        logger.info(f"Added {len(entities)} entities to graph")

    def add_relations(self, relations: list[Relation]):
        """Bulk add relations."""
        added = sum(1 for r in relations if self.add_relation(r))
        logger.info(f"Added {added}/{len(relations)} relations to graph")

    # ── Query Operations ──────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[dict]:
        """Get entity details by ID."""
        return self._nodes.get(entity_id)

    def search_entities(
        self,
        query: str = "",
        entity_type: str = "",
        limit: int = 50,
    ) -> list[dict]:
        """Search entities by name substring and/or type."""
        results = []
        query_lower = query.lower()

        for node_id, data in self._nodes.items():
            if entity_type and data.get("entity_type") != entity_type:
                continue
            if query_lower:
                name = data.get("name", "").lower()
                aliases = [a.lower() for a in data.get("aliases", [])]
                if query_lower not in name and not any(query_lower in a for a in aliases):
                    continue
            results.append(data)
            if len(results) >= limit:
                break

        return results

    def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        direction: str = "both",
    ) -> dict:
        """Get N-hop neighborhood of an entity."""
        if entity_id not in self._nodes:
            return {"center": entity_id, "nodes": [], "edges": []}

        visited = {entity_id}
        frontier = {entity_id}
        all_edges = []

        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                if direction in ("out", "both"):
                    for edge in self._out_edges.get(node, []):
                        all_edges.append(edge)
                        target = edge["target"]
                        if target not in visited:
                            new_frontier.add(target)
                            visited.add(target)
                if direction in ("in", "both"):
                    for edge in self._in_edges.get(node, []):
                        all_edges.append(edge)
                        source = edge["source"]
                        if source not in visited:
                            new_frontier.add(source)
                            visited.add(source)
            frontier = new_frontier

        nodes = [self._nodes[n] for n in visited if n in self._nodes]
        return {"center": entity_id, "nodes": nodes, "edges": all_edges}

    def get_shortest_path(self, source_id: str, target_id: str) -> Optional[dict]:
        """Find shortest path between two entities (BFS on undirected)."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        if source_id == target_id:
            return {"path": [source_id], "nodes": [self._nodes[source_id]], "edges": [], "length": 0}

        # Build undirected adjacency for BFS
        adj: dict[str, set[str]] = defaultdict(set)
        all_edge_lookup: dict[tuple, list[dict]] = defaultdict(list)

        for src, edges in self._out_edges.items():
            for edge in edges:
                tgt = edge["target"]
                adj[src].add(tgt)
                adj[tgt].add(src)
                all_edge_lookup[(src, tgt)].append(edge)

        # BFS
        queue = deque([(source_id, [source_id])])
        visited = {source_id}

        while queue:
            current, path = queue.popleft()
            for neighbor in adj.get(current, set()):
                if neighbor == target_id:
                    full_path = path + [neighbor]
                    # Collect edges along the path
                    edges = []
                    for i in range(len(full_path) - 1):
                        a, b = full_path[i], full_path[i + 1]
                        edge_list = all_edge_lookup.get((a, b)) or all_edge_lookup.get((b, a), [])
                        if edge_list:
                            edges.append(edge_list[0])
                    nodes = [self._nodes[n] for n in full_path if n in self._nodes]
                    return {"path": full_path, "nodes": nodes, "edges": edges, "length": len(full_path) - 1}
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_subgraph_by_type(self, entity_type: str) -> dict:
        """Extract a subgraph containing only nodes of a specific type."""
        node_ids = {nid for nid, d in self._nodes.items() if d.get("entity_type") == entity_type}
        nodes = [self._nodes[nid] for nid in node_ids]
        edges = []
        for src in node_ids:
            for edge in self._out_edges.get(src, []):
                if edge["target"] in node_ids:
                    edges.append(edge)
        return {"entity_type": entity_type, "nodes": nodes, "edges": edges}

    # ── Statistics ────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get graph statistics."""
        total_edges = sum(len(edges) for edges in self._out_edges.values())
        node_types = Counter(d.get("entity_type", "unknown") for d in self._nodes.values())
        edge_types = Counter()
        for edges in self._out_edges.values():
            for e in edges:
                edge_types[e.get("relation_type", "unknown")] += 1

        n = len(self._nodes)
        density = total_edges / (n * (n - 1)) if n > 1 else 0

        # Weakly connected components via BFS
        components = 0
        visited = set()
        adj: dict[str, set[str]] = defaultdict(set)
        for src, edges in self._out_edges.items():
            for edge in edges:
                adj[src].add(edge["target"])
                adj[edge["target"]].add(src)
        for nid in self._nodes:
            if nid not in visited:
                components += 1
                queue = deque([nid])
                while queue:
                    cur = queue.popleft()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    for nb in adj.get(cur, set()):
                        if nb not in visited:
                            queue.append(nb)

        return {
            "total_nodes": n,
            "total_edges": total_edges,
            "nodes_by_type": dict(node_types),
            "edges_by_type": dict(edge_types),
            "density": density,
            "connected_components": components,
        }

    # ── Export / Import ───────────────────────────────────────

    def export_json(self, path: Optional[Path] = None) -> dict:
        """Export graph to JSON format."""
        path = path or GRAPH_EXPORT_PATH
        all_edges = []
        for edges in self._out_edges.values():
            all_edges.extend(edges)

        data = {
            "nodes": list(self._nodes.values()),
            "edges": all_edges,
            "stats": self.get_stats(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Exported graph to {path}")
        return data

    def import_json(self, path: Optional[Path] = None):
        """Import graph from JSON format."""
        path = path or GRAPH_EXPORT_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._nodes.clear()
        self._out_edges.clear()
        self._in_edges.clear()

        for node in data.get("nodes", []):
            nid = node.get("id", "")
            self._nodes[nid] = node

        for edge in data.get("edges", []):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            self._out_edges[src].append(edge)
            self._in_edges[tgt].append(edge)

        logger.info(f"Imported graph: {len(self._nodes)} nodes, {sum(len(e) for e in self._out_edges.values())} edges")

    def export_graphml(self, path: Optional[Path] = None):
        """Export to GraphML XML format for visualization tools."""
        path = path or GRAPH_GRAPHML_PATH
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphstruct.org/graphml">',
            '  <key id="d0" for="node" attr.name="entity_type" attr.type="string"/>',
            '  <key id="d1" for="node" attr.name="name" attr.type="string"/>',
            '  <key id="d2" for="edge" attr.name="relation_type" attr.type="string"/>',
            '  <graph id="G" edgedefault="directed">',
        ]

        for nid, data in self._nodes.items():
            etype = data.get("entity_type", "")
            name = data.get("name", "").replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
            lines.append(f'    <node id="{nid}">')
            lines.append(f'      <data key="d0">{etype}</data>')
            lines.append(f'      <data key="d1">{name}</data>')
            lines.append(f'    </node>')

        edge_id = 0
        for edges in self._out_edges.values():
            for edge in edges:
                rtype = edge.get("relation_type", "")
                lines.append(f'    <edge id="e{edge_id}" source="{edge["source"]}" target="{edge["target"]}">')
                lines.append(f'      <data key="d2">{rtype}</data>')
                lines.append(f'    </edge>')
                edge_id += 1

        lines.append('  </graph>')
        lines.append('</graphml>')

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported GraphML to {path}")


# ── Module-level singleton ────────────────────────────────────
_graph_instance: Optional[KnowledgeGraph] = None


def get_graph() -> KnowledgeGraph:
    """Get or create the singleton KnowledgeGraph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = KnowledgeGraph()
        if GRAPH_EXPORT_PATH.exists():
            _graph_instance.import_json()
    return _graph_instance
