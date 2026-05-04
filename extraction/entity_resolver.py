"""
Entity Resolver
================
Normalizes and deduplicates extracted entities using canonical dictionaries
and embedding-based fuzzy matching.
"""

import logging, re
from collections import defaultdict
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.schema import (
    Entity, EntityType,
    CANONICAL_POLYMERS, CANONICAL_SIZE_CLASSES,
    CANONICAL_DETECTION_METHODS, CANONICAL_COMPARTMENTS,
    CANONICAL_EXPOSURE_PATHWAYS,
)
from config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Build reverse lookup: alias -> canonical name
ALIAS_MAP = {}
for canonical_dict in [
    CANONICAL_POLYMERS, CANONICAL_SIZE_CLASSES,
    CANONICAL_DETECTION_METHODS, CANONICAL_COMPARTMENTS,
    CANONICAL_EXPOSURE_PATHWAYS,
]:
    for canonical, aliases in canonical_dict.items():
        ALIAS_MAP[canonical.lower()] = canonical
        for alias in aliases:
            ALIAS_MAP[alias.lower()] = canonical


class EntityResolver:
    """Resolves and deduplicates entities using canonical names and embeddings."""

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self._embed_model = None
        self._entity_embeddings = {}

    def _get_embed_model(self):
        if self._embed_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                from config import EMBEDDING_MODEL
                self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self.use_embeddings = False
        return self._embed_model

    def resolve_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        Resolve a list of entities: normalize names, merge duplicates.

        Args:
            entities: Raw extracted entities

        Returns:
            Deduplicated and normalized entities
        """
        # Step 1: Normalize names using canonical dictionaries
        normalized = [self._normalize_entity(e) for e in entities]

        # Step 2: Group by (entity_type, normalized_name)
        groups = defaultdict(list)
        for entity in normalized:
            key = (entity.entity_type, entity.name.lower())
            groups[key].append(entity)

        # Step 3: Merge groups
        merged = []
        for (etype, name), group in groups.items():
            merged_entity = self._merge_entity_group(group)
            merged.append(merged_entity)

        # Step 4: Embedding-based fuzzy merge (for near-duplicates)
        if self.use_embeddings and len(merged) > 1:
            merged = self._embedding_merge(merged)

        logger.info(
            f"Entity resolution: {len(entities)} raw → {len(merged)} resolved"
        )
        return merged

    def _normalize_entity(self, entity: Entity) -> Entity:
        """Normalize entity name using canonical dictionaries."""
        name_lower = entity.name.lower().strip()

        # Check alias map
        canonical = ALIAS_MAP.get(name_lower)
        if canonical:
            entity.name = canonical
            if name_lower != canonical.lower():
                if name_lower not in [a.lower() for a in entity.aliases]:
                    entity.aliases.append(entity.name)

        # Regenerate ID with normalized name
        norm_name = re.sub(r"[^a-z0-9\s\-μ]", "", entity.name.lower())
        norm_name = re.sub(r"\s+", "_", norm_name)
        entity.id = f"{entity.entity_type}:{norm_name}"

        return entity

    def _merge_entity_group(self, group: list[Entity]) -> Entity:
        """Merge multiple entity records into one."""
        if len(group) == 1:
            return group[0]

        # Use the entity with highest confidence as base
        base = max(group, key=lambda e: e.confidence)

        # Collect all aliases
        all_aliases = set()
        all_attrs = {}
        source_docs = set()

        for entity in group:
            all_aliases.update(entity.aliases)
            all_aliases.add(entity.name)
            all_attrs.update(entity.attributes)
            if entity.source_doc:
                source_docs.add(entity.source_doc)

        # Remove the canonical name from aliases
        all_aliases.discard(base.name)

        base.aliases = list(all_aliases)
        base.attributes = all_attrs
        # Average confidence
        base.confidence = sum(e.confidence for e in group) / len(group)

        return base

    def _embedding_merge(self, entities: list[Entity]) -> list[Entity]:
        """Merge entities with similar names using embeddings."""
        model = self._get_embed_model()
        if model is None:
            return entities

        import numpy as np

        # Group by entity type first (only merge within same type)
        type_groups = defaultdict(list)
        for e in entities:
            type_groups[e.entity_type].append(e)

        merged_all = []
        for etype, group in type_groups.items():
            if len(group) <= 1:
                merged_all.extend(group)
                continue

            names = [e.name for e in group]
            embeddings = model.encode(names)

            # Find pairs above similarity threshold
            merged_indices = set()
            merge_map = {}  # idx -> list of indices to merge

            for i in range(len(group)):
                if i in merged_indices:
                    continue
                merge_map[i] = [i]
                for j in range(i + 1, len(group)):
                    if j in merged_indices:
                        continue
                    sim = float(np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    ))
                    if sim >= SIMILARITY_THRESHOLD:
                        merge_map[i].append(j)
                        merged_indices.add(j)

            for base_idx, merge_indices in merge_map.items():
                merge_group = [group[i] for i in merge_indices]
                merged_entity = self._merge_entity_group(merge_group)
                merged_all.append(merged_entity)

        return merged_all


def resolve_entities(entities: list[Entity], use_embeddings: bool = True) -> list[Entity]:
    """Convenience function for entity resolution."""
    resolver = EntityResolver(use_embeddings=use_embeddings)
    return resolver.resolve_entities(entities)
