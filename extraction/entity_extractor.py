"""
Entity Extractor (Ollama)
=========================
LLM-based named entity extraction for microplastics domain.
Uses Ollama for local inference with structured JSON output
constrained by the domain ontology schema.
"""

import json
import logging
import re
from typing import Optional

import ollama

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_MODEL, EXTRACTION_TEMPERATURE, EXTRACTION_MAX_RETRIES
from ontology.schema import (
    EntityType, Entity, ENTITY_TYPE_DESCRIPTIONS,
    CANONICAL_POLYMERS, CANONICAL_SIZE_CLASSES,
    CANONICAL_DETECTION_METHODS, CANONICAL_COMPARTMENTS,
    CANONICAL_EXPOSURE_PATHWAYS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# System Prompt for Entity Extraction
# ═══════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a scientific information extraction system specialized in microplastics research.

Your task is to extract named entities from the given text. You MUST only extract entities that belong to the following types:

ENTITY TYPES:
{entity_types}

RULES:
1. Extract ONLY entities explicitly mentioned or clearly implied in the text.
2. Do NOT invent or hallucinate entities not supported by the text.
3. For polymers, use the full chemical name (e.g., "Polyethylene" not just "PE"), and include abbreviations as an attribute.
4. For size classes, include numerical ranges if mentioned (e.g., min_um, max_um).
5. For detection methods, classify as spectroscopic, thermal, visual, or chemical.
6. For health effects, note the organ system and severity if mentioned.
7. For sources, categorize as industrial, domestic, or agricultural.
8. Return results as a JSON array.

OUTPUT FORMAT (JSON array):
[
  {{
    "entity_type": "<one of the entity types above>",
    "name": "<canonical entity name>",
    "attributes": {{"key": "value"}},
    "confidence": <0.0-1.0>,
    "text_span": "<exact text that mentions this entity>"
  }}
]

If no entities are found, return an empty array: []
"""

FEW_SHOT_EXAMPLES = """
EXAMPLE INPUT:
"Polyethylene (PE) and polypropylene (PP) microplastics (100–500 μm) were detected in municipal wastewater effluent using micro-FTIR spectroscopy. These particles were associated with oxidative stress in exposed zebrafish."

EXAMPLE OUTPUT:
[
  {"entity_type": "Polymer", "name": "Polyethylene", "attributes": {"abbreviation": "PE"}, "confidence": 0.95, "text_span": "Polyethylene (PE)"},
  {"entity_type": "Polymer", "name": "Polypropylene", "attributes": {"abbreviation": "PP"}, "confidence": 0.95, "text_span": "polypropylene (PP)"},
  {"entity_type": "SizeClass", "name": "Small microplastic (100-500 μm)", "attributes": {"min_um": 100, "max_um": 500}, "confidence": 0.9, "text_span": "100–500 μm"},
  {"entity_type": "DetectionMethod", "name": "FTIR Spectroscopy", "attributes": {"type": "spectroscopic", "variant": "micro-FTIR"}, "confidence": 0.95, "text_span": "micro-FTIR spectroscopy"},
  {"entity_type": "EnvironmentalCompartment", "name": "Wastewater", "attributes": {"subtype": "effluent"}, "confidence": 0.9, "text_span": "municipal wastewater effluent"},
  {"entity_type": "HealthEffect", "name": "Oxidative stress", "attributes": {"organism": "zebrafish"}, "confidence": 0.85, "text_span": "oxidative stress in exposed zebrafish"}
]
"""


def _build_system_prompt() -> str:
    """Build the full system prompt with entity type descriptions."""
    type_descriptions = "\n".join(
        f"- {et.value}: {desc}"
        for et, desc in ENTITY_TYPE_DESCRIPTIONS.items()
    )
    prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(entity_types=type_descriptions)
    prompt += "\n" + FEW_SHOT_EXAMPLES
    return prompt


def extract_entities_from_text(
    text: str,
    doc_id: str = "",
    model: str = "",
) -> list[Entity]:
    """
    Extract entities from a text chunk using Ollama LLM.

    Args:
        text: Input text to extract entities from
        doc_id: Source document ID for provenance
        model: Ollama model name (defaults to config)

    Returns:
        List of Entity objects
    """
    model = model or OLLAMA_MODEL
    system_prompt = _build_system_prompt()

    user_prompt = f"""Extract all microplastics-related entities from the following scientific text.

TEXT:
\"\"\"{text}\"\"\"

Return ONLY a valid JSON array of entities. No markdown, no explanation."""

    entities = []

    for attempt in range(EXTRACTION_MAX_RETRIES):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": EXTRACTION_TEMPERATURE,
                    "num_predict": 2048,
                },
            )

            content = response["message"]["content"].strip()
            raw_entities = _parse_json_response(content)

            for raw in raw_entities:
                entity = _validate_and_create_entity(raw, doc_id)
                if entity:
                    entities.append(entity)

            logger.info(
                f"Extracted {len(entities)} entities from chunk "
                f"(doc: {doc_id}, attempt: {attempt + 1})"
            )
            break

        except Exception as e:
            logger.warning(
                f"Extraction attempt {attempt + 1} failed: {e}"
            )
            if attempt == EXTRACTION_MAX_RETRIES - 1:
                logger.error(f"All extraction attempts failed for doc {doc_id}")

    return entities


def _parse_json_response(content: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code fences if present
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

    # Find JSON array in the response
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        content = match.group(0)

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return []


def _validate_and_create_entity(
    raw: dict, doc_id: str
) -> Optional[Entity]:
    """Validate raw entity dict and create an Entity object."""
    try:
        entity_type = raw.get("entity_type", "")

        # Validate entity type
        valid_types = {et.value for et in EntityType}
        if entity_type not in valid_types:
            logger.debug(f"Skipping invalid entity type: {entity_type}")
            return None

        name = raw.get("name", "").strip()
        if not name:
            return None

        # Generate stable ID
        entity_id = f"{entity_type}:{_normalize_name(name)}"

        return Entity(
            id=entity_id,
            entity_type=entity_type,
            name=name,
            attributes=raw.get("attributes", {}),
            confidence=float(raw.get("confidence", 0.8)),
            source_doc=doc_id,
        )

    except Exception as e:
        logger.debug(f"Failed to create entity from {raw}: {e}")
        return None


def _normalize_name(name: str) -> str:
    """Normalize entity name for ID generation."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s\-μ]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def extract_entities_batch(
    chunks: list,  # list of Chunk objects
    model: str = "",
    progress: bool = True,
) -> list[Entity]:
    """
    Extract entities from multiple text chunks.

    Args:
        chunks: List of Chunk objects from the chunker
        model: Ollama model name
        progress: Show progress bar

    Returns:
        Aggregated list of all extracted entities
    """
    from tqdm import tqdm

    all_entities = []
    iterator = tqdm(chunks, desc="Extracting entities") if progress else chunks

    for chunk in iterator:
        entities = extract_entities_from_text(
            text=chunk.text,
            doc_id=chunk.doc_id,
            model=model,
        )
        all_entities.extend(entities)

    logger.info(f"Total entities extracted: {len(all_entities)}")
    return all_entities
