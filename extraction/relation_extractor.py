"""
Relation Extractor (Ollama)
============================
LLM-based relation extraction for the microplastics domain.
"""

import json, logging, re
from typing import Optional
import ollama
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_MODEL, EXTRACTION_TEMPERATURE, EXTRACTION_MAX_RETRIES
from ontology.schema import (
    RelationType, Relation, Entity,
    RELATION_CONSTRAINTS, RELATION_TYPE_DESCRIPTIONS,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a relation extraction system for microplastics research.
Given text and extracted entities, identify relationships between them.

VALID RELATIONS (source_type -> target_type):
- DETECTED_BY: Polymer -> DetectionMethod
- FOUND_IN: Polymer -> EnvironmentalCompartment
- ORIGINATES_FROM: Polymer -> Source
- HAS_SIZE_CLASS: Polymer -> SizeClass
- CAUSES: Polymer -> HealthEffect
- EXPOSURE_VIA: HealthEffect -> ExposurePathway
- LEADS_TO: HealthEffect -> HealthEffect
- REPORTED_IN: any -> Publication
- CO_OCCURS_WITH: Polymer -> Polymer
- AFFECTS: Polymer -> EnvironmentalCompartment
- REGULATED_BY: Polymer -> Publication
- MEASURED_IN: DetectionMethod -> EnvironmentalCompartment

RULES:
1. Only relate entities from the provided list.
2. Respect source->target type constraints.
3. Only extract relationships explicitly supported by the text.
4. Return JSON array only.

OUTPUT FORMAT:
[{"source_entity":"<name>","source_type":"<type>","relation_type":"<type>","target_entity":"<name>","target_type":"<type>","confidence":0.9,"evidence":"<text span>"}]

EXAMPLE:
Entities: Polyethylene(Polymer), FTIR(DetectionMethod), Wastewater(EnvironmentalCompartment)
Text: "PE was detected in wastewater using FTIR."
Output: [{"source_entity":"Polyethylene","source_type":"Polymer","relation_type":"DETECTED_BY","target_entity":"FTIR","target_type":"DetectionMethod","confidence":0.95,"evidence":"detected...using FTIR"},{"source_entity":"Polyethylene","source_type":"Polymer","relation_type":"FOUND_IN","target_entity":"Wastewater","target_type":"EnvironmentalCompartment","confidence":0.9,"evidence":"detected in wastewater"}]
"""


def extract_relations_from_text(text, entities, doc_id="", model=""):
    if len(entities) < 2:
        return []
    model = model or OLLAMA_MODEL
    entity_list = "\n".join(f"- {e.name} ({e.entity_type})" for e in entities)
    user_prompt = f"ENTITIES:\n{entity_list}\n\nTEXT:\n\"{text}\"\n\nReturn ONLY a valid JSON array."
    relations = []
    for attempt in range(EXTRACTION_MAX_RETRIES):
        try:
            response = ollama.chat(model=model, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ], options={"temperature": EXTRACTION_TEMPERATURE, "num_predict": 2048})
            content = response["message"]["content"].strip()
            raw_relations = _parse_json(content)
            entity_map = {e.name.lower(): e for e in entities}
            for raw in raw_relations:
                rel = _validate_relation(raw, entity_map, doc_id)
                if rel:
                    relations.append(rel)
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
    return relations


def _parse_json(content):
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content).strip()
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        content = match.group(0)
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _validate_relation(raw, entity_map, doc_id):
    try:
        rel_type = raw.get("relation_type", "")
        if rel_type not in {rt.value for rt in RelationType}:
            return None
        src_name = raw.get("source_entity", "").strip()
        tgt_name = raw.get("target_entity", "").strip()
        if not src_name or not tgt_name:
            return None
        src = entity_map.get(src_name.lower())
        tgt = entity_map.get(tgt_name.lower())
        if not src:
            for k, v in entity_map.items():
                if src_name.lower() in k or k in src_name.lower():
                    src = v; break
        if not tgt:
            for k, v in entity_map.items():
                if tgt_name.lower() in k or k in tgt_name.lower():
                    tgt = v; break
        if not src or not tgt:
            return None
        constraint = RELATION_CONSTRAINTS.get(RelationType(rel_type))
        if constraint:
            src_types = {t.value for t in constraint["source_types"]}
            tgt_types = {t.value for t in constraint["target_types"]}
            if src.entity_type not in src_types or tgt.entity_type not in tgt_types:
                return None
        return Relation(
            source_id=src.id, relation_type=rel_type, target_id=tgt.id,
            confidence=float(raw.get("confidence", 0.7)),
            evidence=raw.get("evidence", ""), source_doc=doc_id,
        )
    except Exception:
        return None


def extract_relations_batch(chunks_with_entities, model="", progress=True):
    from tqdm import tqdm
    all_relations = []
    it = tqdm(chunks_with_entities, desc="Extracting relations") if progress else chunks_with_entities
    for chunk, entities in it:
        rels = extract_relations_from_text(chunk.text, entities, chunk.doc_id, model)
        all_relations.extend(rels)
    return all_relations
