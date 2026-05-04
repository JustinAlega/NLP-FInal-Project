"""
MicroKG Ontology Schema
========================
Defines entity types, relationship types, and their constraints for the
microplastics domain knowledge graph.

Entity Types (8):
    Polymer, SizeClass, Source, DetectionMethod,
    EnvironmentalCompartment, HealthEffect, ExposurePathway, Publication

Relationship Types (12):
    DETECTED_BY, FOUND_IN, ORIGINATES_FROM, HAS_SIZE_CLASS,
    CAUSES, EXPOSURE_VIA, LEADS_TO, REPORTED_IN,
    CO_OCCURS_WITH, AFFECTS, REGULATED_BY, MEASURED_IN
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
import json
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# Entity Type Definitions
# ═══════════════════════════════════════════════════════════════

class EntityType(str, Enum):
    """The 8 core entity types in the microplastics KG."""
    POLYMER = "Polymer"
    SIZE_CLASS = "SizeClass"
    SOURCE = "Source"
    DETECTION_METHOD = "DetectionMethod"
    ENVIRONMENTAL_COMPARTMENT = "EnvironmentalCompartment"
    HEALTH_EFFECT = "HealthEffect"
    EXPOSURE_PATHWAY = "ExposurePathway"
    PUBLICATION = "Publication"


class RelationType(str, Enum):
    """The 12 relationship types connecting entities."""
    DETECTED_BY = "DETECTED_BY"
    FOUND_IN = "FOUND_IN"
    ORIGINATES_FROM = "ORIGINATES_FROM"
    HAS_SIZE_CLASS = "HAS_SIZE_CLASS"
    CAUSES = "CAUSES"
    EXPOSURE_VIA = "EXPOSURE_VIA"
    LEADS_TO = "LEADS_TO"
    REPORTED_IN = "REPORTED_IN"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    AFFECTS = "AFFECTS"
    REGULATED_BY = "REGULATED_BY"
    MEASURED_IN = "MEASURED_IN"


# ═══════════════════════════════════════════════════════════════
# Entity Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """Base entity in the knowledge graph."""
    id: str                         # Unique identifier (type:normalized_name)
    entity_type: str                # EntityType value
    name: str                       # Canonical name
    aliases: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    confidence: float = 1.0
    source_doc: Optional[str] = None  # DOI or document ID

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(**data)


@dataclass
class Relation:
    """A directed relationship between two entities."""
    source_id: str                  # Source entity ID
    relation_type: str              # RelationType value
    target_id: str                  # Target entity ID
    confidence: float = 1.0
    evidence: str = ""              # Text span supporting this relation
    source_doc: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        return cls(**data)


# ═══════════════════════════════════════════════════════════════
# Ontology Constraints (valid source→target for each relation)
# ═══════════════════════════════════════════════════════════════

RELATION_CONSTRAINTS: dict[str, dict[str, list[str]]] = {
    RelationType.DETECTED_BY: {
        "source_types": [EntityType.POLYMER],
        "target_types": [EntityType.DETECTION_METHOD],
    },
    RelationType.FOUND_IN: {
        "source_types": [EntityType.POLYMER],
        "target_types": [EntityType.ENVIRONMENTAL_COMPARTMENT],
    },
    RelationType.ORIGINATES_FROM: {
        "source_types": [EntityType.POLYMER],
        "target_types": [EntityType.SOURCE],
    },
    RelationType.HAS_SIZE_CLASS: {
        "source_types": [EntityType.POLYMER],
        "target_types": [EntityType.SIZE_CLASS],
    },
    RelationType.CAUSES: {
        "source_types": [EntityType.POLYMER],
        "target_types": [EntityType.HEALTH_EFFECT],
    },
    RelationType.EXPOSURE_VIA: {
        "source_types": [EntityType.HEALTH_EFFECT],
        "target_types": [EntityType.EXPOSURE_PATHWAY],
    },
    RelationType.LEADS_TO: {
        "source_types": [EntityType.HEALTH_EFFECT],
        "target_types": [EntityType.HEALTH_EFFECT],
    },
    RelationType.REPORTED_IN: {
        "source_types": [
            EntityType.POLYMER, EntityType.SIZE_CLASS, EntityType.SOURCE,
            EntityType.DETECTION_METHOD, EntityType.ENVIRONMENTAL_COMPARTMENT,
            EntityType.HEALTH_EFFECT, EntityType.EXPOSURE_PATHWAY,
        ],
        "target_types": [EntityType.PUBLICATION],
    },
    RelationType.CO_OCCURS_WITH: {
        "source_types": [EntityType.POLYMER],
        "target_types": [EntityType.POLYMER],
    },
    RelationType.AFFECTS: {
        "source_types": [EntityType.POLYMER],
        "target_types": [EntityType.ENVIRONMENTAL_COMPARTMENT],
    },
    RelationType.REGULATED_BY: {
        "source_types": [EntityType.POLYMER],
        "target_types": [EntityType.PUBLICATION],
    },
    RelationType.MEASURED_IN: {
        "source_types": [EntityType.DETECTION_METHOD],
        "target_types": [EntityType.ENVIRONMENTAL_COMPARTMENT],
    },
}


# ═══════════════════════════════════════════════════════════════
# Canonical Entity Dictionaries (for entity resolution)
# ═══════════════════════════════════════════════════════════════

CANONICAL_POLYMERS = {
    "Polyethylene": ["PE", "HDPE", "LDPE", "LLDPE", "high-density polyethylene",
                     "low-density polyethylene", "polyethylene"],
    "Polypropylene": ["PP", "polypropylene"],
    "Polystyrene": ["PS", "EPS", "expanded polystyrene", "polystyrene"],
    "Polyethylene terephthalate": ["PET", "PETE", "polyester",
                                    "polyethylene terephthalate"],
    "Polyvinyl chloride": ["PVC", "vinyl", "polyvinyl chloride"],
    "Polyamide": ["PA", "nylon", "PA6", "PA66", "polyamide"],
    "Polymethyl methacrylate": ["PMMA", "acrylic", "plexiglass",
                                 "polymethyl methacrylate"],
    "Polyurethane": ["PU", "PUR", "polyurethane"],
    "Polycarbonate": ["PC", "polycarbonate"],
    "Polytetrafluoroethylene": ["PTFE", "Teflon", "polytetrafluoroethylene"],
    "Cellulose acetate": ["CA", "cellulose acetate"],
    "Acrylonitrile butadiene styrene": ["ABS"],
}

CANONICAL_SIZE_CLASSES = {
    "Macroplastic (>25 mm)": ["macroplastic", ">25mm", ">25 mm"],
    "Mesoplastic (5-25 mm)": ["mesoplastic", "5-25mm", "5-25 mm"],
    "Large microplastic (1-5 mm)": ["large microplastic", "1-5mm", "1-5 mm"],
    "Small microplastic (100 μm - 1 mm)": ["small microplastic", "100-1000μm"],
    "Fine microplastic (1-100 μm)": ["fine microplastic", "1-100μm"],
    "Nanoplastic (<1 μm)": ["nanoplastic", "nano-plastic", "<1μm", "<100nm"],
}

CANONICAL_DETECTION_METHODS = {
    "FTIR Spectroscopy": ["FTIR", "Fourier-transform infrared", "μ-FTIR",
                          "micro-FTIR", "ATR-FTIR", "FTIR spectroscopy"],
    "Raman Spectroscopy": ["Raman", "μ-Raman", "micro-Raman",
                           "Raman spectroscopy"],
    "Pyrolysis-GC/MS": ["Py-GC/MS", "pyrolysis GC-MS", "Pyr-GC/MS",
                         "pyrolysis gas chromatography mass spectrometry"],
    "SEM-EDS": ["SEM", "scanning electron microscopy", "SEM-EDX",
                "SEM-EDS"],
    "Visual Identification": ["visual sorting", "stereomicroscopy",
                              "visual identification", "microscopy"],
    "Nile Red Staining": ["Nile Red", "fluorescence staining",
                          "Nile red staining"],
    "TGA": ["thermogravimetric analysis", "TGA"],
    "DSC": ["differential scanning calorimetry", "DSC"],
}

CANONICAL_COMPARTMENTS = {
    "Marine water": ["ocean", "sea", "seawater", "marine water",
                     "marine environment"],
    "Freshwater": ["river", "lake", "freshwater", "stream", "pond"],
    "Drinking water": ["tap water", "drinking water", "bottled water",
                       "potable water"],
    "Soil": ["soil", "terrestrial", "agricultural soil", "sediment"],
    "Marine sediment": ["marine sediment", "ocean sediment",
                        "seafloor sediment"],
    "Air": ["air", "atmosphere", "airborne", "indoor air", "outdoor air"],
    "Biota": ["biota", "fish", "shellfish", "organisms", "wildlife"],
    "Human tissue": ["human tissue", "blood", "placenta", "lung tissue",
                     "stool", "human body"],
    "Food": ["food", "seafood", "salt", "honey", "beer", "milk"],
    "Wastewater": ["wastewater", "sewage", "effluent", "WWTP",
                   "wastewater treatment"],
}

CANONICAL_EXPOSURE_PATHWAYS = {
    "Ingestion": ["ingestion", "oral", "dietary", "eating", "drinking"],
    "Inhalation": ["inhalation", "respiratory", "airborne exposure",
                   "breathing"],
    "Dermal contact": ["dermal", "skin contact", "dermal absorption",
                       "cutaneous"],
}


def get_entity_type_descriptions() -> dict[str, str]:
    """Return human-readable descriptions for each entity type."""
    return {
        EntityType.POLYMER: (
            "A plastic polymer type (e.g., Polyethylene, Polypropylene, PET). "
            "Include abbreviation, chemical properties if mentioned."
        ),
        EntityType.SIZE_CLASS: (
            "Particle size classification (e.g., microplastic 1-5mm, "
            "nanoplastic <1μm). Include min/max size if mentioned."
        ),
        EntityType.SOURCE: (
            "Origin or source of microplastic contamination "
            "(e.g., textile fibers, tire wear, packaging, agricultural film). "
            "Include category: industrial, domestic, or agricultural."
        ),
        EntityType.DETECTION_METHOD: (
            "Analytical technique for detecting/characterizing microplastics "
            "(e.g., FTIR, Raman spectroscopy, Py-GC/MS). "
            "Include type: spectroscopic, thermal, visual, or chemical."
        ),
        EntityType.ENVIRONMENTAL_COMPARTMENT: (
            "Environmental location where microplastics are found "
            "(e.g., marine water, freshwater, soil, air, biota, human tissue). "
            "Include subtype if applicable."
        ),
        EntityType.HEALTH_EFFECT: (
            "Health impact of microplastic exposure on humans or organisms "
            "(e.g., inflammation, oxidative stress, endocrine disruption, "
            "cytotoxicity). Include severity and organ system if mentioned."
        ),
        EntityType.EXPOSURE_PATHWAY: (
            "Route of human exposure to microplastics "
            "(ingestion, inhalation, or dermal contact)."
        ),
        EntityType.PUBLICATION: (
            "Research paper or policy document. Include DOI, title, "
            "authors, year, journal if available."
        ),
    }


def get_relation_type_descriptions() -> dict[str, str]:
    """Return human-readable descriptions for each relation type."""
    return {
        RelationType.DETECTED_BY: "Polymer is detected/identified using a specific method",
        RelationType.FOUND_IN: "Polymer is found in an environmental compartment",
        RelationType.ORIGINATES_FROM: "Polymer originates from a contamination source",
        RelationType.HAS_SIZE_CLASS: "Polymer belongs to a size classification",
        RelationType.CAUSES: "Polymer exposure causes a health effect",
        RelationType.EXPOSURE_VIA: "Health effect occurs via an exposure pathway",
        RelationType.LEADS_TO: "One health effect leads to another (causal chain)",
        RelationType.REPORTED_IN: "Entity is reported/described in a publication",
        RelationType.CO_OCCURS_WITH: "Two polymers are found together",
        RelationType.AFFECTS: "Polymer affects an environmental compartment",
        RelationType.REGULATED_BY: "Polymer is regulated by a policy/regulation",
        RelationType.MEASURED_IN: "Detection method is applied in an environmental compartment",
    }


def export_ontology_json(path: Optional[Path] = None) -> dict:
    """Export the full ontology schema as JSON for documentation."""
    ontology = {
        "name": "MicroKG Ontology",
        "version": "1.0",
        "description": "Domain ontology for microplastics knowledge graph",
        "entity_types": {
            et.value: get_entity_type_descriptions()[et]
            for et in EntityType
        },
        "relation_types": {
            rt.value: {
                "description": get_relation_type_descriptions()[rt],
                "source_types": [
                    s.value for s in RELATION_CONSTRAINTS[rt]["source_types"]
                ],
                "target_types": [
                    t.value for t in RELATION_CONSTRAINTS[rt]["target_types"]
                ],
            }
            for rt in RelationType
        },
        "canonical_entities": {
            "polymers": CANONICAL_POLYMERS,
            "size_classes": CANONICAL_SIZE_CLASSES,
            "detection_methods": CANONICAL_DETECTION_METHODS,
            "compartments": CANONICAL_COMPARTMENTS,
            "exposure_pathways": CANONICAL_EXPOSURE_PATHWAYS,
        },
    }

    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ontology, f, indent=2)

    return ontology


# Export on import for easy access
ENTITY_TYPE_DESCRIPTIONS = get_entity_type_descriptions()
RELATION_TYPE_DESCRIPTIONS = get_relation_type_descriptions()
