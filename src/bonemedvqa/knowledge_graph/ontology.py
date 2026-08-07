"""Maxillofacial clinical ontology for BoneMedVQA Knowledge Graph.

Structured pathways: Finding → Condition → Treatment → Follow-up.
Research/education use only — not a substitute for clinical guidelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NodeType = Literal["anatomy", "finding", "condition", "treatment", "followup", "contraindication"]


@dataclass(frozen=True)
class KGNode:
    node_id: str
    node_type: NodeType
    label_en: str
    label_vi: str
    description: str = ""


@dataclass(frozen=True)
class KGEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    evidence: str = ""


@dataclass
class TreatmentPathway:
    """A ranked treatment recommendation derived from graph traversal."""

    condition_id: str
    condition_label: str
    treatments: list[dict[str, str]]
    followups: list[str]
    contraindications: list[str]
    confidence: float
    reasoning: str
    references: list[str] = field(default_factory=list)
