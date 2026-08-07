"""Tests for maxillofacial Knowledge Graph."""

from bonemedvqa.knowledge_graph.graph_store import MaxillofacialKnowledgeGraph
from bonemedvqa.knowledge_graph.treatment_advisor import TreatmentAdvisor


def test_kg_mandible_fracture_pathway():
    graph = MaxillofacialKnowledgeGraph()
    pathway = graph.query_pathway("find_mandible_fracture")
    assert pathway is not None
    assert pathway.condition_id == "cond_mandible_fracture"
    assert len(pathway.treatments) >= 1
    assert pathway.confidence > 0


def test_advisor_fracture_suggestion():
    advisor = TreatmentAdvisor(language="vi")
    out = advisor.suggest(
        anatomy="mandible",
        abnormality="fracture",
        answer="yes",
        question="Is there a mandible fracture?",
        confidence=0.85,
    )
    assert out["enabled"] is True
    assert out["finding_id"] == "find_mandible_fracture"
    assert len(out["treatments"]) >= 1
    assert out["summary_vi"]


def test_advisor_normal_finding():
    advisor = TreatmentAdvisor()
    out = advisor.suggest(anatomy="mandible", answer="no", question="Any abnormality?", confidence=0.9)
    assert out["finding_id"] == "find_normal"
    assert any(t["id"] == "tx_observation" for t in out["treatments"])
