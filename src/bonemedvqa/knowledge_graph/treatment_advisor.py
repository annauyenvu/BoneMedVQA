"""Treatment advisor powered by maxillofacial Knowledge Graph."""

from __future__ import annotations

import re
from typing import Any

from bonemedvqa.knowledge_graph.graph_store import MaxillofacialKnowledgeGraph


FINDING_MAP = {
    "mandible": "find_mandible_fracture",
    "maxilla": "find_maxilla_fracture",
    "tmj": "find_tmj_dislocation",
    "temporomandibular": "find_tmj_dislocation",
    "zygoma": "find_zygoma_fracture",
    "zygomatic": "find_zygoma_fracture",
    "dental": "find_impacted_tooth",
    "tooth": "find_impacted_tooth",
    "impacted": "find_impacted_tooth",
    "periapical": "find_periapical_lesion",
    "abscess": "find_periapical_lesion",
}

ABNORMALITY_MAP = {
    "fracture": {
        "mandible": "find_mandible_fracture",
        "maxilla": "find_maxilla_fracture",
        "zygoma": "find_zygoma_fracture",
        "default": "find_mandible_fracture",
    },
    "dislocation": {"default": "find_tmj_dislocation"},
    "impacted": {"default": "find_impacted_tooth"},
    "abscess": {"default": "find_periapical_lesion"},
    "normal": {"default": "find_normal"},
}


class TreatmentAdvisor:
    """Map VQA outputs to KG-backed treatment suggestions."""

    def __init__(self, graph: MaxillofacialKnowledgeGraph | None = None, language: str = "vi"):
        self.graph = graph or MaxillofacialKnowledgeGraph()
        self.language = language

    def _resolve_finding_id(
        self,
        anatomy: str | None,
        abnormality: str | None,
        answer: str | None,
        question: str = "",
    ) -> str:
        text = " ".join(filter(None, [anatomy, abnormality, answer, question])).lower()

        if answer and answer.lower() in {"no", "normal", "none"} and "fracture" not in text:
            if re.search(r"\b(no|normal|negative|absent|không)\b", answer.lower()):
                return "find_normal"

        for key, fid in FINDING_MAP.items():
            if key in text:
                return fid

        abn = (abnormality or "").lower()
        anat = (anatomy or "").lower()
        if abn in ABNORMALITY_MAP:
            mapping = ABNORMALITY_MAP[abn]
            return mapping.get(anat, mapping.get("default", "find_normal"))

        if re.search(r"fracture|gãy|broken", text):
            if "maxilla" in text or "hàm trên" in text:
                return "find_maxilla_fracture"
            if "zygoma" in text or "gò má" in text:
                return "find_zygoma_fracture"
            return "find_mandible_fracture"
        if re.search(r"disloc|trật|tmj", text):
            return "find_tmj_dislocation"
        if re.search(r"impacted|mọc ngầm|khôn", text):
            return "find_impacted_tooth"
        if re.search(r"periapical|abscess|áp xe", text):
            return "find_periapical_lesion"

        return "find_normal"

    def _infer_severity(self, answer: str | None, confidence: float) -> str:
        if answer and re.search(r"comminuted|displaced|unstable|nghiêm trọng|dịch chuyển", answer.lower()):
            return "severe"
        if confidence >= 0.75:
            return "moderate"
        return "mild"

    def suggest(
        self,
        *,
        anatomy: str | None = None,
        abnormality: str | None = None,
        answer: str | None = None,
        question: str = "",
        confidence: float = 0.5,
    ) -> dict[str, Any]:
        finding_id = self._resolve_finding_id(anatomy, abnormality, answer, question)
        severity = self._infer_severity(answer, confidence)
        pathway = self.graph.query_pathway(finding_id, severity=severity)
        if pathway is None:
            return {
                "enabled": True,
                "finding_id": finding_id,
                "condition": None,
                "treatments": [],
                "followups": [],
                "contraindications": [],
                "confidence": 0.0,
                "reasoning": "No KG pathway matched.",
                "summary_vi": "Không tìm thấy gợi ý điều trị phù hợp trong KG.",
                "summary_en": "No matching treatment pathway in KG.",
                "disclaimer": "Chỉ tham khảo nghiên cứu — không thay thế chỉ định lâm sàng.",
            }

        lang = self.language.lower()
        if lang.startswith("vi"):
            tx_lines = [f"• {t['label_vi']} — {t['rationale']}" for t in pathway.treatments]
            summary = f"Điều kiện: {pathway.condition_label}\n" + "\n".join(tx_lines)
            if pathway.followups:
                summary += "\nTheo dõi: " + "; ".join(pathway.followups)
        else:
            tx_lines = [f"• {t['label_en']} — {t['rationale']}" for t in pathway.treatments]
            summary = f"Condition: {pathway.condition_label}\n" + "\n".join(tx_lines)
            if pathway.followups:
                summary += "\nFollow-up: " + "; ".join(pathway.followups)

        return {
            "enabled": True,
            "finding_id": finding_id,
            "condition": pathway.condition_label,
            "treatments": pathway.treatments,
            "followups": pathway.followups,
            "contraindications": pathway.contraindications,
            "confidence": pathway.confidence,
            "reasoning": pathway.reasoning,
            "references": pathway.references,
            "summary_vi": summary if lang.startswith("vi") else None,
            "summary_en": summary if not lang.startswith("vi") else None,
            "disclaimer": "Chỉ tham khảo nghiên cứu — không thay thế chỉ định lâm sàng.",
        }
