"""Textual prompt normalization for Med-VQA."""

from __future__ import annotations

import re
from typing import Any, Optional


ANATOMY_KEYWORDS = {
    "mandible": ["mandible", "jaw", "lower jaw", "hàm dưới", "xương hàm dưới"],
    "maxilla": ["maxilla", "upper jaw", "hàm trên", "xương hàm trên"],
    "tmj": ["tmj", "temporomandibular", "khớp thái dương hàm", "thái dương hàm"],
    "zygoma": ["zygoma", "zygomatic", "gò má", "xương gò má"],
    "dental": ["dental", "tooth", "teeth", "răng", "impacted", "mọc ngầm"],
    "wrist": ["wrist", "carpal", "distal radius", "ulna"],
    "elbow": ["elbow", "olecranon", "humerus distal"],
    "shoulder": ["shoulder", "clavicle", "scapula", "humerus proximal"],
    "hand": ["hand", "finger", "metacarpal", "phalanx"],
    "forearm": ["forearm", "radius", "ulna"],
    "knee": ["knee", "patella", "tibial plateau"],
    "ankle": ["ankle", "malleolus"],
    "hip": ["hip", "femoral neck", "pelvis"],
    "spine": ["spine", "vertebra", "lumbar", "cervical", "thoracic"],
    "foot": ["foot", "metatarsal", "toe"],
}

ABNORMALITY_KEYWORDS = {
    "fracture": ["fracture", "break", "broken", "gãy"],
    "dislocation": ["dislocation", "displaced", "di lệch", "luxation"],
    "abnormality": ["abnormal", "lesion", "bất thường"],
    "normal": ["normal", "unremarkable", "bình thường"],
}


class TextualPromptBuilder:
    """Convert a user question into a standardized textual prompt."""

    def __init__(
        self,
        include_task: bool = True,
        include_anatomy: bool = True,
        include_abnormality: bool = True,
        include_output_format: bool = True,
        require_abstention_instruction: bool = True,
        language: str = "en",
    ):
        self.include_task = include_task
        self.include_anatomy = include_anatomy
        self.include_abnormality = include_abnormality
        self.include_output_format = include_output_format
        self.require_abstention_instruction = require_abstention_instruction
        self.language = language

    def classify_question_type(self, question: str, hint: str | None = None) -> str:
        if hint and hint.lower() in {"closed", "open"}:
            return hint.lower()
        q = question.lower().strip()
        closed_patterns = [
            r"^is there",
            r"^are there",
            r"^does ",
            r"^do ",
            r"^is this",
            r"^is the",
            r"\byes\b.+\bno\b",
            r"^what body part",
            r"^which ",
            r"có .* không",
        ]
        for pat in closed_patterns:
            if re.search(pat, q):
                return "closed"
        open_cues = ["describe", "explain", "mô tả", "findings", "characterize"]
        if any(c in q for c in open_cues):
            return "open"
        # Default: short questions tend to be closed
        return "closed" if len(q.split()) <= 12 else "open"

    def extract_anatomy(self, question: str, provided: str | None = None) -> str | None:
        if provided:
            return provided.lower()
        q = question.lower()
        for anatomy, keys in ANATOMY_KEYWORDS.items():
            if any(k in q for k in keys):
                return anatomy
        return None

    def extract_abnormality(self, question: str, provided: str | None = None) -> str | None:
        if provided:
            return provided.lower()
        q = question.lower()
        for abn, keys in ABNORMALITY_KEYWORDS.items():
            if any(k in q for k in keys):
                return abn
        return None

    def infer_task(self, abnormality: str | None, question_type: str) -> str:
        if abnormality == "fracture":
            return "fracture_detection"
        if abnormality == "dislocation":
            return "dislocation_detection"
        if abnormality in {"abnormality", "normal"}:
            return "abnormality_detection"
        return "open_description" if question_type == "open" else "closed_vqa"

    def build(
        self,
        question: str,
        question_type: str | None = "auto",
        anatomy: str | None = None,
        abnormality: str | None = None,
        answer_type: str | None = None,
    ) -> str:
        qtype = self.classify_question_type(question, None if question_type == "auto" else question_type)
        anat = self.extract_anatomy(question, anatomy) if self.include_anatomy else anatomy
        abn = self.extract_abnormality(question, abnormality) if self.include_abnormality else abnormality
        task = self.infer_task(abn, qtype)

        lines: list[str] = []
        if self.include_task:
            lines.append(f"Task: {task}")
        if self.include_anatomy and anat:
            lines.append(f"Anatomy: {anat}")
        if self.include_abnormality and abn:
            lines.append(f"Abnormality focus: {abn}")
        lines.append(f"Question type: {qtype}")
        if answer_type:
            lines.append(f"Answer type: {answer_type}")
        lines.append(f"Question: {question.strip()}")

        if self.include_output_format:
            if qtype == "closed":
                lines.append(
                    "Required output: answer label, confidence score, and relevant image region."
                )
            else:
                lines.append(
                    "Return observation, location, confidence and recommendation. "
                    "Describe only findings supported by the image."
                )

        if self.require_abstention_instruction:
            lines.append(
                "If evidence is insufficient, abstain instead of guessing."
            )

        # Research-only reminder embedded in prompt context
        lines.append("Context: research and education only; not a clinical diagnosis.")
        return "\n".join(lines)

    def build_open_template(self, question: str, anatomy: str | None = None) -> str:
        anat = anatomy or self.extract_anatomy(question) or "prompted anatomical region"
        return (
            "Analyze the provided musculoskeletal X-ray.\n"
            f"Focus on the {anat}.\n"
            "Describe only findings supported by the image.\n"
            "Return observation, location, confidence and recommendation.\n"
            f"Question: {question.strip()}"
        )
