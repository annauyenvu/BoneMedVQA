"""External vision-language backends for higher-quality demo inference."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

from PIL import Image

from bonemedvqa.inference.confidence import WARNING
from bonemedvqa.utils.logger import get_logger

logger = get_logger("bonemedvqa.vlm_backend")

SYSTEM_PROMPT = (
    "You are an assistant for RESEARCH and EDUCATION on musculoskeletal X-rays. "
    "You are NOT a medical device and must NOT give a clinical diagnosis. "
    "Describe only findings visually supported by the image. "
    "If uncertain or image quality is insufficient, abstain. "
    "Always remind the user to consult a clinician. "
    "Return strict JSON with keys: "
    "answer (short), answer_type (closed|open), raw_label (yes|no|unknown|short label), "
    "confidence (0-1 float), anatomy (string|null), abstained (bool), "
    "observation, location, recommendation, reasoning_brief."
)


def _pil_to_data_url(image: Image.Image, max_side: int = 1280) -> str:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _normalize_yes_no(text: str) -> str | None:
    t = text.strip().lower()
    if re.search(r"\b(yes|true|fracture|broken|abnormal|present)\b", t):
        if re.search(r"\b(no|not|without|absent|normal|unremarkable)\b", t) and "yes" not in t.split()[:3]:
            # ambiguous
            if t.startswith("no") or "no evidence" in t or "no fracture" in t:
                return "no"
        if "no evidence" in t or "no fracture" in t or "without fracture" in t:
            return "no"
        return "yes"
    if re.search(r"\b(no|normal|absent|unremarkable|negative)\b", t):
        return "no"
    return None


def format_structured_answer(
    observation: str,
    location: str,
    confidence: float,
    recommendation: str,
    abstained: bool = False,
) -> str:
    if abstained:
        return (
            "Observation: Insufficient visual evidence for a reliable statement.\n"
            "Location: N/A\n"
            f"Confidence: {confidence:.2f}\n"
            "Recommendation: Re-check image quality/prompt or consult a medical expert."
        )
    return (
        f"Observation: {observation}\n"
        f"Location: {location}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Recommendation: {recommendation}"
    )


class OpenAIVisionBackend:
    """GPT-4o / GPT-4o-mini vision backend."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI Vision backend ready (model=%s)", self.model)
            except Exception as exc:
                logger.warning("OpenAI client init failed: %s", exc)
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def predict(
        self,
        image: Image.Image,
        question: str,
        question_type: str = "auto",
        visual_prompt_note: str | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("OpenAI backend unavailable")
        user_text = (
            f"Question type hint: {question_type}\n"
            f"Question: {question}\n"
            f"Visual prompt context: {visual_prompt_note or 'none'}\n"
            "Respond with JSON only."
        )
        data_url = _pil_to_data_url(image)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "answer": raw[:500],
                "confidence": 0.4,
                "abstained": True,
                "raw_label": "unknown",
                "answer_type": "open",
            }
        return self._pack(data, backend="openai")

    @staticmethod
    def _pack(data: dict[str, Any], backend: str) -> dict[str, Any]:
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        abstained = bool(data.get("abstained", conf < 0.45))
        qtype = str(data.get("answer_type", "closed"))
        raw_label = str(data.get("raw_label", "unknown")).lower()
        answer = str(data.get("answer", "")).strip()
        open_answer = format_structured_answer(
            observation=str(data.get("observation", answer)),
            location=str(data.get("location", data.get("anatomy") or "not specified")),
            confidence=conf,
            recommendation=str(
                data.get(
                    "recommendation",
                    "Research use only — consult a clinician for medical decisions.",
                )
            ),
            abstained=abstained,
        )
        if abstained and not answer:
            from bonemedvqa.inference.confidence import apply_abstention

            answer = apply_abstention("unknown", conf, threshold=1.0)["answer"]
        return {
            "answer": open_answer if qtype == "open" else (answer or open_answer),
            "answer_type": qtype,
            "raw_label": raw_label,
            "confidence": conf,
            "anatomy": data.get("anatomy"),
            "abstained": abstained,
            "open_answer": open_answer,
            "warning": WARNING,
            "backend": backend,
            "reasoning_brief": data.get("reasoning_brief"),
        }


class BlipVQABackend:
    """Offline Hugging Face BLIP VQA backend (downloaded once)."""

    def __init__(self, model_name: str = "Salesforce/blip-vqa-base", device: str | None = None):
        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import BlipForQuestionAnswering, BlipProcessor

        logger.info("Loading BLIP VQA model %s on %s ...", self.model_name, self.device)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("BLIP VQA ready")

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(
        self,
        image: Image.Image,
        question: str,
        question_type: str = "auto",
        visual_prompt_note: str | None = None,
    ) -> dict[str, Any]:
        import torch

        img = image.convert("RGB")
        # Ask a clarifying medical-research phrasing for better BLIP answers
        q = question.strip()
        if question_type == "closed" or re.search(r"\b(is there|are there|does|do )\b", q.lower()):
            q_ask = (
                f"{q} Answer with yes or no first, then a short visual reason. "
                "This is for research education only."
            )
        else:
            q_ask = (
                f"{q} Describe only visible findings on this musculoskeletal X-ray. "
                "Research use only, not a diagnosis."
            )
        if visual_prompt_note:
            q_ask = f"{q_ask} Focus region: {visual_prompt_note}."

        inputs = self.processor(images=img, text=q_ask, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_scores=True,
            )
            text = self.processor.decode(gen.sequences[0], skip_special_tokens=True).strip()
            conf = 0.55
            if gen.scores:
                step_conf = []
                for logits in gen.scores:
                    probs = torch.softmax(logits[0], dim=-1)
                    step_conf.append(float(probs.max().item()))
                if step_conf:
                    conf = float(sum(step_conf) / len(step_conf))
                    conf = 0.35 + 0.55 * conf

        yn = _normalize_yes_no(text)
        raw_label = yn or text.lower()[:32]
        # Closed fracture questions: prefer yes/no answer surface
        closed = question_type == "closed" or bool(
            re.match(r"(?i)^(is|are|does|do|can|was|were)\b", question.strip())
        )
        abstained = conf < 0.42 or (closed and yn is None and len(text) < 2)
        if closed and yn is not None:
            answer = "Yes" if yn == "yes" else "No"
            # bump confidence slightly when answer is crisp
            conf = max(conf, 0.62)
            abstained = False
        else:
            answer = text.capitalize() if text else "Unable to answer"

        anatomy = None
        for a in ("wrist", "elbow", "hand", "ankle", "knee", "shoulder", "patella", "tibia", "fibula", "hip"):
            if a in question.lower() or a in text.lower():
                anatomy = a
                break

        open_answer = format_structured_answer(
            observation=text or answer,
            location=anatomy or "region highlighted by visual prompt / full image",
            confidence=conf,
            recommendation="Research use only — not a medical diagnosis; consult a clinician.",
            abstained=abstained,
        )
        if abstained:
            from bonemedvqa.inference.confidence import apply_abstention

            answer = apply_abstention(answer, conf, threshold=1.0)["answer"]

        return {
            "answer": open_answer if (question_type == "open" and not abstained) else answer,
            "answer_type": "open" if question_type == "open" else "closed",
            "raw_label": raw_label,
            "confidence": float(conf),
            "anatomy": anatomy,
            "abstained": abstained,
            "open_answer": open_answer,
            "warning": WARNING,
            "backend": "blip",
            "reasoning_brief": text,
        }


def build_vlm_backend(prefer: str = "auto", device: str | None = None):
    """Build best available backend: openai > blip > None."""
    prefer = (prefer or "auto").lower()
    if prefer in {"openai", "auto"}:
        oai = OpenAIVisionBackend()
        if oai.available:
            return oai
        if prefer == "openai":
            logger.warning("OPENAI_API_KEY missing; falling back if possible")
    if prefer in {"blip", "auto", "openai"}:
        try:
            return BlipVQABackend(device=device)
        except Exception as exc:
            logger.warning("BLIP backend failed to load: %s", exc)
            return None
    return None
