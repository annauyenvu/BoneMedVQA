"""FastAPI schemas for BoneMedVQA."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class VisualPromptSpec(BaseModel):
    type: Literal["point", "box", "mask", "automatic"] = "box"
    coordinates: Optional[list[float]] = None

    @field_validator("coordinates")
    @classmethod
    def check_coords(cls, v, info):
        if v is None:
            return v
        if len(v) not in {2, 4}:
            raise ValueError("coordinates must be [x,y] for point or [x1,y1,x2,y2] for box")
        return v


class PredictRequest(BaseModel):
    image_id: Optional[str] = None
    question: str = Field(..., min_length=1, max_length=1000)
    question_type: Literal["auto", "closed", "open"] = "auto"
    visual_prompt: Optional[VisualPromptSpec] = None
    return_explanation: bool = True
    session_id: Optional[str] = None


class PredictResponse(BaseModel):
    answer: str
    answer_type: str
    confidence: float
    anatomy: Optional[str] = None
    prompt_type: str = "none"
    mask_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    abstained: bool = False
    warning: str
    request_id: Optional[str] = None
    raw_label: Optional[str] = None
    open_answer: Optional[str] = None
    activated_prompts: Optional[dict[str, bool]] = None
    session_id: Optional[str] = None
    history: Optional[list[dict[str, Any]]] = None
    treatment_suggestions: Optional[list[dict[str, Any]]] = None
    treatment_summary: Optional[str] = None
    knowledge_graph: Optional[dict[str, Any]] = None


class SegmentRequest(BaseModel):
    prompt_type: Literal["point", "box", "automatic"] = "automatic"
    coordinates: Optional[list[float]] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    warning: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    profile: str
    use_visual_prompt: bool
    use_textual_prompt: bool
    use_latent_prompt: bool
    trainable_params: Optional[int] = None
    warning: str
