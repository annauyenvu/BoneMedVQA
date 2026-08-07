"""API routes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from api.dependencies import get_predictor
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    SegmentRequest,
)
from bonemedvqa import __version__
from bonemedvqa.inference.confidence import WARNING
from bonemedvqa.inference.predictor import Predictor
from bonemedvqa.prompting.visual_prompt import VisualPromptGenerator
from bonemedvqa.utils.device import get_device
from bonemedvqa.utils.io import ensure_dir

router = APIRouter()
MAX_IMAGE_MB = float(__import__("os").getenv("BONEMEDVQA_MAX_IMAGE_MB", "15"))


def _read_upload(file: UploadFile) -> Image.Image:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "application/dicom", None}:
        # allow octet-stream for some clients
        if file.content_type not in {"application/octet-stream", "image/dicom"}:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")
    data = file.file.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Image exceeds {MAX_IMAGE_MB} MB limit")
    from io import BytesIO

    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image") from exc
    return img


@router.get("/health", response_model=HealthResponse)
def health(predictor: Predictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(predictor.device),
        model_loaded=True,
        warning=WARNING,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(predictor: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
    cfg = predictor.cfg
    return ModelInfoResponse(
        name="BoneMedVQA",
        version=__version__,
        profile=str(cfg.get("experiment", {}).get("profile", "unknown")),
        use_visual_prompt=bool(cfg.get("model", {}).get("use_visual_prompt", False)),
        use_textual_prompt=bool(cfg.get("model", {}).get("use_textual_prompt", True)),
        use_latent_prompt=bool(cfg.get("model", {}).get("use_latent_prompt", False)),
        trainable_params=predictor.model.count_trainable_parameters(),
        warning=WARNING,
    )


@router.post("/prompt/segment")
async def segment(
    file: UploadFile = File(...),
    prompt_type: str = Form("automatic"),
    coordinates: Optional[str] = Form(None),
    predictor: Predictor = Depends(get_predictor),
):
    t0 = time.time()
    request_id = str(uuid4())
    img = _read_upload(file)
    arr = np.array(img)
    gen = predictor.visual_prompt_gen
    coords = None
    if coordinates:
        coords = [float(x) for x in coordinates.split(",")]
    if prompt_type == "point" and coords and len(coords) == 2:
        bundle = gen.generate_from_point(arr, coords)
    elif prompt_type == "box" and coords and len(coords) == 4:
        bundle = gen.generate_from_box(arr, coords)
    else:
        bundle = gen.generate_automatic(arr)
    out_dir = ensure_dir(Path(predictor.output_dir) / "masks")
    mask_path = out_dir / f"{request_id}_mask.png"
    Image.fromarray((bundle.mask * 255).astype(np.uint8)).save(mask_path)
    return {
        "request_id": request_id,
        "box": bundle.box,
        "mask_url": str(mask_path),
        "prompt_metadata": bundle.prompt_metadata,
        "elapsed_sec": time.time() - t0,
        "warning": WARNING,
    }


@router.post("/vqa/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    question: str = Form(...),
    question_type: str = Form("auto"),
    prompt_type: Optional[str] = Form(None),
    coordinates: Optional[str] = Form(None),
    return_explanation: bool = Form(True),
    session_id: Optional[str] = Form(None),
    predictor: Predictor = Depends(get_predictor),
):
    t0 = time.time()
    try:
        img = _read_upload(file)
        visual_prompt = None
        if prompt_type:
            coords = [float(x) for x in coordinates.split(",")] if coordinates else None
            visual_prompt = {"type": prompt_type, "coordinates": coords}
        result = predictor.predict(
            image=img,
            question=question,
            question_type=question_type,
            visual_prompt=visual_prompt,
            return_explanation=return_explanation,
            session_id=session_id,
        )
        result["elapsed_sec"] = time.time() - t0
        return PredictResponse(**{k: v for k, v in result.items() if k in PredictResponse.model_fields})
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")


@router.post("/vqa/conversation", response_model=PredictResponse)
async def conversation(
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form(...),
    question_type: str = Form("auto"),
    prompt_type: Optional[str] = Form(None),
    coordinates: Optional[str] = Form(None),
    predictor: Predictor = Depends(get_predictor),
):
    return await predict(
        file=file,
        question=question,
        question_type=question_type,
        prompt_type=prompt_type,
        coordinates=coordinates,
        return_explanation=True,
        session_id=session_id,
        predictor=predictor,
    )


@router.post("/treatment/suggest")
async def treatment_suggest(
    anatomy: Optional[str] = Form(None),
    abnormality: Optional[str] = Form(None),
    answer: Optional[str] = Form(None),
    question: str = Form(""),
    confidence: float = Form(0.5),
    predictor: Predictor = Depends(get_predictor),
):
    if predictor.treatment_advisor is None:
        raise HTTPException(status_code=501, detail="Knowledge Graph is disabled in this config.")
    suggestion = predictor.treatment_advisor.suggest(
        anatomy=anatomy,
        abnormality=abnormality,
        answer=answer,
        question=question,
        confidence=confidence,
    )
    return {**suggestion, "warning": WARNING}


@router.post("/explain")
async def explain(
    file: UploadFile = File(...),
    question: str = Form("Is there evidence of a fracture?"),
    predictor: Predictor = Depends(get_predictor),
):
    img = _read_upload(file)
    result = predictor.predict(image=img, question=question, return_explanation=True)
    return {
        "request_id": result["request_id"],
        "mask_url": result.get("mask_url"),
        "heatmap_url": result.get("heatmap_url"),
        "activated_prompts": result.get("activated_prompts"),
        "confidence": result.get("confidence"),
        "note": "Attention/heatmap overlays are supportive signals, not definitive explanations.",
        "warning": WARNING,
    }
