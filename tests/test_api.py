"""API schema / health tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.schemas import PredictRequest, PredictResponse, VisualPromptSpec
from bonemedvqa.inference.confidence import WARNING


def test_predict_request_schema():
    req = PredictRequest(
        question="Is there a fracture?",
        visual_prompt=VisualPromptSpec(type="box", coordinates=[1, 2, 3, 4]),
    )
    assert req.question_type == "auto"


def test_predict_response_schema():
    resp = PredictResponse(
        answer="Yes",
        answer_type="closed",
        confidence=0.87,
        anatomy="wrist",
        prompt_type="box",
        abstained=False,
        warning=WARNING,
    )
    assert resp.confidence == 0.87


@pytest.mark.skipif(
    True,  # enable when dependencies installed and app imported in CI with model
    reason="Full API boot test runs in integration environment",
)
def test_api_health():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
