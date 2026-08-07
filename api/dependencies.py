"""FastAPI dependency providers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from bonemedvqa.inference.predictor import Predictor
from bonemedvqa.utils.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load model once per process."""
    cfg_path = Path(os.getenv("BONEMEDVQA_CONFIG", str(ROOT / "configs" / "baseline.yaml")))
    ckpt = os.getenv("BONEMEDVQA_CHECKPOINT", str(ROOT / "outputs" / "checkpoints" / "best.pt"))
    backend = os.getenv("BONEMEDVQA_BACKEND", "auto")
    cfg = load_yaml(cfg_path)
    return Predictor(
        cfg=cfg,
        checkpoint=ckpt if Path(ckpt).exists() else None,
        backend=backend,
    )
