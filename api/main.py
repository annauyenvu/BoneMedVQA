"""BoneMedVQA FastAPI application."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from api.routes import router
from bonemedvqa.inference.confidence import WARNING

app = FastAPI(
    title="BoneMedVQA API",
    description="Research-only musculoskeletal X-ray Visual Question Answering.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(time.time_ns()))
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "warning": WARNING, "request_id": request_id},
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}"
    return response


app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "BoneMedVQA",
        "docs": "/docs",
        "warning": WARNING,
    }
