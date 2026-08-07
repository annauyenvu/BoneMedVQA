"""Offline / in-memory cache for precomputed visual prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


class PromptCache:
    """Disk cache for masks and prompt metadata keyed by sample + prompt args."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, sample_id: str, payload: dict[str, Any]) -> str:
        blob = json.dumps({"sample_id": sample_id, **payload}, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def path_for(self, sample_id: str, payload: dict[str, Any]) -> Path:
        return self.cache_dir / f"{self._key(sample_id, payload)}.npz"

    def get(self, sample_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        path = self.path_for(sample_id, payload)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def set(self, sample_id: str, payload: dict[str, Any], arrays: dict[str, Any]) -> Path:
        path = self.path_for(sample_id, payload)
        np.savez_compressed(path, **arrays)
        return path
