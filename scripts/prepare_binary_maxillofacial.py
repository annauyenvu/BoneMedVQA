#!/usr/bin/env python
"""Prepare binary yes/no maxillofacial subset for direct recognition training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/maxillofacial_annotations.jsonl")
    parser.add_argument("--output", default="data/processed/maxillofacial_binary.jsonl")
    args = parser.parse_args()

    rows = read_jsonl(ROOT / args.input)
    binary = []
    for r in rows:
        if r.get("answer_type") != "yes_no":
            continue
        if r.get("reasoning_level") != "direct_recognition":
            continue
        row = dict(r)
        row["answer"] = "yes" if r.get("finding_id", "find_normal") != "find_normal" else "no"
        binary.append(row)

    write_jsonl(ROOT / args.output, binary)
    print(f"Wrote {len(binary)} binary yes/no samples → {args.output}")


if __name__ == "__main__":
    main()
