"""
run_full_pipeline.py
--------------------
Runs the full Vertex AI pipeline for all 100 papers.
Falls back to the sibling GenAI_FinalGroupProject for reviews_parsed.json
if it is not present in the current repo.

Run from GenAI_Group_Project root:
    python scripts/run_full_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.vertex_orchestrator import run_all_papers, load_config

cfg = load_config("config.yaml")

reviews_path = Path(cfg["data"]["reviews_file"])
if not reviews_path.exists():
    alt = Path(__file__).parent.parent.parent / "GenAI_FinalGroupProject" / "data" / "processed" / "reviews_parsed.json"
    if alt.exists():
        reviews_path = alt
    else:
        print(f"ERROR: reviews_parsed.json not found at {reviews_path} or {alt}")
        sys.exit(1)

run_all_papers(
    reviews_path=str(reviews_path),
    output_dir=cfg["results"]["vertexai_dir"],
    config=cfg,
)
