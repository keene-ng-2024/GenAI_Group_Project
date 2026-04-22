"""
run_full_pipeline.py
--------------------
Runs the Vertex AI pipeline for all 100 papers.

Usage:
    python scripts/run_full_pipeline.py --mode noloop
    python scripts/run_full_pipeline.py --mode fixed
    python scripts/run_full_pipeline.py --mode dynamic   # already done — will skip all

Mode → output dir mapping:
    noloop  → results/vertexai_noloop/
    fixed   → results/vertexai_fixed/
    dynamic → results/vertexai/          (existing results, skipped if present)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vertex.vertex_orchestrator import run_all_papers, load_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["noloop", "fixed", "dynamic"],
    default="dynamic",
    help="Loop type: noloop | fixed | dynamic (default: dynamic)",
)
args = parser.parse_args()

cfg = load_config("config.yaml")

reviews_path = Path(cfg["data"]["reviews_file"])
if not reviews_path.exists():
    alt = (
        Path(__file__).parent.parent.parent
        / "GenAI_FinalGroupProject"
        / "data"
        / "processed"
        / "reviews_parsed.json"
    )
    if alt.exists():
        reviews_path = alt
    else:
        print(f"ERROR: reviews_parsed.json not found at {reviews_path} or {alt}")
        sys.exit(1)

mode_to_dir = {
    "noloop":  cfg["results"]["vertexai_noloop_dir"],
    "fixed":   cfg["results"]["vertexai_fixed_dir"],
    "dynamic": cfg["results"]["vertexai_dir"],
}

output_dir = mode_to_dir[args.mode]
print(f"[INFO] mode={args.mode}  output={output_dir}")

run_all_papers(
    reviews_path=str(reviews_path),
    output_dir=output_dir,
    config=cfg,
    mode=args.mode,
)
