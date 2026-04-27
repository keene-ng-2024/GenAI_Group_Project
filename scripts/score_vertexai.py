"""
score_vertexai.py
-----------------
Score all three Vertex AI loop variants and print a comparison.

Usage:
    python scripts/score_vertexai.py              # scores all three
    python scripts/score_vertexai.py noloop       # score one variant only
    python scripts/score_vertexai.py fixed
    python scripts/score_vertexai.py dynamic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.scorer import score_all, load_config

cfg = load_config("config.yaml")

VARIANTS = {
    "noloop":  cfg["results"]["vertexai_noloop_dir"],
    "fixed":   cfg["results"]["vertexai_fixed_dir"],
    "dynamic": cfg["results"]["vertexai_dir"],
}

requested = sys.argv[1:] if len(sys.argv) > 1 else list(VARIANTS.keys())

for mode in requested:
    if mode not in VARIANTS:
        print(f"Unknown variant '{mode}'. Choose from: {list(VARIANTS.keys())}")
        continue

    results_dir = VARIANTS[mode]
    n_files = len(list(Path(results_dir).glob("paper_*.json"))) if Path(results_dir).exists() else 0

    if n_files == 0:
        print(f"\n[SKIP] {mode} — no results in {results_dir}")
        continue

    print(f"\n{'='*60}")
    print(f"  Scoring vertexai_{mode}  ({n_files} papers)  →  {results_dir}")
    print(f"{'='*60}")

    scores = score_all(
        results_dir=results_dir,
        critique_dicts_dir=cfg["data"]["critique_dicts_dir"],
        cfg=cfg,
    )

    out_file = Path(results_dir) / "scores.json"
    with open(out_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"  Saved → {out_file}")

print("\nDone. Run `python scripts/compare_scores.py` to see full comparison.")
