"""
rerun_zero_point_papers.py
--------------------------
Re-runs the full Vertex AI pipeline for papers that still have 0 critique_points
after the patch script (i.e. their Summarizer transcript was genuinely unparseable).

Run from GenAI_Group_Project root:
    python scripts/rerun_zero_point_papers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.vertex_orchestrator import run_pipeline, load_config


def main() -> None:
    cfg = load_config("config.yaml")
    results_dir = Path(cfg["results"]["vertexai_dir"])

    # reviews_parsed.json may live in the old project if not yet copied over
    reviews_path = Path(cfg["data"]["reviews_file"])
    if not reviews_path.exists():
        alt = Path(__file__).parent.parent.parent / "GenAI_FinalGroupProject" / "data" / "processed" / "reviews_parsed.json"
        if alt.exists():
            reviews_path = alt
        else:
            print(f"ERROR: reviews_parsed.json not found at {reviews_path} or {alt}")
            sys.exit(1)

    # Load all papers
    with open(reviews_path, encoding="utf-8") as f:
        all_papers: dict = json.load(f)

    # Find zero-point papers
    zero_point = []
    for fpath in sorted(results_dir.glob("paper_*.json")):
        data = json.loads(fpath.read_text(encoding="utf-8"))
        if data.get("critique_points") == {}:
            zero_point.append(fpath.stem)

    if not zero_point:
        print("No zero-point papers found — nothing to re-run.")
        return

    print(f"Re-running {len(zero_point)} zero-point papers: {zero_point}\n")

    for paper_id in zero_point:
        paper = all_papers.get(paper_id)
        if not paper:
            print(f"  [SKIP] {paper_id} not found in reviews_parsed.json")
            continue

        print(f"\n{'='*60}\n  [RERUN] {paper_id} — {paper.get('title', '')[:50]}\n{'='*60}")

        try:
            result = run_pipeline(
                paper_id=paper_id,
                paper_text=paper.get("full_text", ""),
                config=cfg,
            )
        except Exception as exc:
            print(f"\n  [ERROR] {paper_id} failed: {exc}")
            continue

        out_file = results_dir / f"{paper_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        n_points = len(result["critique_points"])
        print(f"\n  [SAVED] {n_points} weakness points → {out_file}")
        print(f"  [COST]  {result['token_usage']['input']:,} in / "
              f"{result['token_usage']['output']:,} out tokens  "
              f"({result['latency_seconds']}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
