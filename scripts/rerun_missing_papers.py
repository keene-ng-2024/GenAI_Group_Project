"""
rerun_missing_papers.py
-----------------------
Re-runs the Vertex AI pipeline for papers that have no result file at all.
Also re-scores all papers and updates scores.json.

Run from GenAI_Group_Project root:
    python scripts/rerun_missing_papers.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.vertex_orchestrator import run_pipeline, load_config


def main() -> None:
    cfg = load_config("config.yaml")
    results_dir = Path(cfg["results"]["vertexai_dir"])

    # Locate reviews_parsed.json
    reviews_path = Path(cfg["data"]["reviews_file"])
    if not reviews_path.exists():
        alt = Path(__file__).parent.parent.parent / "GenAI_FinalGroupProject" / "data" / "processed" / "reviews_parsed.json"
        if alt.exists():
            reviews_path = alt
        else:
            print(f"ERROR: reviews_parsed.json not found at {reviews_path} or {alt}")
            sys.exit(1)

    with open(reviews_path, encoding="utf-8") as f:
        all_papers: dict = json.load(f)

    # Find missing result files
    expected = {f"paper_{i:04d}" for i in range(1, 101)}
    present = {f.stem for f in results_dir.glob("paper_*.json")}
    missing = sorted(expected - present)

    if not missing:
        print("No missing result files — all 100 papers have output files.")
    else:
        print(f"Found {len(missing)} missing result files: {missing}\n")
        for paper_id in missing:
            paper = all_papers.get(paper_id)
            if not paper:
                print(f"  [SKIP] {paper_id} not found in reviews_parsed.json")
                continue

            title = paper.get("title", "")[:60]
            print(f"\n{'='*60}\n  [RUN] {paper_id} — {title}\n{'='*60}")

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
            print(f"\n  [SAVED] {n_points} critique points → {out_file}")
            print(f"  [COST]  {result['token_usage']['input']:,} in / "
                  f"{result['token_usage']['output']:,} out tokens  "
                  f"({result['latency_seconds']}s)")

    # Re-score all papers and update scores.json
    print("\n\nRe-scoring all papers and updating scores.json ...")
    _rescore_all(cfg, results_dir, reviews_path, all_papers)
    print("Done.")


def _rescore_all(cfg, results_dir, reviews_path, all_papers):
    """Re-run evaluation scoring for all result files and write scores.json."""
    try:
        from evaluation.scorer import score_all
        scores = score_all(
            results_dir=str(results_dir),
            critique_dicts_dir=cfg["data"]["critique_dicts_dir"],
            cfg=cfg,
        )
        scores_file = results_dir / "scores.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        agg = scores.get("aggregate", {})
        print(f"  Scores updated: {scores_file}")
        print(f"  n_papers={agg.get('n_papers')}, mean_f1={agg.get('mean_f1')}, "
              f"mean_precision={agg.get('mean_precision')}, mean_recall={agg.get('mean_recall')}")
    except Exception as exc:
        print(f"  [WARN] Could not auto-rescore: {exc}")
        print("  Run the evaluation script manually to update scores.json.")


if __name__ == "__main__":
    main()
