"""
llm_judge.py
------------
LLM-as-judge evaluation: asks a cheap model to score how well the generated
structured review covers the ground-truth critique points.

Returns per-paper scores (1-5) with rationale and dimension breakdowns.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Prompt ─────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an impartial evaluator assessing the quality of an AI-generated
peer review against ground-truth critique points extracted from real
human reviewers.

Score the generated review on these dimensions (each 1-5):
  coverage   – how many ground-truth points are addressed?
  specificity – are the generated points specific and actionable?
  grounding  – are the generated points supported by evidence?
  overall    – holistic quality score

Output ONLY valid JSON:
{
  "coverage": <int 1-5>,
  "specificity": <int 1-5>,
  "grounding": <int 1-5>,
  "overall": <int 1-5>,
  "rationale": "<1-2 sentence justification>"
}
"""

JUDGE_USER = """\
=== GROUND-TRUTH CRITIQUE POINTS ===
{ground_truth_block}

=== GENERATED REVIEW ===
{generated_block}

Score the generated review against the ground truth.
"""


# ── Single paper judge ─────────────────────────────────────────────────────────

def judge_paper(
    paper_id: str,
    generated: dict,
    ground_truth: dict[str, str],
    cfg: dict,
) -> dict[str, Any]:
    """Score a generated structured review against ground-truth critique points.

    Args:
        paper_id:     Unique paper identifier.
        generated:    Full result dict (with 'structured' and/or 'critique_points').
        ground_truth: Flat dict {point_id: critique_text} from critique_dicts/.
        cfg:          Config dict.

    Returns:
        Dict with paper_id, dimension scores (1-5), and rationale.
    """
    # Build the ground-truth block
    gt_lines = [f"- {v}" for v in ground_truth.values()]
    gt_block = "\n".join(gt_lines) if gt_lines else "(no ground-truth points)"

    # Build the generated block — prefer structured output if available
    structured = generated.get("structured", {})
    if structured:
        gen_parts = []
        if structured.get("summary"):
            gen_parts.append(f"Summary: {structured['summary']}")
        for s in structured.get("strengths", []):
            gen_parts.append(f"Strength: {s.get('point', '')} | Evidence: {s.get('evidence', '')}")
        for w in structured.get("weaknesses", []):
            gen_parts.append(f"Weakness: {w.get('point', '')} | Evidence: {w.get('evidence', '')}")
        for q in structured.get("questions", []):
            gen_parts.append(f"Question: {q.get('question', '')} | Motivation: {q.get('motivation', '')}")
        gen_block = "\n".join(gen_parts)
    else:
        # Fall back to flat critique_points
        cp = generated.get("critique_points", {})
        gen_block = "\n".join(f"- {v}" for v in cp.values()) if cp else "(no generated points)"

    client = _get_client()
    response = client.messages.create(
        model=cfg["models"]["fast"],
        max_tokens=512,
        temperature=0.0,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": JUDGE_USER.format(
                ground_truth_block=gt_block,
                generated_block=gen_block,
            ),
        }],
    )

    raw = response.content[0].text.strip()

    # Parse JSON response
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [WARN] Judge JSON parse failed for {paper_id}")
        scores = {"coverage": 3, "specificity": 3, "grounding": 3, "overall": 3,
                  "rationale": "Parse failure — default scores assigned."}

    scores["paper_id"] = paper_id
    return scores


# ── Batch judge ────────────────────────────────────────────────────────────────

def judge_all(
    results_dir: str,
    critique_dicts_dir: str,
    cfg: dict,
) -> dict[str, Any]:
    """Run the LLM judge on all papers with results and ground truth."""
    results_path = Path(results_dir)
    gt_path = Path(critique_dicts_dir)

    per_paper: dict[str, Any] = {}

    for result_file in sorted(results_path.glob("*.json")):
        if result_file.stem in ("scores", "judge_scores"):
            continue
        paper_id = result_file.stem
        gt_file = gt_path / f"{paper_id}.json"

        if not gt_file.exists():
            print(f"  [SKIP] No ground-truth for {paper_id}")
            continue

        with open(result_file) as f:
            generated = json.load(f)
        with open(gt_file) as f:
            ground_truth = json.load(f)

        try:
            scores = judge_paper(paper_id, generated, ground_truth, cfg)
        except Exception as exc:
            print(f"  [ERROR] Judge failed for {paper_id}: {exc}")
            continue
        per_paper[paper_id] = scores
        print(f"  {paper_id}: overall={scores.get('overall', '?')}  "
              f"coverage={scores.get('coverage', '?')}  "
              f"grounding={scores.get('grounding', '?')}")

    if not per_paper:
        return {"per_paper": {}, "aggregate": {}}

    # Compute aggregates
    dims = ["coverage", "specificity", "grounding", "overall"]
    aggregate = {}
    for dim in dims:
        vals = [v.get(dim, 0) for v in per_paper.values() if isinstance(v.get(dim), (int, float))]
        aggregate[f"mean_{dim}"] = round(sum(vals) / len(vals), 3) if vals else 0.0
    aggregate["n_papers"] = len(per_paper)

    print(f"\n  Judge aggregate: overall={aggregate['mean_overall']:.2f}  "
          f"coverage={aggregate['mean_coverage']:.2f}  "
          f"grounding={aggregate['mean_grounding']:.2f}")

    return {"per_paper": per_paper, "aggregate": aggregate}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cfg = load_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    results_dir = (
        cfg["results"]["baseline_dir"] if mode == "baseline"
        else cfg["results"]["agents_dir"]
    )

    result = judge_all(
        results_dir=results_dir,
        critique_dicts_dir=cfg["data"]["critique_dicts_dir"],
        cfg=cfg,
    )

    out_file = Path(results_dir) / "judge_scores.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJudge scores saved → {out_file}")
