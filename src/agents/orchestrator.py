"""
orchestrator.py
---------------
Main agentic loop for paper critique.

Workflow
  1. Reader  summarises the paper.
  2. Critic  proposes initial critique points.
  3. Auditor challenges the Critic.                 ─┐
  4. Critic  revises in light of Auditor feedback.   │  repeated up to max_rounds
     (repeat 3–4 for max_rounds iterations)         ─┘
  5. Summariser consolidates the debate into a final structured review JSON.

Output per paper: results/agents/<paper_id>.json
  {
    "paper_id": str,
    "model": str,
    "rounds": int,
    "latency_seconds": float,
    "token_usage": {"input": int, "output": int},
    "transcript": [...],
    "structured": {
      "summary": str,
      "strengths": [{"point": str, "evidence": str}],
      "weaknesses": [{"point": str, "evidence": str}],
      "questions": [{"question": str, "motivation": str}],
      "scores": {"correctness": int, "novelty": int,
                 "recommendation": str, "confidence": int}
    },
    "critique_points": {"point_001": str, ...}
  }
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from src.agents.agents import build_agents

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_structured_output(raw: str) -> dict:
    """Parse the Summariser's structured JSON output with fallbacks."""
    text = raw.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON object with regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: return an empty valid structure
    print("  [WARN] Failed to parse structured output, returning empty structure")
    return {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "scores": {
            "correctness": 3,
            "novelty": 3,
            "recommendation": "borderline",
            "confidence": 1,
        },
    }


def _flatten_to_critique_points(structured: dict) -> dict[str, str]:
    """Convert structured output to flat critique_points dict for scorer compat."""
    points = {}
    idx = 1
    for item in structured.get("weaknesses", []):
        if isinstance(item, str):
            full = item
        elif isinstance(item, dict):
            point_text = item.get("point", "")
            evidence = item.get("evidence", "")
            full = f"{point_text}. {evidence}".strip(" .") if evidence else point_text
        else:
            continue
        points[f"point_{idx:03d}"] = full
        idx += 1
    return points


def _build_reviewer_scores_block(reviews: list[dict]) -> str:
    """Average the raw reviewer scores and format them as context for the Summariser."""
    score_keys = ["Correctness", "Technical Novelty And Significance",
                  "Empirical Novelty And Significance", "Recommendation", "Confidence"]
    averages = {}
    for key in score_keys:
        vals = []
        for r in reviews:
            raw = r.get("scores", {}).get(key, "")
            try:
                vals.append(float(raw))
            except (ValueError, TypeError):
                continue
        if vals:
            averages[key] = round(np.mean(vals), 1)

    if not averages:
        return ""

    lines = ["Reviewer score averages (for context):"]
    for k, v in averages.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ── Agentic loop ───────────────────────────────────────────────────────────────

def run_agentic_critique(
    paper_id: str,
    paper: dict,
    cfg: dict,
) -> dict:
    """Run the full multi-agent critique loop for one paper.

    Args:
        paper_id: Unique identifier for the paper.
        paper:    Full paper dict from reviews_parsed.json (title, full_text, reviews, etc.)
        cfg:      Config dict loaded from config.yaml.
    """
    start_time = time.perf_counter()

    max_rounds = cfg["agent"]["max_rounds"]
    truncate_chars = cfg["agent"].get("truncate_body_chars", 12000)
    early_stop_phrases = cfg["agent"].get("early_stop_phrases", [])
    agents = build_agents(cfg)
    transcript: list[dict] = []

    def log(role: str, content: str) -> None:
        transcript.append({"role": role, "content": content})
        print(f"\n    [{role}]\n{content[:300]}{'…' if len(content) > 300 else ''}")

    # Prepare paper text (truncated for context window management)
    title = paper.get("title", paper_id)
    full_text = paper.get("full_text", "")
    paper_text = full_text[:truncate_chars] if full_text else paper.get("abstract", "")
    if not paper_text:
        paper_text = title

    # ── Step 1: Reader summarises ──────────────────────────────────────────────
    print(f"\n  [ROUND 0] Reading paper …")
    summary = agents["reader"].summarise_paper(
        f"Title: {title}\n\n{paper_text}"
    )
    log("Reader", summary)

    # ── Step 2: Critic generates initial critique ──────────────────────────────
    print(f"\n  [ROUND 0] Critic generating initial points …")
    critique = agents["critic"].generate_critique(summary)
    log("Critic (initial)", critique)

    # ── Steps 3–4: Auditor ↔ Critic debate ────────────────────────────────────
    rounds_done = 0
    for round_num in range(1, max_rounds + 1):
        print(f"\n  [ROUND {round_num}] Auditor auditing …")
        audit_feedback = agents["auditor"].audit(critique, summary)
        log(f"Auditor (round {round_num})", audit_feedback)

        rounds_done = round_num

        # Early stopping if Auditor is satisfied — skip unnecessary Critic revision
        # Guard against negated phrases like "not well-supported" or "disapprove"
        feedback_lower = audit_feedback.lower()
        should_stop = False
        for phrase in early_stop_phrases:
            idx = feedback_lower.find(phrase)
            if idx >= 0:
                prefix = feedback_lower[max(0, idx - 15):idx]
                if any(neg in prefix for neg in ["not ", "no ", "don't ", "isn't ", "hardly "]):
                    continue
                should_stop = True
                break
        if should_stop:
            print(f"  [STOP] Auditor satisfied after round {round_num}.")
            break

        print(f"  [ROUND {round_num}] Critic revising …")
        critique = agents["critic"].revise_critique(audit_feedback)
        log(f"Critic (round {round_num})", critique)

    # ── Step 5: Summariser consolidates ────────────────────────────────────────
    print(f"\n  [SUMMARISE] Consolidating debate …")

    # Build context: reviewer scores + full debate transcript
    reviewer_scores_block = _build_reviewer_scores_block(paper.get("reviews", []))
    full_debate = "\n\n".join(
        f"=== {entry['role']} ===\n{entry['content']}" for entry in transcript
    )
    if reviewer_scores_block:
        full_debate = reviewer_scores_block + "\n\n" + full_debate

    raw_summary = agents["summariser"].summarise(full_debate)
    log("Summariser", raw_summary)

    # Parse structured output
    structured = _parse_structured_output(raw_summary)
    critique_points = _flatten_to_critique_points(structured)

    # Collect total token usage across all agents.
    # NOTE: This undercounts by ~20-30% because tool sub-calls (e.g. extract_claims,
    # flag_missing_baselines) make their own haiku API calls that are not tracked here.
    total_input = sum(a.total_input_tokens for a in agents.values())
    total_output = sum(a.total_output_tokens for a in agents.values())

    latency_seconds = round(time.perf_counter() - start_time, 2)

    return {
        "paper_id": paper_id,
        "title": title,
        "model": cfg["models"]["strong"],
        "rounds": rounds_done,
        "latency_seconds": latency_seconds,
        "token_usage": {"input": total_input, "output": total_output},
        "transcript": transcript,
        "structured": structured,
        "critique_points": critique_points,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_all_papers(reviews_path: str, output_dir: str, cfg: dict) -> None:
    with open(reviews_path) as f:
        all_papers: dict = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for paper_id, paper in all_papers.items():
        out_file = out_dir / f"{paper_id}.json"
        if out_file.exists():
            print(f"  [SKIP] {paper_id}")
            continue

        print(f"\n{'='*60}\n  [PAPER] {paper_id} — {paper.get('title', '')[:50]}\n{'='*60}")

        try:
            result = run_agentic_critique(
                paper_id=paper_id,
                paper=paper,
                cfg=cfg,
            )
        except Exception as exc:
            print(f"\n  [ERROR] {paper_id} failed: {exc}")
            continue

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        n_points = len(result["critique_points"])
        print(f"\n  [SAVED] {n_points} weakness points → {out_file}")
        print(f"  [COST]  {result['token_usage']['input']:,} in / "
              f"{result['token_usage']['output']:,} out tokens  "
              f"({result['latency_seconds']}s)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_all_papers(
        reviews_path=cfg["data"]["reviews_file"],
        output_dir=cfg["results"]["agents_dir"],
        cfg=cfg,
    )
