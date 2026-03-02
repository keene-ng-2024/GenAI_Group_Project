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
  5. Summariser consolidates the debate into a final JSON critique dict.

Output per paper: results/agents/<paper_id>.json
  {
    "paper_id": str,
    "model": str,
    "rounds": int,
    "transcript": [...],
    "critique_points": {"point_001": "...", ...}
  }
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.agents.agents import build_agents

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json_output(raw: str) -> dict[str, str]:
    """Strip markdown fences and parse JSON from an LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ── Agentic loop ───────────────────────────────────────────────────────────────

def run_agentic_critique(
    paper_id: str,
    paper_text: str,
    cfg: dict,
) -> dict:
    max_rounds = cfg["agent"]["max_rounds"]
    agents = build_agents(cfg)
    transcript: list[dict] = []

    def log(role: str, content: str) -> None:
        transcript.append({"role": role, "content": content})
        print(f"\n    [{role}]\n{content[:300]}{'…' if len(content) > 300 else ''}")

    # ── Step 1: Reader summarises ──────────────────────────────────────────────
    print(f"\n  [ROUND 0] Reading paper …")
    summary = agents["reader"].summarise_paper(paper_text)
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

        print(f"  [ROUND {round_num}] Critic revising …")
        critique = agents["critic"].revise_critique(audit_feedback)
        log(f"Critic (round {round_num})", critique)

        rounds_done = round_num

        # Early stopping: if Auditor explicitly approves
        if any(phrase in audit_feedback.lower() for phrase in
               ["no further concerns", "i am satisfied", "well-supported"]):
            print(f"  [STOP] Auditor satisfied after round {round_num}.")
            break

    # ── Step 5: Summariser consolidates ───────────────────────────────────────
    print(f"\n  [SUMMARISE] Consolidating debate …")
    full_debate = "\n\n".join(
        f"=== {entry['role']} ===\n{entry['content']}" for entry in transcript
    )
    raw_summary = agents["summariser"].summarise(full_debate)
    log("Summariser", raw_summary)

    critique_points = _parse_json_output(raw_summary)

    return {
        "paper_id": paper_id,
        "model": cfg["models"]["strong"],
        "rounds": rounds_done,
        "transcript": transcript,
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

        print(f"\n{'='*60}\n  [PAPER] {paper_id}\n{'='*60}")

        # Combine abstract + review texts as a stand-in for the full paper
        parts = [paper.get("title", ""), paper.get("abstract", "")]
        paper_text = "\n\n".join(p for p in parts if p)

        result = run_agentic_critique(
            paper_id=paper_id,
            paper_text=paper_text,
            cfg=cfg,
        )

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n  [SAVED] {len(result['critique_points'])} points → {out_file}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_all_papers(
        reviews_path=cfg["data"]["reviews_file"],
        output_dir=cfg["results"]["agents_dir"],
        cfg=cfg,
    )
