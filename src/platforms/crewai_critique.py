"""
crewai_critique.py
------------------
CrewAI implementation of the paper-critique pipeline with three loop modes.

Loop modes:
  none    — Reader → Critic → Summariser
  fixed   — Reader → Critic 1 → Auditor → Critic 2 → Summariser (hardcoded, no early exit)
  dynamic — Critic ↔ Auditor loop repeats until Auditor satisfied or max_rounds reached

All prompts are identical to those used across other platforms (see README).

Usage:
  python -m src.platforms.crewai_critique none
  python -m src.platforms.crewai_critique fixed
  python -m src.platforms.crewai_critique dynamic
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import yaml
from crewai import Agent, Crew, Process, Task
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

LOOP_MODES = ("none", "fixed", "dynamic")


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Prompts (from README — identical across all platforms) ─────────────────────

def _reader_prompt(paper_text: str) -> str:
    return f"Paper:\n{paper_text}"


def _critic_r1_prompt(summary: str) -> str:
    return f"""Paper summary:
{summary}

Generate 12-15 specific critique points.

For each point you MUST:
- Be concrete and specific, not generic
- Reference specific sections, tables, or claims from the paper
- Focus on ONE issue per point

Cover ALL of these dimensions:
- Novelty: what prior work is missing or inadequately compared?
- Methodology: are there hidden assumptions, missing ablations, or design choices not justified?
- Evaluation: are baselines fair? are comparisons apples-to-apples? are metrics sufficient?
- Reproducibility: what implementation details are missing?
- Clarity: what is confusing or poorly explained in the paper?
- Limitations: what does the method fail to address or acknowledge?
- Generalisability: does it work beyond the tested settings?

IMPORTANT: Only critique what is actually in the paper.
Do NOT invent references, section numbers, or claims not explicitly stated."""


def _auditor_prompt(summary: str, critique: str) -> str:
    return f"""Paper summary:
{summary}

Critic response:
{critique}

For each critique point:
1. Is it specific enough or too generic? Push for concrete details.
2. Is it supported by evidence from the paper?
3. What important issues did the Critic completely miss?

Be aggressive — a weak vague point is worse than no point.
Explicitly list 3-5 issues the Critic missed.

Do NOT suggest ethical implications or bias points unless
the paper explicitly makes claims in these areas."""


def _critic_r2_prompt(summary: str, critique: str, audit_feedback: str) -> str:
    return f"""Original paper summary:
{summary}

Your original critique:
{critique}

Auditor feedback:
{audit_feedback}

Now produce an improved critique that:
- Fixes all weak or vague points the Auditor flagged
- Adds the missing issues the Auditor identified
- Keeps all strong original points
- Generates 12-15 total points

For each point you MUST:
- Be concrete and specific, not generic
- Reference specific sections, tables, or claims from the paper
- Focus on ONE issue per point

Cover ALL of these dimensions:
- Novelty: what prior work is missing or inadequately compared?
- Methodology: are there hidden assumptions, missing ablations, or design choices not justified?
- Evaluation: are baselines fair? are comparisons apples-to-apples? are metrics sufficient?
- Reproducibility: what implementation details are missing?
- Clarity: what is confusing or poorly explained in the paper?
- Limitations: what does the method fail to address or acknowledge?
- Generalisability: does it work beyond the tested settings?"""


def _summariser_prompt(critic2_output: str, summary: str) -> str:
    return f"""Critic2 output:
{critic2_output}

Reader summary:
{summary}

Output in this exact JSON format:
{{
  "summary": "2-3 sentence paper summary",
  "strengths": [
    {{"point": "strength point", "evidence": "evidence from paper"}},
    {{"point": "strength point", "evidence": "evidence from paper"}}
  ],
  "weaknesses": [
    {{"point": "weakness point", "evidence": "evidence from paper"}},
    {{"point": "weakness point", "evidence": "evidence from paper"}}
  ],
  "questions": [
    {{"question": "open question", "motivation": "why this matters"}},
    {{"question": "open question", "motivation": "why this matters"}}
  ],
  "scores": {{
    "correctness": 3,
    "novelty": 3,
    "recommendation": "borderline",
    "confidence": 3
  }}
}}"""


# ── Agent / task helpers ───────────────────────────────────────────────────────

def _make_llm(model: str, cfg: dict) -> LLM:
    return LLM(model=model, temperature=cfg.get("temperature", 0.2))


def _make_agents(cfg: dict) -> dict[str, Agent]:
    crewai_cfg = cfg.get("crewai", {})
    strong = cfg["models"]["strong"]

    reader_llm     = _make_llm(crewai_cfg.get("reader_model",     strong), cfg)
    critic_llm     = _make_llm(crewai_cfg.get("critic_model",     strong), cfg)
    auditor_llm    = _make_llm(crewai_cfg.get("auditor_model",    strong), cfg)
    summariser_llm = _make_llm(crewai_cfg.get("summariser_model", strong), cfg)

    return {
        "reader": Agent(
            role="Reader",
            goal=(
                "Produce a structured summary of the paper covering "
                "Problem & Motivation, Proposed Method, Results, and Claimed Contributions."
            ),
            backstory=(
                "You are a Reader agent. Read the following paper and produce a structured summary. "
                "Cover: Problem & Motivation, Proposed Method, Results, Claimed Contributions."
            ),
            llm=reader_llm,
            verbose=False,
            allow_delegation=False,
        ),
        "critic": Agent(
            role="Critic",
            goal="Generate 12-15 specific, evidence-grounded critique points of an AI/ML research paper.",
            backstory="You are a Critic agent reviewing an AI/ML research paper.",
            llm=critic_llm,
            verbose=False,
            allow_delegation=False,
        ),
        "auditor": Agent(
            role="Auditor",
            goal="Make the Critic's review stronger by challenging weak points and identifying missed issues.",
            backstory="You are an Auditor agent. Your job is to make the Critic's review stronger.",
            llm=auditor_llm,
            verbose=False,
            allow_delegation=False,
        ),
        # Critic R2 has a distinct system prompt (revision framing) per README
        "critic_r2": Agent(
            role="Critic (Revision)",
            goal="Revise the critique incorporating Auditor feedback to produce a stronger, more specific review.",
            backstory="You are the Critic agent revising your review based on Auditor feedback.",
            llm=critic_llm,
            verbose=False,
            allow_delegation=False,
        ),
        "summariser": Agent(
            role="Summariser",
            goal="Consolidate the critique into a final structured review as valid JSON.",
            backstory=(
                "You are a Summariser agent. Consolidate the critique into a final structured review. "
                "Output ONLY valid JSON, no other text, no markdown code fences."
            ),
            llm=summariser_llm,
            verbose=False,
            allow_delegation=False,
        ),
    }


def _run_task(agent: Agent, description: str, expected_output: str) -> str:
    """Execute a single CrewAI task and return the raw string output."""
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return result.raw or str(result)


# ── Early-stop check (dynamic mode only) ──────────────────────────────────────

def _auditor_satisfied(audit_feedback: str, early_stop_phrases: list[str]) -> bool:
    """Return True if the Auditor signals satisfaction (guards against negated phrases)."""
    feedback_lower = audit_feedback.lower()
    for phrase in early_stop_phrases:
        idx = feedback_lower.find(phrase)
        if idx >= 0:
            prefix = feedback_lower[max(0, idx - 15):idx]
            if any(neg in prefix for neg in ["not ", "no ", "don't ", "isn't ", "hardly "]):
                continue
            return True
    return False


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_structured_output(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    print("  [WARN] Failed to parse structured output, returning empty structure")
    return {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "scores": {"correctness": 3, "novelty": 3, "recommendation": "borderline", "confidence": 1},
    }


def _flatten_to_critique_points(structured: dict) -> dict[str, str]:
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


# ── Loop implementations ───────────────────────────────────────────────────────

def _run_none(
    paper_text: str,
    agents: dict[str, Agent],
    transcript: list[dict],
) -> tuple[str, str]:
    """Reader → Critic → Summariser. Returns (final_critique, raw_json)."""

    def log(role: str, content: str) -> None:
        transcript.append({"role": role, "content": content})
        print(f"\n    [{role}]\n{content[:300]}{'…' if len(content) > 300 else ''}")

    print("\n  [STEP] Reader …")
    summary = _run_task(
        agents["reader"],
        _reader_prompt(paper_text),
        "Structured summary with sections: Problem & Motivation, Proposed Method, Results, Claimed Contributions.",
    )
    log("Reader", summary)

    print("\n  [STEP] Critic …")
    critique = _run_task(
        agents["critic"],
        _critic_r1_prompt(summary),
        "12-15 specific critique points covering all required dimensions.",
    )
    log("Critic", critique)

    print("\n  [STEP] Summariser …")
    raw_json = _run_task(
        agents["summariser"],
        _summariser_prompt(critique, summary),
        "Valid JSON with summary, strengths, weaknesses, questions, and scores.",
    )
    log("Summariser", raw_json)

    return critique, raw_json


def _run_fixed(
    paper_text: str,
    agents: dict[str, Agent],
    transcript: list[dict],
) -> tuple[str, str]:
    """Reader → Critic 1 → Auditor → Critic 2 → Summariser. Returns (final_critique, raw_json)."""

    def log(role: str, content: str) -> None:
        transcript.append({"role": role, "content": content})
        print(f"\n    [{role}]\n{content[:300]}{'…' if len(content) > 300 else ''}")

    print("\n  [STEP] Reader …")
    summary = _run_task(
        agents["reader"],
        _reader_prompt(paper_text),
        "Structured summary with sections: Problem & Motivation, Proposed Method, Results, Claimed Contributions.",
    )
    log("Reader", summary)

    print("\n  [STEP] Critic (round 1) …")
    critique = _run_task(
        agents["critic"],
        _critic_r1_prompt(summary),
        "12-15 specific critique points covering all required dimensions.",
    )
    log("Critic (round 1)", critique)

    print("\n  [STEP] Auditor …")
    audit_feedback = _run_task(
        agents["auditor"],
        _auditor_prompt(summary, critique),
        "Detailed feedback challenging weak critique points and listing 3-5 missed issues.",
    )
    log("Auditor", audit_feedback)

    print("\n  [STEP] Critic (round 2 / revision) …")
    revised_critique = _run_task(
        agents["critic_r2"],
        _critic_r2_prompt(summary, critique, audit_feedback),
        "12-15 revised critique points incorporating Auditor feedback.",
    )
    log("Critic (round 2)", revised_critique)

    print("\n  [STEP] Summariser …")
    raw_json = _run_task(
        agents["summariser"],
        _summariser_prompt(revised_critique, summary),
        "Valid JSON with summary, strengths, weaknesses, questions, and scores.",
    )
    log("Summariser", raw_json)

    return revised_critique, raw_json


def _run_dynamic(
    paper_text: str,
    agents: dict[str, Agent],
    cfg: dict,
    transcript: list[dict],
) -> tuple[str, str, int]:
    """Critic ↔ Auditor loop until satisfied or max_rounds. Returns (final_critique, raw_json, rounds_done)."""
    max_rounds = cfg["agent"]["max_rounds"]
    early_stop_phrases = cfg["agent"].get("early_stop_phrases", [])

    def log(role: str, content: str) -> None:
        transcript.append({"role": role, "content": content})
        print(f"\n    [{role}]\n{content[:300]}{'…' if len(content) > 300 else ''}")

    print("\n  [STEP] Reader …")
    summary = _run_task(
        agents["reader"],
        _reader_prompt(paper_text),
        "Structured summary with sections: Problem & Motivation, Proposed Method, Results, Claimed Contributions.",
    )
    log("Reader", summary)

    print("\n  [STEP] Critic (initial) …")
    critique = _run_task(
        agents["critic"],
        _critic_r1_prompt(summary),
        "12-15 specific critique points covering all required dimensions.",
    )
    log("Critic (initial)", critique)

    rounds_done = 0
    for round_num in range(1, max_rounds + 1):
        print(f"\n  [ROUND {round_num}] Auditor …")
        audit_feedback = _run_task(
            agents["auditor"],
            _auditor_prompt(summary, critique),
            "Detailed feedback challenging weak critique points and listing 3-5 missed issues.",
        )
        log(f"Auditor (round {round_num})", audit_feedback)
        rounds_done = round_num

        if _auditor_satisfied(audit_feedback, early_stop_phrases):
            print(f"  [STOP] Auditor satisfied after round {round_num}.")
            break

        print(f"  [ROUND {round_num}] Critic revision …")
        critique = _run_task(
            agents["critic_r2"],
            _critic_r2_prompt(summary, critique, audit_feedback),
            "12-15 revised critique points incorporating Auditor feedback.",
        )
        log(f"Critic (round {round_num})", critique)

    print("\n  [STEP] Summariser …")
    raw_json = _run_task(
        agents["summariser"],
        _summariser_prompt(critique, summary),
        "Valid JSON with summary, strengths, weaknesses, questions, and scores.",
    )
    log("Summariser", raw_json)

    return critique, raw_json, rounds_done


# ── Single paper entry point ───────────────────────────────────────────────────

def critique_paper(
    paper_id: str,
    paper: dict,
    loop_mode: str,
    cfg: dict,
) -> dict:
    """Run the CrewAI critique pipeline for one paper.

    Args:
        paper_id:  Unique identifier.
        paper:     Paper dict with 'title', 'body_text'/'full_text', 'abstract'.
        loop_mode: "none", "fixed", or "dynamic".
        cfg:       Config dict from config.yaml.
    """
    truncate_chars = cfg["agent"].get("truncate_body_chars", 0)
    title = paper.get("title", paper_id)
    full_text = paper.get("body_text", paper.get("full_text", ""))
    paper_text = (
        (full_text[:truncate_chars] if truncate_chars else full_text)
        if full_text
        else paper.get("abstract", title)
    )

    agents = _make_agents(cfg)
    transcript: list[dict] = []
    start_time = time.perf_counter()

    if loop_mode == "none":
        final_critique, raw_json = _run_none(paper_text, agents, transcript)
        rounds = 0
    elif loop_mode == "fixed":
        final_critique, raw_json = _run_fixed(paper_text, agents, transcript)
        rounds = 1
    elif loop_mode == "dynamic":
        final_critique, raw_json, rounds = _run_dynamic(paper_text, agents, cfg, transcript)
    else:
        raise ValueError(f"Unknown loop_mode '{loop_mode}'. Use: {', '.join(LOOP_MODES)}")

    latency_seconds = round(time.perf_counter() - start_time, 2)
    structured = _parse_structured_output(raw_json)
    critique_points = _flatten_to_critique_points(structured)

    return {
        "paper_id": paper_id,
        "title": title,
        "platform": f"crewai_{loop_mode}",
        "model": cfg.get("crewai", {}).get("critic_model", cfg["models"]["strong"]),
        "loop_mode": loop_mode,
        "rounds": rounds,
        "latency_seconds": latency_seconds,
        "transcript": transcript,
        "structured": structured,
        "critique_points": critique_points,
    }


# ── Batch pipeline ─────────────────────────────────────────────────────────────

def run_all_papers(loop_mode: str, cfg: dict) -> None:
    """Run all papers in the eval JSONL through the specified CrewAI loop mode."""
    if loop_mode not in LOOP_MODES:
        raise ValueError(f"Unknown loop_mode '{loop_mode}'. Use: {', '.join(LOOP_MODES)}")

    output_dir = cfg["results"].get(f"crewai_{loop_mode}_dir", f"results/crewai_{loop_mode}")
    jsonl_path = cfg["data"].get("jsonl_file", "data/ReviewCritique.jsonl")

    papers: dict = {}
    with open(jsonl_path) as f:
        for i, line in enumerate(f, start=1):
            papers[f"paper_{i:04d}"] = json.loads(line)
    print(f"  [INFO] {len(papers)} papers, loop_mode={loop_mode}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for paper_id, paper in papers.items():
        out_file = out_dir / f"{paper_id}.json"
        if out_file.exists():
            print(f"  [SKIP] {paper_id}")
            continue

        print(f"\n{'='*60}\n  [PAPER] {paper_id} — {paper.get('title', '')[:50]}\n{'='*60}")

        try:
            result = critique_paper(
                paper_id=paper_id,
                paper=paper,
                loop_mode=loop_mode,
                cfg=cfg,
            )
        except Exception as exc:
            print(f"  [ERROR] {paper_id}: {exc}")
            continue

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        n_points = len(result["critique_points"])
        print(f"  [SAVED] {n_points} points → {out_file}  ({result['latency_seconds']}s)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in LOOP_MODES:
        print(f"Usage: python -m src.platforms.crewai_critique [{' | '.join(LOOP_MODES)}]")
        sys.exit(1)

    mode = sys.argv[1]
    cfg = load_config()
    run_all_papers(loop_mode=mode, cfg=cfg)
