"""
vertex_orchestrator.py
----------------------
Vertex AI orchestrator for the multi-agent paper critique system.

This orchestrator uses Vertex AI (Gemini) models for all agent roles
and integrates with the grounding verifier for evidence-based critique.
"""

from __future__ import annotations

import json
import re
import time
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.vertex_client import get_vertex_ai_client
from agents.personas import AgentRole
from agents.grounding_verifier import verify_all_grounding, should_stop_debate


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Output parsing ─────────────────────────────────────────────────────────────

_VALID_JSON_ESCAPES = set('"' + '\\' + '/' + 'bfnrtu')


def _sanitize_json_escapes(text: str) -> str:
    """Replace invalid JSON escape sequences (e.g. \\alpha → \\\\alpha) so
    json.loads can parse strings containing LaTeX or other backslash sequences."""
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and i + 1 < len(text):
            next_ch = text[i + 1]
            if next_ch in _VALID_JSON_ESCAPES:
                result.append(ch)
                result.append(next_ch)
                i += 2
            else:
                result.append('\\\\')
                i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def _parse_structured_output(raw: str) -> dict:
    """Parse the Summarizer's structured JSON output with fallbacks."""
    text = raw.strip()

    # Strip markdown fences if present (handles ```json, ```, with/without trailing newline)
    fence_match = re.match(r'^```(?:json)?\s*(.*?)\s*```\s*$', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Retry after sanitizing invalid escape sequences (e.g. LaTeX \alpha)
    try:
        return json.loads(_sanitize_json_escapes(text))
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON object with regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            try:
                return json.loads(_sanitize_json_escapes(match.group()))
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


def _flatten_to_critique_points(structured: dict) -> Dict[str, str]:
    """Convert structured output to flat critique_points dict for scorer compat."""
    points: Dict[str, str] = {}
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


def _build_reviewer_scores_block(reviews: List[dict]) -> str:
    """Average the raw reviewer scores and format them as context for the Summarizer."""
    import numpy as np

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


# ── Vertex AI Agent ────────────────────────────────────────────────────────────

class VertexAgent:
    """Agent wrapper for Vertex AI with state management."""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        system_prompt: str,
        model: str,
        config: dict,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.config = config
        self.client = get_vertex_ai_client(config=config)
        self.history: List[Dict[str, str]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, user_message: str) -> str:
        """Send a message and get a reply."""
        self.history.append({"role": "user", "content": user_message})

        response = self.client.generate_content(
            prompt=user_message,
            system_instruction=self.system_prompt,
            model_name=self.model,
        )

        text = response.get("text", "")
        self.total_input_tokens += response.get("input_tokens", 0)
        self.total_output_tokens += response.get("output_tokens", 0)

        self.history.append({"role": "model", "content": text})
        return text

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_pipeline(
    paper_id: str,
    paper_text: str,
    config: Optional[dict] = None,
) -> dict:
    """
    Run the full Vertex AI multi-agent critique pipeline for one paper.

    Args:
        paper_id: Unique identifier for the paper
        paper_text: Full paper text (may be truncated)
        config: Config dict loaded from config.yaml

    Returns:
        Critique result dict with structured output and metadata
    """
    if config is None:
        config = load_config()

    vertex_config = config.get("vertex_ai", {})
    max_rounds = config["agent"].get("max_rounds", 5)
    early_stop_phrases = config["agent"].get("early_stop_phrases", [])

    agents = {
        "reader": VertexAgent(
            name="Reader",
            role=AgentRole.READER,
            system_prompt=(
                "You are a Reader agent. Read the following paper and produce a structured summary. Cover:\n"
                "Problem & Motivation, Proposed Method, Results, Claimed Contributions."
            ),
            model=vertex_config.get("reader_model", "gemini-2.5-flash-lite"),
            config=config,
        ),
        "critic": VertexAgent(
            name="Critic",
            role=AgentRole.CRITIC,
            system_prompt="You are a Critic agent reviewing an AI/ML research paper.",
            model=vertex_config.get("critic_model", "gemini-2.5-flash"),
            config=config,
        ),
        "auditor": VertexAgent(
            name="Auditor",
            role=AgentRole.AUDITOR,
            system_prompt="You are an Auditor agent. Your job is to make the Critic's review stronger.",
            model=vertex_config.get("auditor_model", "gemini-2.5-flash-lite"),
            config=config,
        ),
        "summarizer": VertexAgent(
            name="Summarizer",
            role=AgentRole.SUMMARIZER,
            system_prompt=(
                "You are a Summariser agent. Consolidate the critique into a final structured review.\n"
                "Output ONLY valid JSON, no other text, no markdown code fences."
            ),
            model=vertex_config.get("summariser_model", "gemini-2.5-flash"),
            config=config,
        ),
    }

    start_time = time.perf_counter()
    transcript: List[Dict[str, Any]] = []

    def log(role: str, content: str) -> None:
        transcript.append({"role": role, "content": content, "timestamp": time.time()})
        print(f"\n    [{role}]\n{content[:300]}{'...' if len(content) > 300 else ''}")

    # ── Step 1: Reader summarises ──────────────────────────────────────────────
    print(f"\n  [ROUND 0] Reading paper …")
    summary = agents["reader"].chat(f"Paper:\n{paper_text}")
    log("Reader", summary)
    transcript[-1]["input_tokens"] = agents["reader"].total_input_tokens
    transcript[-1]["output_tokens"] = agents["reader"].total_output_tokens

    # ── Step 2: Critic generates initial critique ──────────────────────────────
    print(f"\n  [ROUND 0] Critic generating initial points …")
    critic_round1_user = (
        f"Paper summary:\n{summary}\n\n"
        "Generate 12-15 specific critique points.\n\n"
        "For each point you MUST:\n"
        "- Be concrete and specific, not generic\n"
        "- Reference specific sections, tables, or claims from the paper\n"
        "- Focus on ONE issue per point\n\n"
        "Cover ALL of these dimensions:\n"
        "- Novelty: what prior work is missing or inadequately compared?\n"
        "- Methodology: are there hidden assumptions, missing ablations, or design choices not justified?\n"
        "- Evaluation: are baselines fair? are comparisons apples-to-apples? are metrics sufficient?\n"
        "- Reproducibility: what implementation details are missing?\n"
        "- Clarity: what is confusing or poorly explained in the paper?\n"
        "- Limitations: what does the method fail to address or acknowledge?\n"
        "- Generalisability: does it work beyond the tested settings?\n\n"
        "IMPORTANT: Only critique what is actually in the paper.\n"
        "Do NOT invent references, section numbers, or claims not explicitly stated."
    )
    critique = agents["critic"].chat(critic_round1_user)
    log("Critic (initial)", critique)

    # ── Steps 3–4: Auditor ↔ Critic debate ────────────────────────────────────
    rounds_done = 0
    for round_num in range(1, max_rounds + 1):
        print(f"\n  [ROUND {round_num}] Auditor auditing …")
        auditor_user = (
            f"Paper summary:\n{summary}\n\n"
            f"Critic response:\n{critique}\n\n"
            "For each critique point:\n"
            "1. Is it specific enough or too generic? Push for concrete details.\n"
            "2. Is it supported by evidence from the paper?\n"
            "3. What important issues did the Critic completely miss?\n\n"
            "Be aggressive — a weak vague point is worse than no point.\n"
            "Explicitly list 3-5 issues the Critic missed.\n\n"
            "Do NOT suggest ethical implications or bias points unless\n"
            "the paper explicitly makes claims in these areas."
        )
        audit_feedback = agents["auditor"].chat(auditor_user)
        log(f"Auditor (round {round_num})", audit_feedback)
        rounds_done = round_num

        # Use negation-aware early stopping from grounding_verifier
        if should_stop_debate(audit_feedback, early_stop_phrases):
            print(f"  [STOP] Auditor satisfied after round {round_num}.")
            break

        print(f"  [ROUND {round_num}] Critic revising …")
        critic_round2_user = (
            f"Original paper summary:\n{summary}\n\n"
            f"Your original critique:\n{critique}\n\n"
            f"Auditor feedback:\n{audit_feedback}\n\n"
            "Now produce an improved critique that:\n"
            "- Fixes all weak or vague points the Auditor flagged\n"
            "- Adds the missing issues the Auditor identified\n"
            "- Keeps all strong original points\n"
            "- Generates 12-15 total points\n\n"
            "For each point you MUST:\n"
            "- Be concrete and specific, not generic\n"
            "- Reference specific sections, tables, or claims from the paper\n"
            "- Focus on ONE issue per point\n\n"
            "Cover ALL of these dimensions:\n"
            "- Novelty: what prior work is missing or inadequately compared?\n"
            "- Methodology: are there hidden assumptions, missing ablations, or design choices not justified?\n"
            "- Evaluation: are baselines fair? are comparisons apples-to-apples? are metrics sufficient?\n"
            "- Reproducibility: what implementation details are missing?\n"
            "- Clarity: what is confusing or poorly explained in the paper?\n"
            "- Limitations: what does the method fail to address or acknowledge?\n"
            "- Generalisability: does it work beyond the tested settings?"
        )
        critique = agents["critic"].chat(critic_round2_user)
        log(f"Critic (round {round_num})", critique)

    # ── Step 5: Summarizer consolidates ────────────────────────────────────────
    print(f"\n  [SUMMARISE] Consolidating debate …")

    reviewer_scores_block = _build_reviewer_scores_block([])
    full_debate = "\n\n".join(
        f"=== {entry['role']} ===\n{entry['content']}" for entry in transcript
    )
    if reviewer_scores_block:
        full_debate = reviewer_scores_block + "\n\n" + full_debate

    raw_summary = agents["summarizer"].chat(
        f"Critic2 output:\n{critique}\n\n"
        f"Reader summary:\n{summary}\n\n"
        "Output your review as a single JSON object with this exact structure:\n"
        '{"summary": "...", "strengths": [{"point": "...", "evidence": "..."}], '
        '"weaknesses": [{"point": "...", "evidence": "..."}], '
        '"questions": [{"question": "...", "motivation": "..."}], '
        '"scores": {"correctness": 1-5, "novelty": 1-5, "recommendation": "accept|borderline|reject", "confidence": 1-5}}\n'
        "No markdown fences. No other text. Only the JSON object."
    )
    log("Summarizer", raw_summary)

    structured = _parse_structured_output(raw_summary)
    critique_points = _flatten_to_critique_points(structured)

    latency_seconds = round(time.perf_counter() - start_time, 2)
    total_input = sum(a.total_input_tokens for a in agents.values())
    total_output = sum(a.total_output_tokens for a in agents.values())

    # Pass structured weaknesses directly to grounding verifier (not raw JSON text)
    grounding_scores = verify_all_grounding(
        structured.get("weaknesses", []), {"full_text": paper_text}, config
    )

    return {
        "paper_id": paper_id,
        "platform": "vertexai",
        "model": vertex_config.get("critic_model", "gemini-2.5-flash"),
        "rounds": rounds_done,
        "latency_seconds": latency_seconds,
        "token_usage": {"input": total_input, "output": total_output},
        "transcript": transcript,
        "structured": structured,
        "critique_points": critique_points,
        "grounding_verifier_scores": grounding_scores,
        "run_metadata": {
            "latency_ms": latency_seconds * 1000,
            "timestamp": time.time(),
        },
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_all_papers(
    reviews_path: str,
    output_dir: str,
    config: Optional[dict] = None,
) -> None:
    """Run the Vertex AI critique pipeline for all papers."""
    if config is None:
        config = load_config()

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
            result = run_pipeline(
                paper_id=paper_id,
                paper_text=paper.get("full_text", ""),
                config=config,
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
        output_dir=cfg["results"]["vertexai_dir"],
        config=cfg,
    )
