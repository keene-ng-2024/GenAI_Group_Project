"""
vertex_orchestrator.py
----------------------
Vertex AI orchestrator for the multi-agent paper critique system.

This orchestrator uses Vertex AI (Gemini) models for all agent roles
and integrates with the grounding verifier for evidence-based critique.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from src.vertex.state import (
    create_initial_state,
    update_transcript,
    update_token_usage,
    increment_rounds,
    should_early_stop,
    get_latency_seconds,
)
from src.vertex.vertex_client import (
    get_vertex_ai_client,
    generate_content,
    load_config,
)
from src.vertex.personas import AgentRole, BaseAgent, build_agents
from src.vertex.grounding_verifier import verify_all_grounding


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_structured_output(raw: str) -> dict:
    """Parse the Summarizer's structured JSON output with fallbacks."""
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
    import re
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
        
        response = generate_content(
            client=self.client,
            model=self.model,
            messages=self.history,
            config=self.config,
        )
        
        text = response.get("text", "")
        token_usage = response.get("token_usage", {})
        
        self.total_input_tokens += token_usage.get("input_tokens", 0)
        self.total_output_tokens += token_usage.get("output_tokens", 0)
        
        self.history.append({"role": "model", "content": text})
        return text
    
    def reset(self):
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
    
    # Build agents with Vertex AI models
    agents = {
        "reader": VertexAgent(
            name="Reader",
            role=AgentRole.READER,
            system_prompt=(
                "You are a careful academic reader. "
                "When given a paper (or section of a paper), produce a structured "
                "summary with the following clearly labelled sections:\n\n"
                "## Problem & Motivation\n"
                "## Proposed Method\n"
                "## Methods\n"
                "## Results\n"
                "## Claimed Contributions\n\n"
                "Be factual and concise. Include specific numbers from experiments."
            ),
            model=vertex_config.get("reader_model", "gemini-1.5-flash"),
            config=config,
        ),
        "critic": VertexAgent(
            name="Critic",
            role=AgentRole.CRITIC,
            system_prompt=(
                "You are a rigorous peer reviewer for a top-tier ML/AI venue. "
                "Given a paper summary, identify substantive weaknesses in: "
                "novelty, methodology, evaluation, clarity, and reproducibility. "
                "For each point give: (a) the issue, (b) why it matters, "
                "(c) what evidence from the paper supports your concern. "
                "Be specific and actionable."
            ),
            model=vertex_config.get("critic_model", "gemini-1.5-pro"),
            config=config,
        ),
        "auditor": VertexAgent(
            name="Auditor",
            role=AgentRole.AUDITOR,
            system_prompt=(
                "You are a senior programme committee member auditing a peer review. "
                "Your job is to challenge poorly-supported critique points: "
                "ask for concrete evidence, flag over-interpretations, and identify "
                "any points that are actually addressed in the paper. "
                "Also highlight genuine issues the Critic may have missed. "
                "Be constructive but demanding."
            ),
            model=vertex_config.get("auditor_model", "gemini-1.5-flash"),
            config=config,
        ),
        "summarizer": VertexAgent(
            name="Summarizer",
            role=AgentRole.SUMMARIZER,
            system_prompt=(
                "You are a senior editor. Given a debate between a Critic and Auditor "
                "about a paper, synthesise their discussion into a final structured review.\n\n"
                "Output ONLY valid JSON matching this exact schema:\n"
                "{\n"
                '  "summary": "<2-3 sentence paper summary>",\n'
                '  "strengths": [\n'
                '    {"point": "<strength>", "evidence": "<supporting evidence from paper>"}\n'
                "  ],\n"
                '  "weaknesses": [\n'
                '    {"point": "<weakness>", "evidence": "<supporting evidence from paper>"}\n'
                "  ],\n"
                '  "questions": [\n'
                '    {"question": "<question for authors>", "motivation": "<why this matters>"}\n'
                "  ],\n"
                '  "scores": {\n'
                '    "correctness": <int 1-5>,\n'
                '    "novelty": <int 1-5>,\n'
                '    "recommendation": "<accept|borderline|reject>",\n'
                '    "confidence": <int 1-5>\n'
                "  }\n"
                "}\n\n"
                "Rules:\n"
                "- Include 3-8 strengths and 3-8 weaknesses (deduplicated).\n"
                "- Include 2-5 questions for the authors.\n"
                "- Base scores on the reviewer score context provided and the debate.\n"
                "- Output nothing except the JSON object. No markdown fences, no commentary."
            ),
            model=vertex_config.get("summariser_model", "gemini-1.5-pro"),
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
    summary = agents["reader"].chat(f"Please summarise the following paper:\n\n{paper_text}")
    log("Reader", summary)
    
    # Update token usage
    transcript[-1]["input_tokens"] = agents["reader"].total_input_tokens
    transcript[-1]["output_tokens"] = agents["reader"].total_output_tokens
    
    # ── Step 2: Critic generates initial critique ──────────────────────────────
    print(f"\n  [ROUND 0] Critic generating initial points …")
    critique = agents["critic"].chat(
        f"Here is the paper summary:\n\n{summary}\n\nNow list your critique points."
    )
    log("Critic (initial)", critique)
    
    # ── Steps 3–4: Auditor ↔ Critic debate ────────────────────────────────────
    rounds_done = 0
    for round_num in range(1, max_rounds + 1):
        print(f"\n  [ROUND {round_num}] Auditor auditing …")
        audit_feedback = agents["auditor"].chat(
            f"Paper summary:\n{summary}\n\n"
            f"Critic's points:\n{critique}\n\n"
            "Challenge weak points and identify anything important that was missed."
        )
        log(f"Auditor (round {round_num})", audit_feedback)
        
        rounds_done = round_num
        
        # Check early stopping
        if should_early_stop(audit_feedback, early_stop_phrases):
            print(f"  [STOP] Auditor satisfied after round {round_num}.")
            break
        
        print(f"  [ROUND {round_num}] Critic revising …")
        critique = agents["critic"].chat(
            f"The Auditor has challenged some of your points:\n\n{audit_feedback}\n\n"
            "Revise or defend your critique points accordingly."
        )
        log(f"Critic (round {round_num})", critique)
    
    # ── Step 5: Summarizer consolidates ────────────────────────────────────────
    print(f"\n  [SUMMARISE] Consolidating debate …")
    
    # Build context: reviewer scores + full debate transcript
    reviewer_scores_block = _build_reviewer_scores_block([])
    full_debate = "\n\n".join(
        f"=== {entry['role']} ===\n{entry['content']}" for entry in transcript
    )
    if reviewer_scores_block:
        full_debate = reviewer_scores_block + "\n\n" + full_debate
    
    raw_summary = agents["summarizer"].chat(
        f"Here is the full debate transcript:\n\n{full_debate}\n\n"
        "Produce the final structured review JSON."
    )
    log("Summarizer", raw_summary)
    
    # Parse structured output
    structured = _parse_structured_output(raw_summary)
    critique_points = _flatten_to_critique_points(structured)
    
    # Calculate latency and token usage
    latency_seconds = round(time.perf_counter() - start_time, 2)
    total_input = sum(a.total_input_tokens for a in agents.values())
    total_output = sum(a.total_output_tokens for a in agents.values())
    
    # Verify grounding
    grounding_scores = verify_all_grounding(raw_summary, {"full_text": paper_text}, config)
    
    return {
        "paper_id": paper_id,
        "model": vertex_config.get("critic_model", "gemini-1.5-pro"),
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
    """
    Run the Vertex AI critique pipeline for all papers.
    
    Args:
        reviews_path: Path to reviews_parsed.json
        output_dir: Output directory for results
        config: Config dict loaded from config.yaml
    """
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
                paper_text=paper.get("body_text", paper.get("full_text", "")),
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
