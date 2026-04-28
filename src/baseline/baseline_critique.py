"""
baseline_critique.py
--------------------
Zero-shot baseline: a single LLM call that critiques a paper.

The model is given the paper title + abstract (and optionally the full text)
and asked to produce a structured peer review in JSON format matching the
project's final output schema.

Uses OpenAI 4o model as the baseline for comparison with agentic approaches.

Output per paper: results/baseline/<paper_id>.json
  {
    "paper_id": str,
    "model": str,
    "latency_seconds": float,
    "token_usage": {"input": int, "output": int},
    "structured": {
      "summary": str,
      "strengths": [{"point": str, "evidence": str}],
      "weaknesses": [{"point": str, "evidence": str}],
      "questions": [{"question": str, "motivation": str}],
      "scores": {
        "correctness": str, "novelty": str,
        "recommendation": str, "confidence": str
      }
    },
    "critique_points": {"point_001": str, ...}   # flattened for evaluation
  }
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert peer reviewer for machine learning and AI conferences.
You will read a paper and produce a structured peer review in a single pass,
performing the roles of Reader, Critic, and Summariser internally.
"""

USER_TEMPLATE = """\
Paper:
Title: {title}

Abstract:
{abstract}

{full_text_section}

First, read and understand the paper. Then generate 12-15 specific critique points.

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
Do NOT invent references, section numbers, or claims not explicitly stated.

Output ONLY valid JSON matching this exact schema:
{{
  "summary": "<1-2 sentence overview of the paper>",
  "strengths": [
    {{"point": "<strength>", "evidence": "<where in the paper>"}}
  ],
  "weaknesses": [
    {{"point": "<weakness or limitation>", "evidence": "<where in the paper>"}}
  ],
  "questions": [
    {{"question": "<question for the authors>", "motivation": "<why it matters>"}}
  ]
}}

Output only the JSON object, no extra text.
"""


def format_user_message(title: str, abstract: str, full_text: str = "",
                        truncate_chars: int = 12000) -> str:
    full_text_section = ""
    if full_text:
        truncated = full_text[:truncate_chars] if truncate_chars else full_text
        full_text_section = f"Full text:\n{truncated}\n"
    return USER_TEMPLATE.format(
        title=title,
        abstract=abstract,
        full_text_section=full_text_section,
    )



# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str, paper_id: str) -> dict | None:
    """Strip markdown fences and parse JSON; return None on failure."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the fenced block
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"  [ERROR] JSON parse failed for {paper_id}: {exc}")
        return None


def _flatten_to_critique_points(structured: dict) -> dict[str, str]:
    """Derive a flat critique_points dict from the structured output for evaluation."""
    points: dict[str, str] = {}
    idx = 1
    for item in structured.get("weaknesses", []):
        text = item.get("point", "")
        evidence = item.get("evidence", "")
        combined = f"{text}. {evidence}".strip(" .") if evidence else text
        if combined:
            points[f"point_{idx:03d}"] = combined
            idx += 1
    return points


# ── Single paper critique ──────────────────────────────────────────────────────

def critique_paper(
    paper_id: str,
    title: str,
    abstract: str,
    full_text: str = "",
    cfg: dict = None,
) -> dict:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = cfg["models"]["baseline"]
    truncate_chars = cfg.get("agent", {}).get("truncate_body_chars", 12000)

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_message(
                title, abstract, full_text, truncate_chars=truncate_chars)}
        ],
    )
    latency = round(time.time() - t0, 3)

    raw = response.choices[0].message.content
    structured = _parse_json(raw, paper_id) or {}

    return {
        "paper_id": paper_id,
        "model": model,
        "latency_seconds": latency,
        "token_usage": {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
        "structured": structured,
        "critique_points": _flatten_to_critique_points(structured),
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _process_paper(paper_id: str, paper: dict, out_dir: Path, cfg: dict) -> None:
    out_file = out_dir / f"{paper_id}.json"
    if out_file.exists():
        print(f"  [SKIP] {paper_id}")
        return
    print(f"  [RUN ] {paper_id} …")
    try:
        result = critique_paper(
            paper_id=paper_id,
            title=paper.get("title", paper_id),
            abstract=paper.get("abstract", ""),
            full_text=paper.get("body_text", paper.get("full_text", "")),
            cfg=cfg,
        )
    except Exception as exc:
        print(f"  [ERROR] {paper_id} failed: {exc}")
        return
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"         → {len(result['critique_points'])} points  "
          f"({result['latency_seconds']}s, "
          f"{result['token_usage']['input']}+{result['token_usage']['output']} tokens)")


def run_baseline(reviews_path: str, output_dir: str, cfg: dict, workers: int = 5) -> None:
    with open(reviews_path) as f:
        all_papers: dict = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_paper, paper_id, paper, out_dir, cfg): paper_id
            for paper_id, paper in all_papers.items()
        }
        for future in as_completed(futures):
            future.result()  # surface any uncaught exceptions


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cfg = load_config()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_baseline(
        reviews_path=cfg["data"]["reviews_file"],
        output_dir=cfg["results"]["baseline_dir"],
        cfg=cfg,
        workers=workers,
    )
