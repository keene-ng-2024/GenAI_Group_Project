"""
baseline_critique.py
--------------------
Zero-shot baseline: a single LLM call that critiques a paper.

The model is given the paper title + abstract (and optionally the full text)
and asked to produce a list of critique points in JSON format, mirroring the
structure of the ground-truth critique dicts.

Output per paper: results/baseline/<paper_id>.json
  {
    "paper_id": str,
    "model": str,
    "critique_points": {"point_001": "...", "point_002": "...", ...}
  }
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
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
Given a paper's title and abstract (and optionally its full text), identify
the most important weaknesses, limitations, and areas for improvement.

Output ONLY valid JSON: a dict mapping "point_NNN" keys to critique strings.
Example: {{"point_001": "...", "point_002": "..."}}
Aim for 5–12 distinct, substantive critique points.
"""

USER_TEMPLATE = """\
Title: {title}

Abstract:
{abstract}

{full_text_section}

Produce a JSON critique dictionary of the paper's weaknesses.
"""


def format_user_message(title: str, abstract: str, full_text: str = "") -> str:
    full_text_section = ""
    if full_text:
        # Truncate to avoid exceeding context limits
        truncated = full_text[:8000]
        full_text_section = f"Full text (truncated):\n{truncated}\n"
    return USER_TEMPLATE.format(
        title=title,
        abstract=abstract,
        full_text_section=full_text_section,
    )


# ── Single paper critique ──────────────────────────────────────────────────────

def critique_paper(
    paper_id: str,
    title: str,
    abstract: str,
    full_text: str = "",
    cfg: dict = None,
) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = cfg["models"]["baseline"]

    response = client.messages.create(
        model=model,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": format_user_message(title, abstract, full_text)}
        ],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        points: dict[str, str] = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [ERROR] JSON parse failed for {paper_id}: {exc}")
        points = {}

    return {
        "paper_id": paper_id,
        "model": model,
        "critique_points": points,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_baseline(reviews_path: str, output_dir: str, cfg: dict) -> None:
    with open(reviews_path) as f:
        all_papers: dict = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for paper_id, paper in all_papers.items():
        out_file = out_dir / f"{paper_id}.json"
        if out_file.exists():
            print(f"  [SKIP] {paper_id}")
            continue

        print(f"  [RUN ] {paper_id} …")
        result = critique_paper(
            paper_id=paper_id,
            title=paper.get("title", paper_id),
            abstract=paper.get("abstract", ""),
            cfg=cfg,
        )

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"         → {len(result['critique_points'])} points")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_baseline(
        reviews_path=cfg["data"]["reviews_file"],
        output_dir=cfg["results"]["baseline_dir"],
        cfg=cfg,
    )
