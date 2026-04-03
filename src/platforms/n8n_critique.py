"""
n8n_critique.py
---------------
Python adapter that drives the n8n multi-agent critique workflow.

Architecture
  This module POSTs each paper to the n8n webhook URL.
  The n8n workflow handles the Reader → Critic → Auditor → Summariser
  pipeline by calling the Anthropic API at each step, then returns
  the Summariser's structured JSON output.

  Python (this file)              n8n workflow
  ─────────────────               ────────────────────────────────────
  POST {paper_id, title,    ───►  Webhook trigger
        paper_text}               │
                                  ├─ Reader Agent   (Anthropic API)
                                  ├─ Critic Agent   (Anthropic API)
                                  ├─ Auditor Agent  (Anthropic API)
                                  ├─ Summariser     (Anthropic API)
  ◄─── {structured JSON}          └─ Respond to Webhook

Setup
  1. Import n8n_workflow.json into your n8n instance.
  2. Activate the workflow and copy the webhook URL (e.g.
     http://localhost:5678/webhook/paper-critique).
  3. Set n8n.webhook_url in config.yaml.

Output per paper: results/n8n/<paper_id>.json  (same schema as baseline)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Output parsing (same logic as orchestrator) ────────────────────────────────

def _parse_structured_output(raw: str | dict) -> dict:
    """Parse structured JSON from n8n response with fallbacks."""
    # If n8n already parsed it into a dict, use directly
    if isinstance(raw, dict) and "weaknesses" in raw:
        return raw

    text = str(raw).strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

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

    print("  [WARN] Failed to parse n8n structured output, returning empty structure")
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


# ── Single paper ───────────────────────────────────────────────────────────────

def critique_paper_via_n8n(
    paper_id: str,
    paper: dict,
    webhook_url: str,
    cfg: dict,
    timeout: int = 300,
) -> dict:
    """
    POST a paper to the n8n webhook and return the structured result.

    Args:
        paper_id:    Unique paper identifier.
        paper:       Full paper dict from reviews_parsed.json.
        webhook_url: n8n webhook URL (e.g. http://localhost:5678/webhook/paper-critique).
        cfg:         Config dict.
        timeout:     HTTP timeout in seconds (n8n pipelines take ~60-120s per paper).
    """
    truncate_chars = cfg["agent"].get("truncate_body_chars", 12000)
    title = paper.get("title", paper_id)
    full_text = paper.get("full_text", "")
    paper_text = full_text[:truncate_chars] if full_text else paper.get("abstract", title)

    payload = {
        "paper_id": paper_id,
        "title": title,
        "paper_text": paper_text,
    }

    start_time = time.perf_counter()

    response = requests.post(
        webhook_url,
        json=payload,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()

    latency_seconds = round(time.perf_counter() - start_time, 2)

    # n8n may return the structured dict directly, or wrap it in {"output": ...}
    body = response.json()
    raw = body.get("output") or body.get("structured") or body

    structured = _parse_structured_output(raw)
    critique_points = _flatten_to_critique_points(structured)

    return {
        "paper_id": paper_id,
        "title": title,
        "platform": "n8n",
        "model": "gpt-4o",
        "latency_seconds": latency_seconds,
        "structured": structured,
        "critique_points": critique_points,
    }


# ── Batch pipeline ─────────────────────────────────────────────────────────────

def run_all_papers(reviews_path: str, output_dir: str, cfg: dict) -> None:
    webhook_url = cfg.get("n8n", {}).get("webhook_url", "")
    if not webhook_url:
        raise ValueError(
            "n8n.webhook_url is not set in config.yaml. "
            "Import n8n_workflow.json into n8n, activate it, and paste the webhook URL."
        )

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
            result = critique_paper_via_n8n(
                paper_id=paper_id,
                paper=paper,
                webhook_url=webhook_url,
                cfg=cfg,
            )
        except requests.exceptions.ConnectionError:
            print(f"  [ERROR] Cannot reach n8n at {webhook_url}. Is n8n running?")
            break
        except requests.exceptions.Timeout:
            print(f"  [ERROR] {paper_id} timed out — n8n took >300s")
            continue
        except Exception as exc:
            print(f"  [ERROR] {paper_id} failed: {exc}")
            continue

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        n_points = len(result["critique_points"])
        print(f"\n  [SAVED] {n_points} weakness points → {out_file}  ({result['latency_seconds']}s)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_all_papers(
        reviews_path=cfg["data"]["reviews_file"],
        output_dir=cfg["results"]["n8n_dir"],
        cfg=cfg,
    )
