"""
n8n_critique.py
---------------
Python adapter that drives the n8n multi-agent critique workflows.

Two workflow modes:
  noloop — Reader → Critic → Summariser
            webhook path: /paper-critique-noloop
            output dir:   results/n8n_noloop/

  1round  — Reader → Critic 1 → Auditor → Critic 2 → Summariser
            webhook path: /paper-critique
            output dir:   results/n8n/

Usage:
  python -m src.platforms.n8n_critique noloop
  python -m src.platforms.n8n_critique 1round
"""

from __future__ import annotations

import json
import re
import sys
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


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_structured_output(raw: str | dict) -> dict:
    """Parse structured JSON from n8n response with fallbacks."""
    if isinstance(raw, dict) and "weaknesses" in raw:
        return raw

    text = str(raw).strip()

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
    mode: str,
    cfg: dict,
    timeout: int = 300,
) -> dict:
    """POST a paper to the n8n webhook and return the structured result."""
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

    body = response.json()

    # New workflows return critique_points + structured + transcript directly
    if "critique_points" in body and "structured" in body:
        return {
            "paper_id": paper_id,
            "title": title,
            "platform": f"n8n_{mode}",
            "model": "gpt-4.1-mini",
            "latency_seconds": latency_seconds,
            "structured": body["structured"],
            "critique_points": body["critique_points"],
            "transcript": body.get("transcript", {}),
        }

    # Fallback: parse structured output manually
    raw = body.get("output") or body.get("structured") or body
    structured = _parse_structured_output(raw)
    critique_points = _flatten_to_critique_points(structured)

    return {
        "paper_id": paper_id,
        "title": title,
        "platform": f"n8n_{mode}",
        "model": "gpt-4.1-mini",
        "latency_seconds": latency_seconds,
        "structured": structured,
        "critique_points": critique_points,
        "transcript": {},
    }


# ── Batch pipeline ─────────────────────────────────────────────────────────────

def run_all_papers(mode: str, cfg: dict) -> None:
    """
    Run all eval papers through the specified n8n workflow.

    Args:
        mode: "noloop" or "1round"
        cfg:  Config dict from config.yaml
    """
    n8n_cfg = cfg.get("n8n", {})

    if mode == "noloop":
        webhook_url = n8n_cfg.get("webhook_url_noloop", "")
        output_dir = cfg["results"].get("n8n_noloop_dir", "results/n8n_noloop")
    elif mode == "1round":
        webhook_url = n8n_cfg.get("webhook_url", "")
        output_dir = cfg["results"].get("n8n_dir", "results/n8n")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'noloop' or '1round'.")

    if not webhook_url:
        raise ValueError(
            f"Webhook URL for mode '{mode}' is not set in config.yaml.\n"
            "Set n8n.webhook_url (1round) or n8n.webhook_url_noloop (noloop)."
        )

    reviews_path = cfg["data"]["reviews_file"]
    with open(reviews_path) as f:
        all_papers: dict = json.load(f)

    # Only run eval split
    eval_ids = set()
    eval_path = cfg["data"].get("eval_split", "data/eval_split.jsonl")
    if Path(eval_path).exists():
        with open(eval_path) as f:
            for line in f:
                row = json.loads(line)
                eval_ids.add(row.get("paper_id", ""))
        papers_to_run = {k: v for k, v in all_papers.items() if k in eval_ids}
        print(f"  [INFO] Running {len(papers_to_run)} eval papers (mode={mode})")
    else:
        papers_to_run = all_papers
        print(f"  [INFO] eval_split.jsonl not found, running all {len(papers_to_run)} papers")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for paper_id, paper in papers_to_run.items():
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
                mode=mode,
                cfg=cfg,
            )
        except requests.exceptions.ConnectionError:
            print(f"  [ERROR] Cannot reach n8n at {webhook_url}. Is n8n running?")
            break
        except requests.exceptions.Timeout:
            print(f"  [ERROR] {paper_id} timed out (>{300}s)")
            continue
        except Exception as exc:
            print(f"  [ERROR] {paper_id} failed: {exc}")
            continue

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        n_points = len(result["critique_points"])
        print(f"  [SAVED] {n_points} weakness points → {out_file}  ({result['latency_seconds']}s)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("noloop", "1round"):
        print("Usage: python -m src.platforms.n8n_critique [noloop|1round]")
        sys.exit(1)

    mode = sys.argv[1]
    cfg = load_config()
    run_all_papers(mode=mode, cfg=cfg)
