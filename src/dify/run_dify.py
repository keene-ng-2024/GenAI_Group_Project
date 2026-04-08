"""
run_dify.py
-----------
Automates running all 100 papers through the Dify workflow.

Steps for each paper:
1. Find the PDF in data/papers/ by matching title to filename
2. Upload the PDF to Dify file upload API
3. Run the Dify workflow with the uploaded file
4. Parse the JSON output
5. Save to results/dify/<mode>/paper_XXXX.json

Usage:
    python -m src.dify.run_dify single_critic   # Reader → Critic → Summariser
    python -m src.dify.run_dify dual_critic     # Reader → Critic 1 → Auditor → Critic 2 → Summariser
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

DIFY_API_KEYS = {
    "single_critic": os.environ["DIFY_API_KEY_SINGLE"],
    "dual_critic": os.environ["DIFY_API_KEY_DUAL"],
}

DIFY_BASE_URL = "https://api.dify.ai/v1"

CHECKPOINT_FILE = Path("data/checkpoint.json")
PAPERS_DIR = Path("data/papers")

SLEEP_BETWEEN_PAPERS = 3  # seconds, to avoid rate limiting


# ── PDF matching ────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_pdf(title: str, pdf_files: list[Path]) -> Path | None:
    norm_title = normalize(title)
    best_match = None
    best_ratio = 0.0

    for f in pdf_files:
        norm_name = normalize(f.stem)
        ratio = difflib.SequenceMatcher(None, norm_title, norm_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = f

    if best_ratio >= 0.5:
        return best_match
    return None


# ── Dify API calls ──────────────────────────────────────────────────────────────

def upload_file(pdf_path: Path, headers: dict) -> str | None:
    """Upload a PDF to Dify and return the upload_file_id."""
    url = f"{DIFY_BASE_URL}/files/upload"
    try:
        with open(pdf_path, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                files={"file": (pdf_path.name, f, "application/pdf")},
                data={"user": "automation"},
            )
        response.raise_for_status()
        return response.json()["id"]
    except Exception as e:
        print(f"    [ERROR] File upload failed: {e}")
        return None


def run_workflow(upload_file_id: str, headers: dict) -> dict | None:
    """Run the Dify workflow using streaming mode to avoid gateway timeouts."""
    url = f"{DIFY_BASE_URL}/workflows/run"
    payload = {
        "inputs": {
            "single_pdf": {
                "transfer_method": "local_file",
                "upload_file_id": upload_file_id,
                "type": "document",
            }
        },
        "response_mode": "streaming",
        "user": "automation",
    }
    try:
        response = requests.post(
            url,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=600,  # 10 minutes total
        )
        response.raise_for_status()

        # Read SSE stream and look for workflow_finished event
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                    if data.get("event") == "workflow_finished":
                        return {"data": data.get("data", {})}
                except json.JSONDecodeError:
                    continue

        print("    [ERROR] Stream ended without workflow_finished event")
        return None
    except Exception as e:
        print(f"    [ERROR] Workflow run failed: {e}")
        return None


def parse_output(workflow_response: dict) -> dict | None:
    """Extract and parse the final JSON from the workflow response."""
    try:
        outputs = workflow_response["data"]["outputs"]
        print(f"    [DEBUG] Output keys: {list(outputs.keys())}")

        # Try all output keys
        raw = outputs.get("final_review") or outputs.get("text") or ""
        if not raw:
            for k, v in outputs.items():
                print(f"    [DEBUG] {k}: {str(v)[:200]}")
            print(f"    [ERROR] No output found in workflow response")
            return None

        print(f"    [DEBUG] Raw output (first 300 chars): {str(raw)[:300]}")

        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"```$", "", raw.strip())

        return json.loads(raw.strip())
    except (KeyError, json.JSONDecodeError) as e:
        print(f"    [ERROR] Failed to parse output: {e}")
        return None


def build_critique_points(structured: dict) -> dict[str, str]:
    """Build flat critique_points dict from structured weaknesses for scorer compatibility."""
    points = {}
    for i, item in enumerate(structured.get("weaknesses", []), 1):
        if isinstance(item, str):
            full = item
        elif isinstance(item, dict):
            point_text = item.get("point", "")
            evidence = item.get("evidence", "")
            full = f"{point_text}. {evidence}".strip(" .") if evidence else point_text
        else:
            continue
        points[f"point_{i:03d}"] = full
    return points


# ── Main pipeline ───────────────────────────────────────────────────────────────

def main(mode: str) -> None:
    if mode not in ("single_critic", "dual_critic"):
        print(f"Unknown mode '{mode}'. Use 'single_critic' or 'dual_critic'.")
        sys.exit(1)

    api_key = DIFY_API_KEYS[mode]
    if not api_key:
        print(f"[ERROR] DIFY_API_KEY_{'SINGLE' if mode == 'single_critic' else 'DUAL'} is not set in .env")
        sys.exit(1)
    headers = {"Authorization": f"Bearer {api_key}"}

    output_dir = Path(f"results/dify/{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(CHECKPOINT_FILE) as f:
        checkpoint = json.load(f)

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Mode: {mode}")
    print(f"Found {len(pdf_files)} PDFs in {PAPERS_DIR}")
    print(f"Processing {len(checkpoint)} papers...\n")

    success = 0
    skipped = 0
    failed = 0

    for idx_str, paper in sorted(checkpoint.items(), key=lambda x: int(x[0])):
        idx = int(idx_str)
        paper_id = f"paper_{idx:04d}"
        output_file = output_dir / f"{paper_id}.json"
        title = paper["title"]

        # Skip already processed papers
        if output_file.exists():
            print(f"  [SKIP] {paper_id} — already done")
            skipped += 1
            continue

        print(f"  [{idx:03d}/100] {paper_id}: {title[:60]}...")

        # Find PDF
        pdf_path = find_pdf(title, pdf_files)
        if pdf_path is None:
            print(f"    [SKIP] No matching PDF found for: {title}")
            skipped += 1
            continue
        print(f"    PDF: {pdf_path.name}")

        # Upload PDF
        upload_id = upload_file(pdf_path, headers)
        if upload_id is None:
            failed += 1
            continue

        # Run workflow
        result = run_workflow(upload_id, headers)
        if result is None:
            failed += 1
            time.sleep(SLEEP_BETWEEN_PAPERS)
            continue

        # Check workflow status
        status = result.get("data", {}).get("status", "unknown")
        if status != "succeeded":
            print(f"    [ERROR] Workflow status: {status}")
            failed += 1
            time.sleep(SLEEP_BETWEEN_PAPERS)
            continue

        # Parse output
        structured = parse_output(result)
        if structured is None:
            failed += 1
            time.sleep(SLEEP_BETWEEN_PAPERS)
            continue

        # Extract latency
        latency = round(result.get("data", {}).get("elapsed_time", 0), 2)

        # Build critique_points from weaknesses for scorer compatibility
        critique_points = build_critique_points(structured)

        # Save result in standard format (matches n8n output structure)
        output = {
            "paper_id": paper_id,
            "title": title,
            "platform": f"dify_{mode}",
            "model": "gpt-4.1-mini",
            "latency_seconds": latency,
            "structured": structured,
            "critique_points": critique_points,
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        n_points = len(critique_points)
        print(f"    [OK] {n_points} weakness points → {output_file} (latency: {latency:.1f}s)")
        success += 1

        time.sleep(SLEEP_BETWEEN_PAPERS)

    print(f"\n{'='*50}")
    print(f"Done: {success} succeeded, {skipped} skipped, {failed} failed")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single_critic"
    main(mode)
