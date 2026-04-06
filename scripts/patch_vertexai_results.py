"""
patch_vertexai_results.py
--------------------------
Retroactively repairs all result files in results/vertexai/:

1. Adds "platform": "vertexai" to every file that is missing it.
2. For zero-point papers (critique_points == {}), re-parses the Summarizer
   entry from the transcript using the fixed _parse_structured_output logic,
   then re-flattens to critique_points and updates structured in-place.

Run from the GenAI_Group_Project root:
    python scripts/patch_vertexai_results.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# ── Parsing helpers (mirrors the fixed vertex_orchestrator.py logic) ──────────

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
                # Double the backslash to make it a valid JSON escape
                result.append('\\\\')
                i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def _parse_structured_output(raw: str) -> dict:
    """Fixed fence-stripping parser — robust to all markdown fence variants
    and invalid JSON escape sequences (e.g. LaTeX in Summarizer output)."""
    text = raw.strip()

    # Strip markdown fences (handles ```json, ```, with/without trailing newline)
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

    # Fallback: extract the outermost JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            try:
                return json.loads(_sanitize_json_escapes(match.group()))
            except json.JSONDecodeError:
                pass

    return {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "scores": {"correctness": 3, "novelty": 3, "recommendation": "borderline", "confidence": 1},
    }


def _flatten_to_critique_points(structured: dict) -> dict[str, str]:
    """Convert structured weaknesses to flat critique_points dict."""
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


# ── Main patch logic ──────────────────────────────────────────────────────────

def patch_results(results_dir: str) -> None:
    results_path = Path(results_dir)
    files = sorted(results_path.glob("paper_*.json"))

    if not files:
        print(f"No paper_*.json files found in {results_dir}")
        sys.exit(1)

    platform_patched = 0
    zero_point_repaired = 0
    zero_point_still_empty = 0

    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [ERROR] Could not load {fpath.name}: {exc}")
            continue

        modified = False

        # ── Fix 1: add platform field ─────────────────────────────────────────
        if "platform" not in data:
            data["platform"] = "vertexai"
            modified = True
            platform_patched += 1

        # ── Fix 2: re-parse zero-point papers from transcript ─────────────────
        if data.get("critique_points") == {}:
            # Find the Summarizer entry in the transcript
            summarizer_content = None
            for entry in data.get("transcript", []):
                if entry.get("role", "").lower().startswith("summariz"):
                    summarizer_content = entry.get("content", "")
                    break

            if summarizer_content:
                structured = _parse_structured_output(summarizer_content)
                critique_points = _flatten_to_critique_points(structured)

                if critique_points:
                    data["structured"] = structured
                    data["critique_points"] = critique_points
                    modified = True
                    zero_point_repaired += 1
                    print(f"  [REPAIRED] {fpath.name}: {len(critique_points)} points recovered")
                else:
                    zero_point_still_empty += 1
                    print(f"  [WARN] {fpath.name}: still 0 points after re-parse "
                          f"(Summarizer may have returned unparseable content)")
            else:
                zero_point_still_empty += 1
                print(f"  [WARN] {fpath.name}: no Summarizer entry found in transcript")

        if modified:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"  Files processed  : {len(files)}")
    print(f"  platform added   : {platform_patched}")
    print(f"  zero-pt repaired : {zero_point_repaired}")
    if zero_point_still_empty:
        print(f"  zero-pt remaining: {zero_point_still_empty}  (need API re-run)")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Default: run from GenAI_Group_Project root
    results_dir = Path(__file__).parent.parent / "results" / "vertexai"
    patch_results(str(results_dir))
