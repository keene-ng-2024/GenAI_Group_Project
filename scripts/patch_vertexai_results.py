"""
patch_vertexai_results.py
--------------------------
Retroactively repairs all result files in results/vertexai/:

1. Adds "platform": "vertexai" to every file that is missing it.
2. For zero-point papers (critique_points == {}), re-parses the Summarizer
   entry from the transcript using the fixed parser, then re-flattens.

Handles multiple Summariser output schemas:
- Canonical: {"summary":..., "weaknesses":[...], ...}
- Wrapped:   {"review": {"weaknesses":[...]}}
- Alternate: {"structured_review": {"critique_points":[...]}}

Run from the GenAI_Group_Project root:
    python scripts/patch_vertexai_results.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_VALID_JSON_ESCAPES = set('"' + '\\' + '/' + 'bfnrtu')


def _sanitize_json_escapes(text: str) -> str:
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


def _try_parse_json(text: str):
    """Try json.loads, then with escape sanitisation, then None."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_sanitize_json_escapes(text))
    except json.JSONDecodeError:
        return None


def _parse_structured_output(raw: str) -> dict:
    """Parse Summariser output robustly, handling fences and alternate schemas."""
    text = raw.strip()

    # Strip markdown fences
    m = re.match(r'^`{3}(?:json)?\s*([\s\S]*?)\s*`{3}\s*$', text)
    if m:
        text = m.group(1).strip()

    parsed = _try_parse_json(text)

    if parsed is None:
        # Try extracting outermost JSON object
        m2 = re.search(r'\{[\s\S]*\}', text)
        if m2:
            parsed = _try_parse_json(m2.group())

    if parsed is None:
        return {
            "summary": "", "strengths": [], "weaknesses": [], "questions": [],
            "scores": {"correctness": 3, "novelty": 3,
                       "recommendation": "borderline", "confidence": 1},
        }

    # Normalise alternate schemas
    if "weaknesses" not in parsed:
        # One level of nesting: {"review": {...}, "structured_review": {...}}
        for v in parsed.values():
            if isinstance(v, dict) and "weaknesses" in v:
                parsed = v
                break

    # Still no weaknesses — look for any list of dicts with point/critique/issue keys
    if "weaknesses" not in parsed:
        for val in parsed.values():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                keys = set(val[0].keys())
                if keys & {"point", "critique", "issue", "weakness", "critique_point"}:
                    parsed["weaknesses"] = val
                    break

    return parsed


def _flatten_to_critique_points(structured: dict) -> dict:
    points = {}
    idx = 1
    for item in structured.get("weaknesses", []):
        if isinstance(item, str):
            full = item
        elif isinstance(item, dict):
            # Handle various key names
            point_text = (item.get("point") or item.get("critique") or
                          item.get("issue") or item.get("weakness") or "")
            evidence = item.get("evidence", "")
            full = f"{point_text}. {evidence}".strip(" .") if evidence else point_text
        else:
            continue
        if full:
            points[f"point_{idx:03d}"] = full
            idx += 1
    return points


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

        # Fix 1: add platform field
        if "platform" not in data:
            data["platform"] = "vertexai"
            modified = True
            platform_patched += 1

        # Fix 2: re-parse zero-point papers from transcript
        if data.get("critique_points") == {}:
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
                    print(f"  [WARN] {fpath.name}: still 0 points after re-parse")
            else:
                zero_point_still_empty += 1
                print(f"  [WARN] {fpath.name}: no Summarizer entry in transcript")

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
    results_dir = Path(__file__).parent.parent / "results" / "vertexai"
    patch_results(str(results_dir))
