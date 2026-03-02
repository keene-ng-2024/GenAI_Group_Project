"""
parse_reviews.py
----------------
Extract and normalise text from raw human-review files.

Supported input formats
  - OpenReview JSON exports  (list of review dicts with 'content' field)
  - Plain-text / PDF         (PDF → text via pypdf)

Output: data/processed/reviews_parsed.json
  {
    "<paper_id>": {
      "title": str,
      "abstract": str,
      "reviews": [
        {"reviewer": str, "rating": int|None, "text": str},
        ...
      ]
    },
    ...
  }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # PDF support is optional


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Readers ────────────────────────────────────────────────────────────────────

def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("pypdf is required to read PDF files: pip install pypdf")
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_openreview_json(path: Path) -> dict[str, Any]:
    """Parse a single OpenReview JSON export file for one paper."""
    with open(path) as f:
        data = json.load(f)

    # data may be a list of notes (reviews) or a dict with 'notes' key
    notes = data if isinstance(data, list) else data.get("notes", [])

    paper_meta: dict[str, Any] = {"title": "", "abstract": "", "reviews": []}

    for note in notes:
        content = note.get("content", {})
        invitation = note.get("invitation", "")

        # Detect paper submission note
        if "submission" in invitation.lower() or "paper" in invitation.lower():
            paper_meta["title"] = _extract_field(content, ["title"])
            paper_meta["abstract"] = _extract_field(content, ["abstract"])
            continue

        # Detect review note
        if "review" in invitation.lower() or "official" in invitation.lower():
            text = _extract_field(content, ["review", "main_review", "comments"])
            rating_raw = _extract_field(content, ["rating", "recommendation", "score"])
            paper_meta["reviews"].append(
                {
                    "reviewer": note.get("signatures", ["anonymous"])[0],
                    "rating": _parse_rating(rating_raw),
                    "text": text,
                }
            )

    return paper_meta


def _extract_field(content: dict, keys: list[str]) -> str:
    for key in keys:
        val = content.get(key, "")
        if isinstance(val, dict):
            val = val.get("value", "")
        if val:
            return str(val).strip()
    return ""


def _parse_rating(raw: str) -> int | None:
    match = re.search(r"\d+", str(raw))
    return int(match.group()) if match else None


# ── Main parsing pipeline ──────────────────────────────────────────────────────

def parse_all_reviews(raw_dir: str, output_path: str) -> dict[str, Any]:
    raw = Path(raw_dir)
    all_papers: dict[str, Any] = {}

    json_files = list(raw.glob("**/*.json"))
    pdf_files = list(raw.glob("**/*.pdf"))

    for jf in json_files:
        paper_id = jf.stem
        try:
            all_papers[paper_id] = parse_openreview_json(jf)
            print(f"  [OK] {jf.name}  ({len(all_papers[paper_id]['reviews'])} reviews)")
        except Exception as exc:
            print(f"  [WARN] Skipping {jf.name}: {exc}")

    for pf in pdf_files:
        paper_id = pf.stem
        if paper_id in all_papers:
            continue  # already handled by JSON
        try:
            text = read_pdf(pf)
            all_papers[paper_id] = {
                "title": paper_id,
                "abstract": "",
                "reviews": [{"reviewer": "unknown", "rating": None, "text": text}],
            }
            print(f"  [OK] {pf.name} (PDF)")
        except Exception as exc:
            print(f"  [WARN] Skipping {pf.name}: {exc}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_papers, f, indent=2)

    print(f"\nSaved {len(all_papers)} papers → {output_path}")
    return all_papers


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    parse_all_reviews(
        raw_dir=cfg["data"]["raw_dir"],
        output_path=cfg["data"]["reviews_file"],
    )
