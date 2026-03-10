"""
parse_reviews.py
----------------
Extract and normalise text from raw human-review files.

Supported input formats
  - ReviewCritique JSONL     (one paper per line, review#1..N keys)
  - OpenReview JSON exports  (list of review dicts with 'content' field)
  - Plain-text / PDF         (PDF → text via pypdf)

Output: data/processed/reviews_parsed.json
  {
    "<paper_id>": {
      "title": str,
      "abstract": str,
      "full_text": str,
      "decision": str,
      "reviews": [
        {
          "reviewer": str,
          "rating": int|None,       # Recommendation score normalised to 1-5
          "text": str,              # concatenated segment_text values
          "scores": dict            # raw score fields from the source
        },
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


# ── ReviewCritique JSONL parser ────────────────────────────────────────────────

def _normalise_score(raw: str) -> int | None:
    """Parse a score string like ' 8' and return an int, or None on failure."""
    try:
        return int(str(raw).strip())
    except (ValueError, TypeError):
        return None


def _recommendation_to_rating(raw: str) -> int | None:
    """Convert Recommendation (1-10 scale) to 1-5 by halving and rounding."""
    val = _normalise_score(raw)
    if val is None:
        return None
    return max(1, min(5, round(val / 2)))


def parse_jsonl(jsonl_path: str, output_path: str) -> dict[str, Any]:
    """
    Parse ReviewCritique.jsonl into the standard reviews_parsed.json format.

    Each line has: decision, title, body_text, review#1 .. review#N
    Each review#N has:
      - review: list of {segment_text, topic_class_1, ...}
      - score:  {Correctness, Technical Novelty And Significance,
                 Empirical Novelty And Significance, Flag For Ethics Review,
                 Recommendation, Confidence}

    paper_id is assigned as 'paper_NNNN' (1-indexed line number) to avoid
    collisions from title slugification.
    """
    all_papers: dict[str, Any] = {}
    paper_count = 0

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Skipping malformed JSON on line {line_num}: {exc}")
                continue

            paper_count += 1
            paper_id = f"paper_{paper_count:04d}"

            # Collect review#N keys dynamically (handles 3, 4, or 5 reviews)
            review_keys = sorted(k for k in entry if k.startswith("review#"))
            reviews = []
            for rk in review_keys:
                r = entry[rk]

                # Join all non-empty segment_text values into one review string
                segments = r.get("review", [])
                text = "\n".join(
                    seg["segment_text"]
                    for seg in segments
                    if isinstance(seg.get("segment_text"), str)
                    and seg["segment_text"].strip()
                )

                raw_scores: dict = r.get("score", {})
                rating = _recommendation_to_rating(raw_scores.get("Recommendation", ""))

                reviews.append({
                    "reviewer": rk,
                    "rating": rating,
                    "text": text,
                    "scores": {k: str(v).strip() for k, v in raw_scores.items()},
                })

            all_papers[paper_id] = {
                "title": entry.get("title", "").strip(),
                "abstract": "",          # not separately available in JSONL
                "full_text": entry.get("body_text", ""),
                "decision": entry.get("decision", "").strip(),
                "reviews": reviews,
            }

            print(f"  [OK] {paper_id}  '{entry.get('title', '')[:50]}'  "
                  f"({len(reviews)} reviews, decision={entry.get('decision', '?')})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_papers, f, indent=2)

    print(f"\nSaved {len(all_papers)} papers → {output_path}")
    return all_papers


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
    import sys
    cfg = load_config()

    # If a .jsonl file is passed as argument, use the JSONL parser.
    # Otherwise fall back to the OpenReview JSON / PDF parser.
    if len(sys.argv) > 1 and sys.argv[1].endswith(".jsonl"):
        parse_jsonl(
            jsonl_path=sys.argv[1],
            output_path=cfg["data"]["reviews_file"],
        )
    elif Path(cfg["data"].get("jsonl_file", "")).exists():
        parse_jsonl(
            jsonl_path=cfg["data"]["jsonl_file"],
            output_path=cfg["data"]["reviews_file"],
        )
    else:
        parse_all_reviews(
            raw_dir=cfg["data"]["raw_dir"],
            output_path=cfg["data"]["reviews_file"],
        )
