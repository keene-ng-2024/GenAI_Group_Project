"""
build_critique_dict.py
----------------------
For each paper, call an LLM to distil N human reviews into a deduplicated
dictionary of unique critique points:

  {
    "point_001": "The paper does not include ablation studies for ...",
    "point_002": "The baseline comparisons are missing ...",
    ...
  }

Uses OpenAI model for distillation.
"""

from __future__ import annotations

import json
import os
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
You are a meticulous research assistant.  Your task is to read multiple
peer reviews of a scientific paper and extract a deduplicated set of
unique critique points.

Rules:
1. Each point must be a *distinct* issue (methodology, evaluation, clarity, novelty, etc.).
2. Merge near-duplicate points into one; do NOT list the same concern twice.
3. Paraphrase in your own words — do NOT copy reviewer text verbatim.
4. Output ONLY valid JSON: a dict mapping "point_NNN" keys to critique strings.
   Example: {{"point_001": "...", "point_002": "..."}}
5. Include between {min_points} and {max_points} points.
"""

USER_TEMPLATE = """\
Paper title: {title}

=== HUMAN REVIEWS ===
{reviews_block}
=== END REVIEWS ===

Now produce the JSON critique dictionary.
"""


def build_reviews_block(reviews: list[dict]) -> str:
    parts = []
    for i, r in enumerate(reviews, 1):
        rating = f"  Rating: {r['rating']}" if r.get("rating") else ""
        parts.append(f"--- Reviewer {i}{rating} ---\n{r['text'].strip()}")
    return "\n\n".join(parts)


# ── LLM call ──────────────────────────────────────────────────────────────────

def distil_critique_points(
    paper_id: str,
    title: str,
    reviews: list[dict],
    cfg: dict,
) -> dict[str, str]:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    min_p = cfg["critique_dict"]["min_points"]
    max_p = cfg["critique_dict"]["max_points"]

    system = SYSTEM_PROMPT.format(min_points=min_p, max_points=max_p)
    user = USER_TEMPLATE.format(
        title=title,
        reviews_block=build_reviews_block(reviews),
    )

    response = client.chat.completions.create(
        model=cfg["models"]["strong"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        points: dict[str, str] = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [ERROR] JSON parse failed for {paper_id}: {exc}\n{raw[:300]}")
        points = {}

    return points


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_all_critique_dicts(
    reviews_path: str,
    output_dir: str,
    cfg: dict,
) -> None:
    with open(reviews_path) as f:
        all_papers: dict = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for paper_id, paper in all_papers.items():
        out_file = out_dir / f"{paper_id}.json"
        if out_file.exists():
            print(f"  [SKIP] {paper_id} (already built)")
            continue

        reviews = paper.get("reviews", [])
        if not reviews:
            print(f"  [SKIP] {paper_id} (no reviews)")
            continue

        print(f"  [BUILD] {paper_id} ({len(reviews)} reviews) …")
        points = distil_critique_points(
            paper_id=paper_id,
            title=paper.get("title", paper_id),
            reviews=reviews,
            cfg=cfg,
        )

        with open(out_file, "w") as f:
            json.dump(points, f, indent=2)

        print(f"         → {len(points)} unique points saved to {out_file}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    build_all_critique_dicts(
        reviews_path=cfg["data"]["reviews_file"],
        output_dir=cfg["data"]["critique_dicts_dir"],
        cfg=cfg,
    )
