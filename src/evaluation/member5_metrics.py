#!/usr/bin/env python3
"""
src/evaluation/member5_metrics.py
Member 5 – Evaluation & Metrics Lead

Additive evaluation module. Adds new outputs without modifying any existing files.

Five metrics
------------
1. Coverage   – per-reference semantic recall/precision/F1, macro-averaged over
                up to N human reviews extracted from ReviewCritique.jsonl.
2. Grounding  – fraction of generated critique points that have sentence-level
                semantic support in the paper body text (sim >= threshold).
3. Overlap    – mean pairwise cosine similarity among generated critique points
                (redundancy indicator; lower = more diverse critique).
4. Judge      – loaded from results/<mode>/judge_scores.json if present; null
                if absent. No API calls are made.
5. Latency    – latency_seconds read from each per-paper result JSON.

Multi-reference heuristic (documented)
---------------------------------------
Each paper in ReviewCritique.jsonl has up to N review objects (review#1 …).
For each review we extract segments whose topic_class_1 does NOT belong to the
NON_CRITIQUE_TOPICS set (Paper Summary, Strengths, Rating, Confidence, Score,
etc.).  The remaining segment_text values are concatenated, then split into
individual sentences.  Each review produces one reference sentence list.
Per-reference P/R/F1 is computed via cosine-similarity matching at the
configured threshold, then macro-averaged across all reviews.

Paper-ID → JSONL matching
--------------------------
result["paper_id"] is expected to follow the convention "paper_NNN" (1-indexed,
zero-padded) established by src/data_processing/parse_reviews.py.  Index N-1 is
used to look up the corresponding JSONL entry.  If the format does not match,
the function returns None and JSONL-dependent metrics (coverage, grounding) will
be null for that paper.

Outputs (all additive – no existing files modified)
----------------------------------------------------
results/<mode>/member5_scores.json    per-paper + aggregate
results/<mode>/member5_per_paper.csv  per-paper flat table
results/member5_comparison.csv        baseline vs agents aggregates
results/member5_comparison.png        bar chart – key metrics
results/member5_latency.png           latency comparison bar chart

CLI
---
python -m src.evaluation.member5_metrics baseline
python -m src.evaluation.member5_metrics agents
python -m src.evaluation.member5_metrics both
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# topic_class_1 values (lower-cased) that do NOT represent a critique point.
# Segments with these topics are excluded when building human-review references.
NON_CRITIQUE_TOPICS: frozenset[str] = frozenset(
    {
        "paper summary",
        "summary",
        "abstract",
        "strengths",
        "strength",
        "positive aspects",
        "positive",
        "rating",
        "confidence",
        "score",
        "recommendation",
        "ethics",
        "flag for ethics review",
        "ethics review",
    }
)

# Result files that are outputs, not per-paper inputs; skip when globbing.
SKIP_FILES: frozenset[str] = frozenset(
    {
        "scores.json",
        "judge_scores.json",
        "member5_scores.json",
    }
)

# Repo root (two levels above this file: src/evaluation/member5_metrics.py)
ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load config.yaml, resolving relative paths against the repo root."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_normalize(texts: list[str], embedder: SentenceTransformer) -> np.ndarray:
    """
    Embed a list of strings and L2-normalise each row.
    Returns array of shape (len(texts), dim).  Returns an empty (0, dim) array
    when texts is empty.
    """
    dim = embedder.get_sentence_embedding_dimension()
    if not texts:
        return np.empty((0, dim), dtype=np.float32)
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (embs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _sentences_from_text(text: str) -> list[str]:
    """
    Split a block of text into non-trivial sentences.
    A sentence must be > 10 characters after stripping whitespace.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


# ---------------------------------------------------------------------------
# Reference building from ReviewCritique.jsonl
# ---------------------------------------------------------------------------


def _is_critique_segment(seg: dict) -> bool:
    """
    Return True if this review segment likely contains a critique point.
    Segments are excluded when their topic_class_1 matches NON_CRITIQUE_TOPICS.
    """
    topic = str(seg.get("topic_class_1", "")).strip().lower()
    return bool(topic) and topic not in NON_CRITIQUE_TOPICS


def build_review_references(review_obj: dict) -> list[str]:
    """
    Extract critique-relevant sentences from one review object.

    review_obj["review"] is a list of segment dicts.  We keep segments whose
    topic_class_1 is not in NON_CRITIQUE_TOPICS, concatenate their
    segment_text values, then split into sentences.
    """
    segments = review_obj.get("review", [])
    texts: list[str] = []
    for seg in segments:
        if _is_critique_segment(seg):
            t = str(seg.get("segment_text", "")).strip()
            if t:
                texts.append(t)
    return _sentences_from_text(" ".join(texts))


def load_jsonl_papers(jsonl_path: Path) -> list[dict]:
    """Load all papers from ReviewCritique.jsonl (one JSON object per line)."""
    papers: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def get_paper_reviews(paper_entry: dict) -> dict[str, list[str]]:
    """
    Return {review_key: [sentence, ...]} for each review#N found in paper_entry.
    Only includes reviews that yield at least one non-trivial sentence.
    """
    refs: dict[str, list[str]] = {}
    for key in sorted(paper_entry.keys()):
        if re.match(r"review#\d+$", key):
            sents = build_review_references(paper_entry[key])
            if sents:
                refs[key] = sents
    return refs


def paper_index_from_id(paper_id: str, n_papers: int) -> int | None:
    """
    Parse a 1-based numeric suffix from paper_id (e.g. "paper_001" → 0).
    Returns a 0-based list index, or None if parsing fails or index is out of range.
    """
    m = re.search(r"(\d+)$", paper_id)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_papers:
            return idx
    return None


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> dict[str, dict]:
    """
    Load all per-paper result JSONs from results_dir, skipping known output files.
    Returns {paper_id: result_dict}.
    """
    results: dict[str, dict] = {}
    for f in sorted(results_dir.glob("*.json")):
        if f.name in SKIP_FILES:
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        paper_id = data.get("paper_id", f.stem)
        results[paper_id] = data
    return results


def extract_generated_points(result: dict) -> list[str]:
    """
    Extract generated critique point texts from a result dict.

    Priority:
      1. result["structured"]["weaknesses"] + result["structured"]["questions"]
         (structured output from both baseline and agent pipelines)
      2. Fallback: result["critique_points"] (flat {id: text} dict)

    Mirrors the extraction logic used by scorer.py and llm_judge.py.
    """
    points: list[str] = []
    structured = result.get("structured", {})
    weaknesses = structured.get("weaknesses", [])
    questions = structured.get("questions", [])

    if weaknesses or questions:
        for w in weaknesses:
            text = w.get("point", "") or w.get("text", "") if isinstance(w, dict) else str(w)
            if text.strip():
                points.append(text.strip())
        for q in questions:
            text = (
                q.get("question", "") or q.get("text", "") if isinstance(q, dict) else str(q)
            )
            if text.strip():
                points.append(text.strip())
    else:
        for v in result.get("critique_points", {}).values():
            if isinstance(v, str) and v.strip():
                points.append(v.strip())

    return points


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------


def _prf_from_embs(
    gen_embs: np.ndarray,
    ref_embs: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """
    Compute precision, recall, F1 via cosine-similarity threshold matching.

    gen_embs : (n_gen, dim) L2-normalised
    ref_embs : (n_ref, dim) L2-normalised

    Precision = fraction of generated points that match ≥1 reference
    Recall    = fraction of reference points covered by ≥1 generated point
    F1        = harmonic mean
    """
    if len(gen_embs) == 0 or len(ref_embs) == 0:
        return 0.0, 0.0, 0.0

    sim = gen_embs @ ref_embs.T  # (n_gen, n_ref)
    recall = float((sim.max(axis=0) >= threshold).mean())
    precision = float((sim.max(axis=1) >= threshold).mean())
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compute_coverage(
    gen_points: list[str],
    review_refs: dict[str, list[str]],
    embedder: SentenceTransformer,
    threshold: float,
) -> dict[str, Any]:
    """
    Multi-reference coverage.

    For each human review reference set, compute P/R/F1 against the generated
    points.  Then macro-average across all reviews.

    Returns a dict with:
      per_review : {review_key: {precision, recall, f1}}
      macro_precision, macro_recall, macro_f1 : floats (None if no refs)
    """
    gen_embs = embed_normalize(gen_points, embedder)
    per_review: dict[str, dict] = {}

    for rev_key, ref_sents in review_refs.items():
        ref_embs = embed_normalize(ref_sents, embedder)
        p, r, f = _prf_from_embs(gen_embs, ref_embs, threshold)
        per_review[rev_key] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "n_ref_sentences": len(ref_sents),
        }

    if not per_review:
        return {
            "per_review": {},
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
        }

    macro_p = float(np.mean([v["precision"] for v in per_review.values()]))
    macro_r = float(np.mean([v["recall"] for v in per_review.values()]))
    macro_f = float(np.mean([v["f1"] for v in per_review.values()]))

    return {
        "per_review": per_review,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f, 4),
    }


def compute_grounding(
    gen_points: list[str],
    body_text: str,
    embedder: SentenceTransformer,
    threshold: float,
    max_paper_sents: int = 500,
) -> dict[str, Any]:
    """
    Grounding: fraction of generated critique points that have semantic support
    in the paper's body text.

    Heuristic:
      - Split body_text into sentences; truncate to max_paper_sents for memory.
      - Embed both sets and L2-normalise.
      - For each generated point, find the maximum cosine similarity to any
        paper sentence.
      - A point is considered "grounded" if max_sim >= threshold.

    Returns:
      grounding_score : float (0-1) or None if inputs are empty
      n_grounded      : int
      n_total         : int
    """
    if not gen_points or not body_text:
        return {"grounding_score": None, "n_grounded": 0, "n_total": len(gen_points)}

    paper_sents = _sentences_from_text(body_text)[:max_paper_sents]
    if not paper_sents:
        return {"grounding_score": None, "n_grounded": 0, "n_total": len(gen_points)}

    gen_embs = embed_normalize(gen_points, embedder)   # (n_gen, dim)
    pap_embs = embed_normalize(paper_sents, embedder)  # (n_pap, dim)

    sim = gen_embs @ pap_embs.T                        # (n_gen, n_pap)
    grounded_mask = sim.max(axis=1) >= threshold
    n_grounded = int(grounded_mask.sum())
    score = round(float(grounded_mask.mean()), 4)

    return {"grounding_score": score, "n_grounded": n_grounded, "n_total": len(gen_points)}


def compute_overlap(
    gen_points: list[str],
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """
    Overlap / redundancy: mean pairwise cosine similarity among generated points.

    A lower score indicates more diverse critique (desirable).
    Returns None when there are fewer than 2 points.
    """
    if len(gen_points) < 2:
        return {"overlap_score": None, "n_points": len(gen_points)}

    embs = embed_normalize(gen_points, embedder)
    sim_matrix = embs @ embs.T                     # (n, n)
    n = len(gen_points)
    upper_idx = np.triu_indices(n, k=1)            # upper triangle, no diagonal
    pairwise = sim_matrix[upper_idx]
    score = round(float(pairwise.mean()), 4)

    return {"overlap_score": score, "n_points": n}


# ---------------------------------------------------------------------------
# Judge score loading
# ---------------------------------------------------------------------------


def load_judge_scores(results_dir: Path) -> dict[str, dict] | None:
    """
    Load per-paper judge scores from judge_scores.json if it exists.
    Returns the per_paper dict, or None if the file is absent.
    No API calls are made.
    """
    path = results_dir / "judge_scores.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("per_paper", {})


# ---------------------------------------------------------------------------
# Per-paper orchestration
# ---------------------------------------------------------------------------


def score_paper(
    paper_id: str,
    result: dict,
    jsonl_papers: list[dict],
    paper_idx: int | None,
    judge_per_paper: dict | None,
    embedder: SentenceTransformer,
    threshold: float,
) -> dict[str, Any]:
    """Compute all five metrics for one paper and return a structured dict."""
    gen_points = extract_generated_points(result)
    latency = result.get("latency_seconds")

    # Look up the JSONL entry; gracefully degrade if not found
    if paper_idx is not None:
        jsonl_entry = jsonl_papers[paper_idx]
    else:
        jsonl_entry = {}

    review_refs = get_paper_reviews(jsonl_entry)
    body_text = jsonl_entry.get("body_text", "")

    coverage = compute_coverage(gen_points, review_refs, embedder, threshold)
    grounding = compute_grounding(gen_points, body_text, embedder, threshold)
    overlap = compute_overlap(gen_points, embedder)

    judge = (
        judge_per_paper.get(paper_id) if judge_per_paper else None
    )

    return {
        "paper_id": paper_id,
        "n_generated_points": len(gen_points),
        "coverage": coverage,
        "grounding": grounding,
        "overlap": overlap,
        "judge": judge,
        "latency_seconds": latency,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _safe_mean(values: list) -> float | None:
    """Mean of non-None values; returns None if list is empty after filtering."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(float(np.mean(vals)), 4)


def aggregate_scores(per_paper: dict[str, dict]) -> dict[str, Any]:
    """Compute aggregate statistics across all papers."""
    papers = list(per_paper.values())

    def _get(paper, *keys):
        obj = paper
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k)
            else:
                return None
        return obj

    return {
        "n_papers": len(papers),
        "mean_coverage_macro_precision": _safe_mean(
            [_get(p, "coverage", "macro_precision") for p in papers]
        ),
        "mean_coverage_macro_recall": _safe_mean(
            [_get(p, "coverage", "macro_recall") for p in papers]
        ),
        "mean_coverage_macro_f1": _safe_mean(
            [_get(p, "coverage", "macro_f1") for p in papers]
        ),
        "mean_grounding": _safe_mean(
            [_get(p, "grounding", "grounding_score") for p in papers]
        ),
        "mean_overlap": _safe_mean(
            [_get(p, "overlap", "overlap_score") for p in papers]
        ),
        "mean_latency_seconds": _safe_mean(
            [p.get("latency_seconds") for p in papers]
        ),
        "mean_judge_coverage": _safe_mean(
            [_get(p, "judge", "coverage") for p in papers]
        ),
        "mean_judge_grounding": _safe_mean(
            [_get(p, "judge", "grounding") for p in papers]
        ),
        "mean_judge_overall": _safe_mean(
            [_get(p, "judge", "overall") for p in papers]
        ),
    }


# ---------------------------------------------------------------------------
# CSV / plot outputs
# ---------------------------------------------------------------------------


def per_paper_to_df(per_paper: dict[str, dict]) -> pd.DataFrame:
    """Flatten the per-paper nested dict into a tidy DataFrame."""
    rows = []
    for pid, d in per_paper.items():
        cov = d.get("coverage", {})
        grn = d.get("grounding", {})
        ovl = d.get("overlap", {})
        jdg = d.get("judge") or {}
        rows.append(
            {
                "paper_id": pid,
                "n_generated_points": d.get("n_generated_points"),
                "coverage_macro_precision": cov.get("macro_precision"),
                "coverage_macro_recall": cov.get("macro_recall"),
                "coverage_macro_f1": cov.get("macro_f1"),
                "n_reviews_used": len(cov.get("per_review", {})),
                "grounding_score": grn.get("grounding_score"),
                "n_grounded": grn.get("n_grounded"),
                "overlap_score": ovl.get("overlap_score"),
                "judge_coverage": jdg.get("coverage"),
                "judge_grounding": jdg.get("grounding"),
                "judge_overall": jdg.get("overall"),
                "latency_seconds": d.get("latency_seconds"),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(
    baseline_agg: dict,
    agent_agg: dict,
    output_path: Path,
) -> None:
    """Grouped bar chart – key metrics, baseline vs. agent."""
    metrics = [
        ("Cov. Recall", "mean_coverage_macro_recall"),
        ("Cov. Precision", "mean_coverage_macro_precision"),
        ("Cov. F1", "mean_coverage_macro_f1"),
        ("Grounding", "mean_grounding"),
        ("Overlap", "mean_overlap"),
    ]
    labels = [m[0] for m in metrics]
    base_vals = [baseline_agg.get(m[1]) or 0.0 for m in metrics]
    agent_vals = [agent_agg.get(m[1]) or 0.0 for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, base_vals, width, label="Baseline", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, agent_vals, width, label="Agent", color="#DD8452")

    ax.set_ylabel("Score (0–1)")
    ax.set_title("Member 5 – Key Metrics: Baseline vs. Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_latency(
    baseline_agg: dict,
    agent_agg: dict,
    output_path: Path,
) -> None:
    """Bar chart comparing mean latency."""
    b_lat = baseline_agg.get("mean_latency_seconds") or 0.0
    a_lat = agent_agg.get("mean_latency_seconds") or 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Baseline", "Agent"],
        [b_lat, a_lat],
        color=["#4C72B0", "#DD8452"],
        width=0.5,
    )
    ax.set_ylabel("Mean Latency (seconds)")
    ax.set_title("Member 5 – Latency: Baseline vs. Agent")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_mode(
    mode: str,
    cfg: dict,
    jsonl_papers: list[dict],
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """
    Run Member 5 evaluation for one mode ('baseline' or 'agents').
    Returns full scores dict {per_paper, aggregate}.
    Writes results/<mode>/member5_scores.json and member5_per_paper.csv.
    """
    results_dir = ROOT / cfg["results"][f"{mode}_dir"]
    threshold: float = cfg["evaluation"]["similarity_threshold"]
    n_jsonl = len(jsonl_papers)

    print(f"\n{'=' * 60}")
    print(f"  MODE: {mode.upper()}")
    print(f"  Results dir : {results_dir}")
    print(f"  JSONL papers: {n_jsonl}")
    print(f"  Threshold   : {threshold}")
    print(f"{'=' * 60}")

    results = load_results(results_dir)
    if not results:
        print(f"  [WARN] No per-paper result files found in {results_dir}.")
        print("         Run the baseline/agent pipeline first, then re-run this script.")
        return {"per_paper": {}, "aggregate": {"n_papers": 0}}

    judge_per_paper = load_judge_scores(results_dir)
    print(
        f"  Judge scores : {'loaded for ' + str(len(judge_per_paper)) + ' papers' if judge_per_paper else 'absent (judge=null)'}"
    )

    per_paper: dict[str, dict] = {}
    for i, (paper_id, result) in enumerate(results.items(), start=1):
        paper_idx = paper_index_from_id(paper_id, n_jsonl)
        if paper_idx is None:
            print(
                f"  [{i}/{len(results)}] {paper_id}  "
                f"[WARN: cannot map to JSONL; coverage/grounding will be null]"
            )
        else:
            print(f"  [{i}/{len(results)}] {paper_id} (JSONL idx {paper_idx}) ... ", end="", flush=True)

        paper_data = score_paper(
            paper_id=paper_id,
            result=result,
            jsonl_papers=jsonl_papers,
            paper_idx=paper_idx,
            judge_per_paper=judge_per_paper,
            embedder=embedder,
            threshold=threshold,
        )
        per_paper[paper_id] = paper_data

        if paper_idx is not None:
            cov_f1 = paper_data["coverage"].get("macro_f1", "N/A")
            grnd = paper_data["grounding"].get("grounding_score", "N/A")
            ovlp = paper_data["overlap"].get("overlap_score", "N/A")
            n_rev = len(paper_data["coverage"].get("per_review", {}))
            print(f"cov_f1={cov_f1}, grounding={grnd}, overlap={ovlp}, n_reviews={n_rev}")

    aggregate = aggregate_scores(per_paper)

    # ── Persist JSON ────────────────────────────────────────────────────────
    scores = {"per_paper": per_paper, "aggregate": aggregate}
    out_json = results_dir / "member5_scores.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, default=str)
    print(f"\n  Saved: {out_json}")

    # ── Persist CSV ──────────────────────────────────────────────────────────
    df = per_paper_to_df(per_paper)
    out_csv = results_dir / "member5_per_paper.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # ── Print aggregate summary ──────────────────────────────────────────────
    print("\n  Aggregate:")
    for k, v in aggregate.items():
        print(f"    {k}: {v}")

    return scores


def run_both(
    cfg: dict,
    jsonl_papers: list[dict],
    embedder: SentenceTransformer,
) -> None:
    """Run both modes and generate comparison CSV + plots."""
    base_scores = run_mode("baseline", cfg, jsonl_papers, embedder)
    agent_scores = run_mode("agents", cfg, jsonl_papers, embedder)

    base_agg = base_scores.get("aggregate", {})
    agent_agg = agent_scores.get("aggregate", {})

    results_root = ROOT / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    # ── Comparison CSV ───────────────────────────────────────────────────────
    metric_keys = [
        "n_papers",
        "mean_coverage_macro_precision",
        "mean_coverage_macro_recall",
        "mean_coverage_macro_f1",
        "mean_grounding",
        "mean_overlap",
        "mean_latency_seconds",
        "mean_judge_coverage",
        "mean_judge_grounding",
        "mean_judge_overall",
    ]
    rows = [
        {"metric": k, "baseline": base_agg.get(k), "agents": agent_agg.get(k)}
        for k in metric_keys
    ]
    comp_df = pd.DataFrame(rows)
    comp_csv = results_root / "member5_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n  Saved: {comp_csv}")
    print(comp_df.to_string(index=False))

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_comparison(base_agg, agent_agg, results_root / "member5_comparison.png")
    plot_latency(base_agg, agent_agg, results_root / "member5_latency.png")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    valid_modes = ("baseline", "agents", "both")
    if len(sys.argv) < 2 or sys.argv[1] not in valid_modes:
        print(__doc__)
        print(f"\nUsage: python -m src.evaluation.member5_metrics [{' | '.join(valid_modes)}]")
        sys.exit(1)

    mode = sys.argv[1]
    print(f"Loading config.yaml ...")
    cfg = load_config()

    print(f"Loading sentence-transformer '{cfg['evaluation']['embedding_model']}' ...")
    embedder = get_embedder(cfg["evaluation"]["embedding_model"])

    jsonl_path = ROOT / cfg["data"]["jsonl_file"]
    print(f"Loading JSONL from {jsonl_path} ...")
    jsonl_papers = load_jsonl_papers(jsonl_path)
    print(f"  {len(jsonl_papers)} papers loaded.")

    if mode == "both":
        run_both(cfg, jsonl_papers, embedder)
    else:
        run_mode(mode, cfg, jsonl_papers, embedder)

    print("\nDone.")


if __name__ == "__main__":
    main()
