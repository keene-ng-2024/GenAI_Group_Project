"""
scorer.py
---------
Compare LLM-generated critique points against the ground-truth critique dict.

A generated point "covers" a ground-truth point if their sentence-embedding
cosine similarity exceeds a threshold (default 0.75).

Returns per-paper and aggregate precision / recall / F1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Embedding helpers ──────────────────────────────────────────────────────────

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Core scoring ───────────────────────────────────────────────────────────────

def score_paper(
    generated: dict[str, str],
    ground_truth: dict[str, str],
    threshold: float,
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """
    Args:
        generated   : {point_id: text} from LLM output
        ground_truth: {point_id: text} from human-review distillation
        threshold   : cosine-similarity threshold for a "match"
        embedder    : sentence-transformer model

    Returns:
        dict with precision, recall, f1, matched pairs, etc.
    """
    gen_texts = list(generated.values())
    gt_texts = list(ground_truth.values())

    if not gen_texts or not gt_texts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_generated": len(gen_texts), "n_ground_truth": len(gt_texts),
                "covered_gt": [], "uncovered_gt": list(ground_truth.keys())}

    gen_embs = embedder.encode(gen_texts, normalize_embeddings=True)
    gt_embs = embedder.encode(gt_texts, normalize_embeddings=True)

    # similarity matrix: shape (n_gen, n_gt)
    sim_matrix = gen_embs @ gt_embs.T

    # Recall: which GT points are covered by at least one generated point?
    gt_covered = sim_matrix.max(axis=0) >= threshold
    recall = float(gt_covered.mean())

    # Precision: which generated points match at least one GT point?
    gen_relevant = sim_matrix.max(axis=1) >= threshold
    precision = float(gen_relevant.mean())

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    gt_keys = list(ground_truth.keys())
    covered_gt = [gt_keys[i] for i, c in enumerate(gt_covered) if c]
    uncovered_gt = [gt_keys[i] for i, c in enumerate(gt_covered) if not c]

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_generated": len(gen_texts),
        "n_ground_truth": len(gt_texts),
        "covered_gt": covered_gt,
        "uncovered_gt": uncovered_gt,
    }


# ── Batch scoring ──────────────────────────────────────────────────────────────

def score_all(
    results_dir: str,
    critique_dicts_dir: str,
    cfg: dict,
) -> dict[str, Any]:
    threshold = cfg["evaluation"]["similarity_threshold"]
    emb_model = cfg["evaluation"]["embedding_model"]
    embedder = get_embedder(emb_model)

    results_path = Path(results_dir)
    gt_path = Path(critique_dicts_dir)

    per_paper: dict[str, Any] = {}

    for result_file in sorted(results_path.glob("*.json")):
        paper_id = result_file.stem
        gt_file = gt_path / f"{paper_id}.json"

        if not gt_file.exists():
            print(f"  [SKIP] No ground-truth for {paper_id}")
            continue

        with open(result_file) as f:
            result = json.load(f)
        with open(gt_file) as f:
            ground_truth = json.load(f)

        generated = result.get("critique_points", {})
        scores = score_paper(generated, ground_truth, threshold, embedder)
        per_paper[paper_id] = scores
        print(f"  {paper_id}: P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}")

    if not per_paper:
        return {"per_paper": {}, "aggregate": {}}

    agg_precision = np.mean([v["precision"] for v in per_paper.values()])
    agg_recall = np.mean([v["recall"] for v in per_paper.values()])
    agg_f1 = np.mean([v["f1"] for v in per_paper.values()])

    aggregate = {
        "mean_precision": round(float(agg_precision), 4),
        "mean_recall": round(float(agg_recall), 4),
        "mean_f1": round(float(agg_f1), 4),
        "n_papers": len(per_paper),
    }

    print(f"\n  Aggregate: P={aggregate['mean_precision']:.3f}  "
          f"R={aggregate['mean_recall']:.3f}  F1={aggregate['mean_f1']:.3f}")

    return {"per_paper": per_paper, "aggregate": aggregate}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cfg = load_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    results_dir = (
        cfg["results"]["baseline_dir"] if mode == "baseline"
        else cfg["results"]["agents_dir"]
    )

    scores = score_all(
        results_dir=results_dir,
        critique_dicts_dir=cfg["data"]["critique_dicts_dir"],
        cfg=cfg,
    )

    out_file = Path(results_dir) / "scores.json"
    with open(out_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores saved → {out_file}")
