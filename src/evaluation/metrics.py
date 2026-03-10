"""
metrics.py
----------
Higher-level metrics and visualisation utilities built on top of scorer.py.

Functions
  coverage_curve      – precision/recall at different similarity thresholds
  print_summary_table – formatted table comparing baseline vs. agent scores
  plot_comparison     – bar chart saved to a file
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation.scorer import get_embedder, score_all, score_paper


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Coverage curve ─────────────────────────────────────────────────────────────

def coverage_curve(
    results_dir: str,
    critique_dicts_dir: str,
    cfg: dict,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute mean precision/recall/F1 across a range of similarity thresholds.
    Returns a DataFrame with columns [threshold, precision, recall, f1].
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    emb_model = cfg["evaluation"]["embedding_model"]
    embedder = get_embedder(emb_model)

    results_path = Path(results_dir)
    gt_path = Path(critique_dicts_dir)

    # Pre-load all matched paper pairs
    paper_pairs: list[tuple[dict, dict]] = []
    for result_file in sorted(results_path.glob("*.json")):
        if result_file.stem in ("scores", "judge_scores"):
            continue
        paper_id = result_file.stem
        gt_file = gt_path / f"{paper_id}.json"
        if not gt_file.exists():
            continue
        with open(result_file) as f:
            result = json.load(f)
        with open(gt_file) as f:
            ground_truth = json.load(f)
        paper_pairs.append((result.get("critique_points", {}), ground_truth))

    rows = []
    for threshold in thresholds:
        precisions, recalls, f1s = [], [], []
        for generated, gt in paper_pairs:
            s = score_paper(generated, gt, threshold, embedder)
            precisions.append(s["precision"])
            recalls.append(s["recall"])
            f1s.append(s["f1"])
        rows.append({
            "threshold": threshold,
            "precision": round(float(np.mean(precisions)), 4),
            "recall": round(float(np.mean(recalls)), 4),
            "f1": round(float(np.mean(f1s)), 4),
        })

    return pd.DataFrame(rows)


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary_table(baseline_scores: dict, agent_scores: dict) -> None:
    b = baseline_scores.get("aggregate", {})
    a = agent_scores.get("aggregate", {})

    header = f"{'Metric':<18} {'Baseline':>10} {'Agent':>10} {'Δ':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for metric, label in [
        ("mean_precision", "Precision"),
        ("mean_recall", "Recall"),
        ("mean_f1", "F1"),
        ("mean_latency_seconds", "Latency (s)"),
    ]:
        bv = b.get(metric, float("nan"))
        av = a.get(metric, float("nan"))
        delta = av - bv if not (np.isnan(bv) or np.isnan(av)) else float("nan")
        sign = "+" if delta > 0 else ""
        print(f"  {label:<16} {bv:>10.4f} {av:>10.4f} {sign}{delta:>7.4f}")

    print("=" * len(header) + "\n")


# ── Bar chart ──────────────────────────────────────────────────────────────────

def plot_comparison(
    baseline_scores: dict,
    agent_scores: dict,
    output_path: str = "results/comparison.png",
) -> None:
    metrics = ["mean_precision", "mean_recall", "mean_f1"]
    labels = ["Precision", "Recall", "F1"]

    b_vals = [baseline_scores["aggregate"].get(m, 0) for m in metrics]
    a_vals = [agent_scores["aggregate"].get(m, 0) for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_b = ax.bar(x - width / 2, b_vals, width, label="Baseline", color="#4C72B0")
    bars_a = ax.bar(x + width / 2, a_vals, width, label="Agent", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Baseline vs. Agent: Critique Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars_b, fmt="%.3f", padding=3)
    ax.bar_label(bars_a, fmt="%.3f", padding=3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved → {output_path}")
    plt.close(fig)


# ── Latency chart ─────────────────────────────────────────────────────────────

def plot_latency(
    baseline_scores: dict,
    agent_scores: dict,
    output_path: str = "results/latency_comparison.png",
) -> None:
    """Bar chart comparing mean latency between baseline and agent."""
    b_lat = baseline_scores.get("aggregate", {}).get("mean_latency_seconds")
    a_lat = agent_scores.get("aggregate", {}).get("mean_latency_seconds")

    if b_lat is None and a_lat is None:
        print("No latency data available — skipping latency plot.")
        return

    labels, vals = [], []
    if b_lat is not None:
        labels.append("Baseline")
        vals.append(b_lat)
    if a_lat is not None:
        labels.append("Agent")
        vals.append(a_lat)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=["#4C72B0", "#DD8452"][:len(labels)], width=0.5)
    ax.set_ylabel("Mean latency (seconds)")
    ax.set_title("Latency Comparison")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Latency plot saved → {output_path}")
    plt.close(fig)


# ── LLM-judge chart ──────────────────────────────────────────────────────────

def plot_judge_comparison(
    baseline_judge: dict,
    agent_judge: dict,
    output_path: str = "results/judge_comparison.png",
) -> None:
    """Bar chart comparing LLM-as-judge scores across dimensions."""
    dims = ["mean_coverage", "mean_specificity", "mean_grounding", "mean_overall"]
    labels = ["Coverage", "Specificity", "Grounding", "Overall"]

    b_vals = [baseline_judge.get("aggregate", {}).get(d, 0) for d in dims]
    a_vals = [agent_judge.get("aggregate", {}).get(d, 0) for d in dims]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_b = ax.bar(x - width / 2, b_vals, width, label="Baseline", color="#4C72B0")
    bars_a = ax.bar(x + width / 2, a_vals, width, label="Agent", color="#DD8452")

    ax.set_ylabel("Score (1-5)")
    ax.set_title("LLM-as-Judge: Baseline vs. Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.bar_label(bars_b, fmt="%.2f", padding=3)
    ax.bar_label(bars_a, fmt="%.2f", padding=3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Judge comparison plot saved → {output_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()

    baseline_scores_file = Path(cfg["results"]["baseline_dir"]) / "scores.json"
    agent_scores_file = Path(cfg["results"]["agents_dir"]) / "scores.json"

    if not baseline_scores_file.exists() or not agent_scores_file.exists():
        print("Run scorer.py for both baseline and agents first.")
        raise SystemExit(1)

    with open(baseline_scores_file) as f:
        baseline_scores = json.load(f)
    with open(agent_scores_file) as f:
        agent_scores = json.load(f)

    print_summary_table(baseline_scores, agent_scores)
    plot_comparison(baseline_scores, agent_scores)
    plot_latency(baseline_scores, agent_scores)

    # LLM-judge comparison (if judge_scores.json files exist)
    baseline_judge_file = Path(cfg["results"]["baseline_dir"]) / "judge_scores.json"
    agent_judge_file = Path(cfg["results"]["agents_dir"]) / "judge_scores.json"

    if baseline_judge_file.exists() and agent_judge_file.exists():
        with open(baseline_judge_file) as f:
            baseline_judge = json.load(f)
        with open(agent_judge_file) as f:
            agent_judge = json.load(f)
        plot_judge_comparison(baseline_judge, agent_judge)
