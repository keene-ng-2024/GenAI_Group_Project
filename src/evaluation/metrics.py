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


# ── Pretty labels for config keys ─────────────────────────────────────────────

_PLATFORM_LABELS: dict[str, str] = {
    "baseline":            "Baseline",
    "n8n_noloop":          "n8n (no-loop)",
    "n8n":                 "n8n (1-round)",
    "langgraph_none":      "LangGraph (none)",
    "langgraph_fixed":     "LangGraph (fixed)",
    "langgraph_dynamic":   "LangGraph (dynamic)",
    "vertexai":            "Vertex AI",
    "dify_single_critic":  "Dify (single)",
    "dify_dual_critic":    "Dify (dual)",
    "crewai_none":         "CrewAI (none)",
    "crewai_fixed":        "CrewAI (fixed)",
    "crewai_dynamic":      "CrewAI (dynamic)",
}

# Skip top-level umbrella dirs that have no scores.json of their own
_SKIP_KEYS = {"langgraph_dir", "dify_dir", "crewai_dir"}


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary_table(all_scores: dict[str, dict]) -> None:
    """Print a comparison table for all platforms.

    Args:
        all_scores: {platform_label: scores_dict} mapping.
    """
    col_w = 12
    metrics = [
        ("mean_precision", "Precision"),
        ("mean_recall",    "Recall"),
        ("mean_f1",        "F1"),
        ("mean_latency_seconds", "Latency (s)"),
    ]

    platform_labels = list(all_scores.keys())
    header_parts = [f"{'Metric':<18}"] + [f"{lbl:>{col_w}}" for lbl in platform_labels]
    header = " ".join(header_parts)
    sep = "=" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for metric_key, metric_label in metrics:
        row = f"  {metric_label:<16}"
        for lbl in platform_labels:
            val = all_scores[lbl].get("aggregate", {}).get(metric_key, float("nan"))
            row += f" {val:>{col_w}.4f}"
        print(row)

    print(sep + "\n")


# ── Bar chart ──────────────────────────────────────────────────────────────────

def plot_comparison(
    all_scores: dict[str, dict],
    output_path: str = "results/comparison.png",
) -> None:
    """Grouped bar chart of Precision / Recall / F1 across all platforms."""
    metric_keys = ["mean_precision", "mean_recall", "mean_f1"]
    metric_labels = ["Precision", "Recall", "F1"]

    platform_labels = list(all_scores.keys())
    n_platforms = len(platform_labels)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    total_width = 0.8
    width = total_width / n_platforms

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_platforms)]

    fig, ax = plt.subplots(figsize=(max(8, n_platforms * 2), 5))

    for i, (lbl, color) in enumerate(zip(platform_labels, colors)):
        agg = all_scores[lbl].get("aggregate", {})
        vals = [agg.get(k, 0) for k in metric_keys]
        offset = (i - n_platforms / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=lbl, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=6)

    ax.set_ylabel("Score")
    ax.set_title("Critique Coverage — All Platforms")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved → {output_path}")
    plt.close(fig)


# ── Latency chart ─────────────────────────────────────────────────────────────

def plot_latency(
    all_scores: dict[str, dict],
    output_path: str = "results/latency_comparison.png",
) -> None:
    """Horizontal bar chart of mean latency across all platforms."""
    labels, vals = [], []
    for lbl, scores in all_scores.items():
        lat = scores.get("aggregate", {}).get("mean_latency_seconds")
        if lat is not None:
            labels.append(lbl)
            vals.append(lat)

    if not vals:
        print("No latency data available — skipping latency plot.")
        return

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.5)))
    bars = ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Mean latency (seconds)")
    ax.set_title("Latency — All Platforms")
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

    # Discover all platforms that have a scores.json
    all_scores: dict[str, dict] = {}
    for key, path_str in cfg["results"].items():
        if not key.endswith("_dir") or key in _SKIP_KEYS:
            continue
        scores_file = Path(path_str) / "scores.json"
        if not scores_file.exists():
            continue
        platform_key = key.removesuffix("_dir")
        label = _PLATFORM_LABELS.get(platform_key, platform_key)
        with open(scores_file) as f:
            all_scores[label] = json.load(f)

    if not all_scores:
        print("No scores.json files found. Run scorer.py for each platform first.")
        raise SystemExit(1)

    print(f"Loaded scores for: {', '.join(all_scores)}")

    print_summary_table(all_scores)
    plot_comparison(all_scores)
    plot_latency(all_scores)

    # LLM-judge comparison (if judge_scores.json exist for baseline and at least one other)
    baseline_label = _PLATFORM_LABELS.get("baseline", "Baseline")
    baseline_judge_file = Path(cfg["results"]["baseline_dir"]) / "judge_scores.json"

    if baseline_judge_file.exists():
        with open(baseline_judge_file) as f:
            baseline_judge = json.load(f)
        for key, path_str in cfg["results"].items():
            if not key.endswith("_dir") or key in _SKIP_KEYS or key == "baseline_dir":
                continue
            judge_file = Path(path_str) / "judge_scores.json"
            if judge_file.exists():
                platform_key = key.removesuffix("_dir")
                label = _PLATFORM_LABELS.get(platform_key, platform_key)
                with open(judge_file) as f:
                    agent_judge = json.load(f)
                out = f"results/judge_{platform_key}.png"
                plot_judge_comparison(baseline_judge, agent_judge, output_path=out)
