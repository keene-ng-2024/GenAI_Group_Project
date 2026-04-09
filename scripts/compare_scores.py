import json, glob, os

# Explicit mapping of display name -> scores.json path
PLATFORMS = {
    "baseline":          "results/baseline/scores.json",
    "n8n (no loop)":     "results/n8n_noloop/scores.json",
    "n8n (1 round)":     "results/n8n/scores.json",
    "dify (no loop)":    "results/dify/single_critic/scores.json",
    "dify (1 round)":    "results/dify/dual_critic/scores.json",
    "langgraph_none":    "results/langgraph_none/scores.json",
    "langgraph_fixed":   "results/langgraph_fixed/scores.json",
    "langgraph_dynamic": "results/langgraph_dynamic/scores.json",
    "crewai_none":       "results/crewai_none/scores.json",
    "crewai_fixed":      "results/crewai_fixed/scores.json",
    "crewai_dynamic":    "results/crewai_dynamic/scores.json",
    "vertexai":          "results/vertexai/scores.json",
}

rows = []
for name, path in PLATFORMS.items():
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    agg = data.get('aggregate', data.get('summary', data.get('overall', {})))
    per_paper = data.get('per_paper', {})
    zero_papers = sum(1 for v in per_paper.values() if v.get('f1', -1) == 0.0)
    total = len(per_paper)
    rows.append({
        'platform': name,
        'precision': agg.get('mean_precision'),
        'recall': agg.get('mean_recall'),
        'f1': agg.get('mean_f1'),
        'latency': agg.get('mean_latency_seconds'),
        'zero_papers': zero_papers,
        'total': total,
    })

rows.sort(key=lambda x: x['f1'] if x['f1'] is not None else 0, reverse=True)

def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "N/A"

print(f"\n{'Platform':<24} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Latency(s)':>12} {'Zero-F1':>10}")
print('-' * 78)
for r in rows:
    zp = f"{r['zero_papers']}/{r['total']}" if r['total'] else "N/A"
    print(f"{r['platform']:<24} {fmt(r['precision']):>10} {fmt(r['recall']):>8} {fmt(r['f1']):>8} {fmt(r['latency'],1):>12} {zp:>10}")
