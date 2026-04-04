# Paper Critique Agent Study

A comparative study of **one-shot LLM prompting** versus **multi-agent agentic
workflows** for generating peer-review-style critiques of ML/AI research papers.

Each team member implements the same four-agent critique pipeline on a different
orchestration platform, then all systems are evaluated against the same
ground-truth critique points distilled from real human reviews.

---

## Research question

> Can a multi-agent debate loop (Reader → Critic ↔ Auditor → Summariser)
> produce higher-quality paper critiques than a single LLM call — and does the
> choice of orchestration platform matter?

---

## Platforms compared

| Platform | Branch | Runner | Models |
|----------|--------|--------|--------|
| **Baseline** (one-shot) | `main` | `src/baseline/baseline_critique.py` | Claude Sonnet |
| **Anthropic agents** (raw API) | `main` | `src/agents/orchestrator.py` | Claude Sonnet / Haiku |
| **n8n** | `main` | `src/platforms/n8n_critique.py` | GPT-4o via OpenRouter |
| **Dify** | `Dify` | `src/dify/run_dify.py` | via Dify API |
| **LangGraph** | `langgraph` | `src/platforms/langgraph_critique.py` | GPT-4o / GPT-4o-mini |
| *(additional platforms)* | *(TBD)* | | |

Each platform implements the same workflow and outputs the same JSON schema
to `results/<platform>/`, enabling apples-to-apples comparison using a shared
evaluation pipeline.

---

## Agentic workflow (shared across platforms)

```
Paper text
    │
    ▼
┌─────────┐    summary     ┌─────────┐
│  Reader │ ─────────────► │  Critic │ ─── initial critique ──┐
└─────────┘                └─────────┘                        │
                                ▲  revised critique           │ audit feedback
                                │                             ▼
                                └────────────────────  ┌─────────┐
                                                       │ Auditor │
                                                       └─────────┘
                                                            │
                                                  (repeat up to N rounds)
                                                            │
                                                            ▼
                                                    ┌─────────────┐
                                                    │ Summariser  │
                                                    └─────────────┘
                                                            │
                                                            ▼
                                                   JSON critique dict
```

---

## Repository layout

```
paper-critique-agent-study/
├── config.yaml                  # model names, seeds, hyperparams
├── requirements.txt
├── .env.example                 # copy → .env and fill in API keys
│
├── data/
│   ├── raw/                     # original human review files
│   ├── papers/                  # downloaded PDFs
│   ├── processed/
│   │   ├── reviews_parsed.json  # parsed human reviews per paper
│   │   └── critique_dicts/      # ground truth: one JSON per paper
│   ├── dev_split.jsonl          # 5 papers for development
│   ├── eval_split.jsonl         # 15 papers for final evaluation
│   └── README.md                # dataset preparation details
│
├── src/
│   ├── data_processing/
│   │   ├── parse_reviews.py        # extract text from raw reviews
│   │   └── build_critique_dict.py  # distil reviews → unique points
│   ├── baseline/
│   │   └── baseline_critique.py    # single LLM call (one-shot)
│   ├── agents/
│   │   ├── orchestrator.py         # Anthropic API agentic loop
│   │   ├── agents.py               # role definitions
│   │   └── tools.py                # tools agents can invoke
│   ├── platforms/
│   │   ├── n8n_critique.py         # n8n webhook adapter
│   │   └── langgraph_critique.py   # LangGraph StateGraph workflow
│   └── evaluation/
│       ├── scorer.py               # embedding similarity (P/R/F1)
│       ├── llm_judge.py            # LLM-as-judge scoring
│       └── metrics.py              # comparison tables & plots
│
├── results/
│   ├── baseline/
│   ├── agents/
│   ├── n8n/
│   ├── langgraph/
│   └── dify/
│
├── notebooks/                   # interactive exploration
└── report/
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and add your API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
```

### 3. Add data

See [data/README.md](data/README.md) for dataset preparation and reviewer agreement statistics.

### 4. Run a platform

```bash
# One-shot baseline
python -m src.baseline.baseline_critique

# Anthropic multi-agent
python -m src.agents.orchestrator

# n8n (requires running n8n instance)
python -m src.platforms.n8n_critique

# LangGraph
python -m src.platforms.langgraph_critique
```

### 5. Evaluate

```bash
# Embedding-based scoring (precision / recall / F1)
python -m src.evaluation.scorer <platform>       # baseline | agents | n8n | langgraph

# LLM-as-judge scoring (coverage / specificity / grounding / overall)
python -m src.evaluation.llm_judge <platform>

# Print comparison tables & plots
python -m src.evaluation.metrics
```

---

## Evaluation

### Embedding similarity
Each generated critique point is embedded with `sentence-transformers`.
A ground-truth point is *covered* if at least one generated point has
cosine similarity ≥ threshold.

| Metric    | Definition                                      |
|-----------|-------------------------------------------------|
| Recall    | Fraction of GT points covered by the system     |
| Precision | Fraction of generated points that match a GT pt |
| F1        | Harmonic mean of precision and recall            |

### LLM-as-judge
An independent LLM scores each generated review on four dimensions (1–5):
coverage, specificity, grounding, and overall quality.

---

## Output schema (all platforms)

Every platform must produce per-paper JSON files matching this schema:

```json
{
  "paper_id": "paper_0001",
  "platform": "<platform name>",
  "model": "<primary model used>",
  "latency_seconds": 42.5,
  "structured": {
    "summary": "...",
    "strengths": [{"point": "...", "evidence": "..."}],
    "weaknesses": [{"point": "...", "evidence": "..."}],
    "questions": [{"question": "...", "motivation": "..."}],
    "scores": { "correctness": 4, "novelty": 3, "recommendation": "...", "confidence": 3 }
  },
  "critique_points": { "point_001": "...", "point_002": "..." }
}
```

---

## Configuration

Key settings in [config.yaml](config.yaml):

| Key | Default | Description |
|-----|---------|-------------|
| `agent.max_rounds` | `3` | Max Critic ↔ Auditor debate rounds |
| `evaluation.similarity_threshold` | `0.50` | Cosine sim for "covered" |
| `temperature` | `0.2` | Generation temperature |

Platform-specific model configuration lives under each platform's section
(`n8n`, `langgraph`, etc.) in `config.yaml`.
