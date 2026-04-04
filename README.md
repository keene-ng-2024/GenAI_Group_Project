# Paper Critique Agent Study

Comparing **single-call LLM baselines** against **multi-agent agentic loops**
for generating peer-review-style critiques of ML/AI papers, across six
orchestration platforms.

The ground truth is a deduplicated dictionary of critique points distilled
from real human reviews.  We measure how well each system *covers* those points
using semantic-similarity recall / precision.

---

## Platform Overview

Six platforms implement the same 4-agent pipeline under controlled conditions.
**Prompts are identical across all platforms.** The only dimensions that vary
are workflow structure and model.

| Platform | Workflow | Loop type | Model |
|----------|----------|-----------|-------|
| Baseline | Single call | None | GPT-4.1-mini |
| n8n | Visual DAG | Fixed rounds | GPT-4.1-mini |
| Dify | Visual DAG | Fixed rounds | GPT-4.1-mini |
| Vertex AI | Python (google-genai SDK) | Dynamic conditional | Gemini 2.5 Flash* |
| LangGraph | StateGraph (code-first) | Dynamic conditional | GPT-4.1-mini |
| CrewAI | Role-based sequential | Fixed rounds | GPT-4.1-mini |

*Vertex AI uses Gemini 2.5 Flash due to platform model lock-in.

**Loop types explained:**
- **None** — single LLM call, no debate
- **Fixed rounds** — Critic → Auditor unrolled N times (hardcoded, no early exit); used by platforms that cannot express conditional cycles (n8n, Dify, CrewAI)
- **Dynamic conditional** — Critic ↔ Auditor loop with early exit when Auditor is satisfied; native to LangGraph and implemented in Python for Vertex AI

---

## Fixed Constants

The following are held constant across all platforms to isolate workflow
structure as the independent variable.

### Agent prompts

| Agent | Role |
|-------|------|
| **Reader** | "You are a careful academic reader. Produce a structured summary with sections: Problem & Motivation, Proposed Method, Methods, Results, Claimed Contributions. Be factual and concise. Include specific numbers from experiments." |
| **Critic** | "You are a rigorous peer reviewer for a top-tier ML/AI venue. Identify substantive weaknesses in: novelty, methodology, evaluation, clarity, and reproducibility. For each point give: (a) the issue, (b) why it matters, (c) evidence from the paper." |
| **Auditor** | "You are a senior programme committee member auditing a peer review. Challenge poorly-supported critique points, ask for concrete evidence, flag over-interpretations, and identify issues the Critic missed. Be constructive but demanding." |
| **Summariser** | "You are a senior editor. Synthesise the Critic–Auditor debate into a final structured review. Output valid JSON only: summary, strengths, weaknesses, questions, scores." |

### Model assignment

All agents on all platforms use `gpt-4.1-mini` — best cost:performance ratio
for language tasks, and keeps the controlled experiment clean (one model,
one variable: workflow structure).

> Vertex AI is the only exception due to platform model lock-in (Gemini 2.5 Flash).
> This is treated as a platform constraint and noted as a limitation in the report.

---

## LangGraph Ablation

LangGraph additionally runs all three loop conditions with the same model and
prompts, isolating loop structure as the sole variable:

| Condition | Config | Description |
|-----------|--------|-------------|
| `loop_mode: none` | Reader → Critic → Summariser | No debate, single-pass |
| `loop_mode: fixed` | Reader → Critic → Auditor × N → Summariser | Fixed rounds, no early exit |
| `loop_mode: dynamic` | Reader → Critic ↔ Auditor → Summariser | Conditional early exit |

Results stored in `results/langgraph_none/`, `results/langgraph_fixed/`,
`results/langgraph_dynamic/`.

---

## Repository layout

```
paper-critique-agent-study/
├── config.yaml                  # model names, seeds, hyperparams
├── requirements.txt
├── .env.example                 # copy → .env and fill in API keys
│
├── data/
│   ├── raw/                     # original human review files (PDFs / JSONs)
│   ├── processed/
│   │   ├── reviews_parsed.json  # parsed human reviews per paper
│   │   └── critique_dicts/      # ground truth: one JSON per paper
│   └── README.md                # how to obtain / place data
│
├── src/
│   ├── data_processing/
│   │   ├── parse_reviews.py        # extract text from raw human reviews
│   │   └── build_critique_dict.py  # LLM call to distil reviews → unique points
│   ├── baseline/
│   │   └── baseline_critique.py    # single LLM call to critique a paper
│   ├── agents/
│   │   ├── orchestrator.py         # main agentic loop / workflow
│   │   ├── agents.py               # role definitions (Reader, Critic, Auditor, …)
│   │   └── tools.py                # tools agents can invoke
│   └── evaluation/
│       ├── scorer.py               # compare output vs ground truth → scores
│       └── metrics.py              # precision/recall, plots, summary tables
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_build_critique_dicts.ipynb
│   ├── 03_run_baseline.ipynb
│   ├── 04_run_agents.ipynb
│   └── 05_evaluation_results.ipynb
│
├── results/
│   ├── baseline/
│   └── agents/
│
└── report/
    └── final_report.pdf
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
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Add data

See [data/README.md](data/README.md) for how to obtain and place review files.

### 4. Run the pipeline

```bash
# Parse raw reviews
python -m src.data_processing.parse_reviews

# Build ground-truth critique dicts
python -m src.data_processing.build_critique_dict

# Run baseline (single LLM call per paper)
python -m src.baseline.baseline_critique

# Run agentic system (multi-agent loop)
python -m src.agents.orchestrator

# Score both systems
python -m src.evaluation.scorer baseline
python -m src.evaluation.scorer agents

# Print comparison table + plots
python -m src.evaluation.metrics
```

Or run everything interactively via the notebooks in order (01 → 05).

---

## Agentic workflow

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

## Evaluation

Each generated critique point is embedded with `sentence-transformers`.
A ground-truth point is considered *covered* if at least one generated point
has cosine similarity ≥ threshold (default 0.75).

| Metric    | Definition                                      |
|-----------|-------------------------------------------------|
| Recall    | Fraction of GT points covered by the system     |
| Precision | Fraction of generated points that match a GT pt |
| F1        | Harmonic mean of precision and recall           |

---

## Configuration

Key settings in [config.yaml](config.yaml):

| Key | Default | Description |
|-----|---------|-------------|
| `models.strong` | `claude-sonnet-4-6` | Agent / dict-builder model |
| `models.fast` | `claude-haiku-4-5-20251001` | Cheap sub-calls |
| `agent.max_rounds` | `5` | Max Critic ↔ Auditor debate rounds |
| `evaluation.similarity_threshold` | `0.75` | Cosine sim for "covered" |
| `temperature` | `0.2` | Generation temperature |
