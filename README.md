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
are workflow structure, input format, and model (Vertex AI only).

| Platform | Workflow file(s) | Loop type | Model | Input |
|----------|-----------------|-----------|-------|-------|
| Baseline | `src/baseline/baseline_critique.py` | None | GPT-4.1-mini | JSONL body_text |
| n8n (no loop) | `src/platforms/n8n_workflow_noloop.json` | None | GPT-4.1-mini | JSONL body_text |
| n8n (1 round) | `src/platforms/n8n_workflow.json` | Fixed 1 round | GPT-4.1-mini | JSONL body_text |
| Dify (no loop) | Dify workflow (`single_critic`) | None | GPT-4.1-mini | Raw PDF |
| Dify (1 round) | Dify workflow (`dual_critic`) | Fixed 1 round | GPT-4.1-mini | Raw PDF |
| Vertex AI | `src/agents/vertex_orchestrator.py` | Dynamic conditional | Gemini 2.5 Flash* | JSONL body_text |
| LangGraph (none) | `src/platforms/langgraph_critique.py` | None | GPT-4.1-mini | JSONL body_text |
| LangGraph (fixed) | `src/platforms/langgraph_critique.py` | Fixed 1 round | GPT-4.1-mini | JSONL body_text |
| LangGraph (dynamic) | `src/platforms/langgraph_critique.py` | Dynamic conditional | GPT-4.1-mini | JSONL body_text |

*Vertex AI uses Gemini 2.5 Flash due to platform model lock-in.

> **Input format note:** Dify ingests raw PDFs via its file upload API and handles parsing internally.
> All other platforms receive `body_text` from the JSONL dataset via the Python adapters.
> This difference is a platform constraint, not a controlled variable, and is noted as a limitation
> when comparing Dify scores against other platforms.

**Loop types explained:**
- **None** — Reader → Critic → Summariser. No debate, no Auditor.
- **Fixed 1 round** — Reader → Critic 1 → Auditor → Critic 2 → Summariser. Auditor challenges Critic 1, Critic 2 revises based on feedback, Summariser consolidates. Hardcoded, no early exit. Used by platforms that cannot express conditional cycles (n8n, Dify).
- **Dynamic conditional** — same agents but Critic ↔ Auditor loop repeats until Auditor is satisfied or max rounds reached. Native to LangGraph, implemented in Python for Vertex AI.

> **Note:** The fixed-round design requires an explicit Critic 2 node — a second Critic call that receives the Auditor's feedback and revises accordingly. Passing Auditor feedback directly to the Summariser (skipping Critic 2) is a weaker design as the critique points never get revised.

---

## Fixed Constants

The following are held constant across all platforms to isolate workflow
structure as the independent variable.

### Agent prompts

All prompts are identical across every platform. Variables in `{curly braces}` are filled in at runtime.

---

**Reader**

*System:*
```
You are a Reader agent. Read the following paper and produce a structured summary. Cover:
Problem & Motivation, Proposed Method, Results, Claimed Contributions.
```

*User:*
```
Paper:
{paper_text}
```

---

**Critic — Round 1 (initial)**

*System:*
```
You are a Critic agent reviewing an AI/ML research paper.
```

*User:*
```
Paper summary:
{summary}

Generate 12-15 specific critique points.

For each point you MUST:
- Be concrete and specific, not generic
- Reference specific sections, tables, or claims from the paper
- Focus on ONE issue per point

Cover ALL of these dimensions:
- Novelty: what prior work is missing or inadequately compared?
- Methodology: are there hidden assumptions, missing ablations, or design choices not justified?
- Evaluation: are baselines fair? are comparisons apples-to-apples? are metrics sufficient?
- Reproducibility: what implementation details are missing?
- Clarity: what is confusing or poorly explained in the paper?
- Limitations: what does the method fail to address or acknowledge?
- Generalisability: does it work beyond the tested settings?

IMPORTANT: Only critique what is actually in the paper.
Do NOT invent references, section numbers, or claims not explicitly stated.
```

---

**Auditor**

*System:*
```
You are an Auditor agent. Your job is to make the Critic's review stronger.
```

*User:*
```
Paper summary:
{summary}

Critic response:
{critique}

For each critique point:
1. Is it specific enough or too generic? Push for concrete details.
2. Is it supported by evidence from the paper?
3. What important issues did the Critic completely miss?

Be aggressive — a weak vague point is worse than no point.
Explicitly list 3-5 issues the Critic missed.

Do NOT suggest ethical implications or bias points unless
the paper explicitly makes claims in these areas.
```

---

**Critic — Round 2 (revision)**

*System:*
```
You are the Critic agent revising your review based on Auditor feedback.
```

*User:*
```
Original paper summary:
{summary}

Your original critique:
{critique}

Auditor feedback:
{audit_feedback}

Now produce an improved critique that:
- Fixes all weak or vague points the Auditor flagged
- Adds the missing issues the Auditor identified
- Keeps all strong original points
- Generates 12-15 total points

For each point you MUST:
- Be concrete and specific, not generic
- Reference specific sections, tables, or claims from the paper
- Focus on ONE issue per point

Cover ALL of these dimensions:
- Novelty: what prior work is missing or inadequately compared?
- Methodology: are there hidden assumptions, missing ablations, or design choices not justified?
- Evaluation: are baselines fair? are comparisons apples-to-apples? are metrics sufficient?
- Reproducibility: what implementation details are missing?
- Clarity: what is confusing or poorly explained in the paper?
- Limitations: what does the method fail to address or acknowledge?
- Generalisability: does it work beyond the tested settings?
```

---

**Summariser**

*System:*
```
You are a Summariser agent. Consolidate the critique into a final structured review.
Output ONLY valid JSON, no other text, no markdown code fences.
```

*User:*
```
Critic2 output:
{critic2_output}

Reader summary:
{summary}

Output in this exact JSON format:
{
  "summary": "2-3 sentence paper summary",
  "strengths": [
    {"point": "strength point", "evidence": "evidence from paper"},
    {"point": "strength point", "evidence": "evidence from paper"}
  ],
  "weaknesses": [
    {"point": "weakness point", "evidence": "evidence from paper"},
    {"point": "weakness point", "evidence": "evidence from paper"}
  ],
  "questions": [
    {"question": "open question", "motivation": "why this matters"},
    {"question": "open question", "motivation": "why this matters"}
  ]
}
```

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
# Edit .env and add OPENAI_API_KEY, DIFY_API_KEY, and any other required keys
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

# Start n8n locally via Docker (first time sets up persistent volume)
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -e GENERIC_TIMEZONE="Asia/Singapore" \
  -e TZ="Asia/Singapore" \
  -e N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true \
  -e N8n_RUNNERS_ENABLED=true \
  -e OPENAI_API_KEY=your_openai_api_key \
  -e N8N_BLOCK_ENV_ACCESS_IN_NODE=false \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n

# Import workflows into n8n (do this once after starting n8n):
# 1. Open http://localhost:5678 in your browser
# 2. Go to Workflows → Add Workflow → Import from file
# 3. Import src/platforms/n8n_workflow.json          (1-round debate)
# 4. Import src/platforms/n8n_workflow_noloop.json   (no loop)
# 5. Open each workflow, click Activate (toggle top-right) to publish the webhook

# Run n8n workflows (requires n8n running locally at localhost:5678)
python -m src.platforms.n8n_critique noloop   # Reader → Critic → Summariser
python -m src.platforms.n8n_critique 1round   # Reader → Critic 1 → Auditor → Critic 2 → Summariser

# Run Dify workflows (requires DIFY_API_KEY in .env)
python -m src.dify.run_dify                   # runs single_critic workflow by default

# Score all systems
python -m src.evaluation.scorer baseline
python -m src.evaluation.scorer agents
python -m src.evaluation.scorer n8n
python -m src.evaluation.scorer n8n_noloop
python -m src.evaluation.scorer dify

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
has cosine similarity ≥ threshold (default **0.50**, set in `config.yaml`).

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
| `models.strong` | `gpt-4.1-mini` | Agent / dict-builder model |
| `models.fast` | `gpt-4.1-mini` | Cheap sub-calls |
| `agent.max_rounds` | `3` | Max Critic ↔ Auditor debate rounds |
| `langgraph.loop_mode` | `dynamic` | `none` / `fixed` / `dynamic` |
| `evaluation.similarity_threshold` | `0.50` | Cosine sim for "covered" |
| `temperature` | `0.2` | Generation temperature |
