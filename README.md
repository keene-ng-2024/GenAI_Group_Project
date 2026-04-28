# Paper Critique Agent Study

This project compares a single-call LLM baseline against multi-agent critique
workflows for generating peer-review-style critiques of AI/ML papers.

The evaluation target is a deduplicated dictionary of critique points distilled
from real human reviews. Each system output is scored by semantic similarity
against those human-review-derived points.

## What Is Compared

The study keeps prompts, temperature, and evaluation logic aligned as much as
possible, then varies the orchestration platform and loop structure.

| Implementation | Loop mode | Main code / workflow | Model(s) | Input used |
| --- | --- | --- | --- | --- |
| Baseline | none | `src/baseline/baseline_critique.py` | `gpt-4.1-mini` | Parsed paper text |
| n8n | none | `src/platforms/n8n_workflow_noloop.json` | `gpt-4.1-mini` | Parsed paper text |
| n8n | fixed 1 round | `src/platforms/n8n_workflow.json` | `gpt-4.1-mini` | Parsed paper text |
| Dify | none | `src/dify/Paper CritiqueAgent(Single Critic).yml` | `gpt-4.1-mini` | Raw PDF upload |
| Dify | fixed 1 round | `src/dify/Paper CritiqueAgent (Dual Critic).yml` | `gpt-4.1-mini` | Raw PDF upload |
| LangGraph | none / fixed / dynamic | `src/platforms/langgraph_critique.py` | `gpt-4.1-mini` | Parsed paper text |
| CrewAI | none / fixed / dynamic | `src/platforms/crewai_critique.py` | `gpt-4.1-mini` | Parsed paper text |
| Vertex AI | none / fixed / dynamic | `src/vertex/vertex_orchestrator.py` | Gemini 2.5 Flash / Flash Lite | Parsed paper text |

Loop modes:

- `none`: Reader -> Critic -> Summariser.
- `fixed`: Reader -> Critic -> Auditor -> Critic revision -> Summariser.
- `dynamic`: Reader -> Critic <-> Auditor until the auditor is satisfied or
  `agent.max_rounds` is reached.

Important input note: Dify is the only implementation that uploads raw PDFs and
lets Dify parse them internally. Other implementations use the parsed paper text
from `data/processed/reviews_parsed.json`.

## Repository Layout

```text
GenAI_Group_Project/
|-- config.yaml
|-- requirements.txt
|-- .env.example
|-- data/
|   |-- ReviewCritique.jsonl
|   |-- checkpoint.json
|   |-- paper_links.csv
|   |-- direct_pdf_links.csv
|   |-- papers/
|   |-- processed/
|   |   |-- reviews_parsed.json
|   |   `-- critique_dicts/
|   `-- README.md
|-- scripts/
|   |-- run_full_pipeline.py
|   |-- score_vertexai.py
|   |-- patch_vertexai_results.py
|   `-- compare_scores.py
|-- src/
|   |-- baseline/
|   |-- data_processing/
|   |-- dify/
|   |-- evaluation/
|   |-- platforms/
|   `-- vertex/
|-- tests/
`-- results/
    |-- LLM_baseline/
    |-- n8n/
    |-- dify/
    |-- langgraph/
    |-- crewai/
    |-- vertexai/
    |-- vertexai_noloop/
    `-- vertexai_fixed/
```

Most output paths are configured in `config.yaml` under `results`. Prefer
changing paths there instead of hardcoding new paths in scripts.

## Quick Start

### 1. Create an environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install google-genai pytest
```

`google-genai` is needed for the Vertex AI runner. `pytest` is only needed for
the local test suite.

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in the required values.

PowerShell:

```powershell
Copy-Item .env.example .env
```

Bash:

```bash
cp .env.example .env
```

Required keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
DIFY_API_KEY_SINGLE=your_dify_single_critic_api_key_here
DIFY_API_KEY_DUAL=your_dify_dual_critic_api_key_here
```

Vertex AI also requires Google Cloud authentication for the project and location
set in `config.yaml`.

### 3. Prepare the data

Place the ReviewCritique dataset file at:

```text
data/ReviewCritique.jsonl
```

Then parse reviews and build the ground-truth critique dictionaries:

```bash
python -m src.data_processing.parse_reviews
python -m src.data_processing.build_critique_dict
```

For Dify, make sure PDFs exist in `data/papers/`. If the PDFs are missing, run
the helper scripts from the `data` directory:

```bash
cd data
python fetch_pdf_links.py
python download_papers.py
cd ..
```

## Platform Setup

### n8n

Start n8n locally with Docker:

```bash
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -e GENERIC_TIMEZONE="Asia/Singapore" \
  -e TZ="Asia/Singapore" \
  -e N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true \
  -e N8N_RUNNERS_ENABLED=true \
  -e OPENAI_API_KEY=your_openai_api_key \
  -e N8N_BLOCK_ENV_ACCESS_IN_NODE=false \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n
```

Import and activate both workflows:

1. Open `http://localhost:5678`.
2. Import `src/platforms/n8n_workflow_noloop.json`.
3. Import `src/platforms/n8n_workflow.json`.
4. Activate both workflows so the webhook URLs in `config.yaml` are live.

### Dify

Import the DSL files into a Dify workspace:

1. Create an app from `src/dify/Paper CritiqueAgent(Single Critic).yml`.
2. Create an app from `src/dify/Paper CritiqueAgent (Dual Critic).yml`.
3. Copy each workflow API key into `.env`.
4. Confirm the Dify LLM nodes use temperature `0.2`.

### Vertex AI

Set the project, location, models, and rate limits in `config.yaml` under
`vertex_ai`. Authenticate with Google Cloud before running the Vertex scripts.

## Running Systems

Baseline:

```bash
python -m src.baseline.baseline_critique
```

n8n:

```bash
python -m src.platforms.n8n_critique noloop
python -m src.platforms.n8n_critique 1round
```

Dify:

```bash
python -m src.dify.run_dify single_critic
python -m src.dify.run_dify dual_critic
```

LangGraph:

```bash
python -m src.platforms.langgraph_critique --mode none
python -m src.platforms.langgraph_critique --mode fixed --max-rounds 1
python -m src.platforms.langgraph_critique --mode dynamic
```

CrewAI:

```bash
python -m src.platforms.crewai_critique none
python -m src.platforms.crewai_critique fixed
python -m src.platforms.crewai_critique dynamic
```

Vertex AI:

```bash
python scripts/run_full_pipeline.py --mode noloop
python scripts/run_full_pipeline.py --mode fixed
python scripts/run_full_pipeline.py --mode dynamic
```

All runners skip papers whose output JSON already exists. Remove or move the
existing result file if you need to rerun a paper.

## Result Directories

| Score mode | Result directory |
| --- | --- |
| `baseline` | `results/LLM_baseline` |
| `n8n_noloop` | `results/n8n/n8n_noloop` |
| `n8n` | `results/n8n/n8n_1loop` |
| `dify_single_critic` | `results/dify/single_critic` |
| `dify_dual_critic` | `results/dify/dual_critic` |
| `langgraph_none` | `results/langgraph/no_loop` |
| `langgraph_fixed` | `results/langgraph/fixed_1round` |
| `langgraph_dynamic` | `results/langgraph/dynamic` |
| `crewai_none` | `results/crewai/crewai_none` |
| `crewai_fixed` | `results/crewai/crewai_fixed` |
| `crewai_dynamic` | `results/crewai/crewai_dynamic` |
| `vertexai_noloop` | `results/vertexai_noloop` |
| `vertexai_fixed` | `results/vertexai_fixed` |
| `vertexai` | `results/vertexai` |

For the scored variants above, the score mode is the `config.yaml` result key
without the `_dir` suffix. Umbrella directories such as `results/langgraph/`
are only containers for variant outputs.

## Scoring and Plots

Score any completed result directory:

```bash
python -m src.evaluation.scorer baseline
python -m src.evaluation.scorer n8n_noloop
python -m src.evaluation.scorer n8n
python -m src.evaluation.scorer dify_single_critic
python -m src.evaluation.scorer dify_dual_critic
python -m src.evaluation.scorer langgraph_none
python -m src.evaluation.scorer langgraph_fixed
python -m src.evaluation.scorer langgraph_dynamic
python -m src.evaluation.scorer crewai_none
python -m src.evaluation.scorer crewai_fixed
python -m src.evaluation.scorer crewai_dynamic
python -m src.evaluation.scorer vertexai_noloop
python -m src.evaluation.scorer vertexai_fixed
python -m src.evaluation.scorer vertexai
```

Create comparison tables and plots from available `scores.json` files:

```bash
python -m src.evaluation.metrics
```

This writes comparison figures under `results/`, including
`results/comparison.png` and `results/latency_comparison.png`.

## Evaluation Method

Each generated weakness point is embedded with the configured
`sentence-transformers` model. A ground-truth critique point is considered
covered when the maximum cosine similarity against generated points is at least
`evaluation.similarity_threshold` from `config.yaml` (default `0.50`).

| Metric | Meaning |
| --- | --- |
| Precision | Fraction of generated critique points that match a ground-truth point |
| Recall | Fraction of ground-truth points covered by generated critique points |
| F1 | Harmonic mean of precision and recall |

## Key Configuration

Common settings in `config.yaml`:

| Key | Default | Purpose |
| --- | --- | --- |
| `models.strong` | `gpt-4.1-mini` | Main OpenAI model for agents and dict building |
| `models.baseline` | `gpt-4.1-mini` | Single-call baseline model |
| `temperature` | `0.2` | Generation temperature |
| `agent.max_rounds` | `3` | Max debate rounds for dynamic loops |
| `agent.truncate_body_chars` | `0` | `0` means use full parsed paper text |
| `evaluation.similarity_threshold` | `0.50` | Cosine similarity threshold for coverage |
| `evaluation.embedding_model` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `langgraph.loop_mode` | `dynamic` | Default LangGraph mode when no CLI mode is passed |

## Tests

Run the available test suite with:

```bash
pytest
```

The current tests focus on Vertex AI result parsing and patching helpers.

## Notes and Limitations

- Dify uses raw PDF input, while other implementations use parsed paper text.
- Vertex AI uses Gemini model variants because that platform does not run the
  OpenAI model used elsewhere.
- Existing outputs are skipped by design so long runs can be resumed safely.
- Some helper scripts in `scripts/` are operational utilities; the main scoring
  path is `python -m src.evaluation.scorer <mode>` followed by
  `python -m src.evaluation.metrics`.
