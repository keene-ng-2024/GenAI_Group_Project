# GenAI Final Group Project — Multi-Agent Paper Critique System

## Project Overview
SMU IS469 group project comparing agentic platforms for automated academic paper critique.
Ernest's delegation: **n8n platform integration** (Role 3: Multi-Agent Pipeline Lead).

## Goal
Compare multiple agentic platforms building the same multi-agent paper critique pipeline:
- Reader → Critic → Auditor → Summariser (4-agent debate)
- Evaluate each platform's output against human peer reviews from OpenReview

## Systems Being Compared

> **Note:** The scores and model assignments in this table are for one **n8n-specific experimental run/configuration** documented in this file. Other project documents (for example, `VERTEXAI.md`) may report different baseline/n8n models or scores for **different runs, dates, or configs** and should not be read as the same experiment unless explicitly stated.

| System | Platform | Models | Experimental run / status |
|--------|----------|--------|----------------------------|
| Baseline | Custom Python (direct API) | Claude Sonnet | n8n comparison run in this document; Done, F1≈0.497 |
| Agents | Custom Python (multi-agent) | Claude Sonnet + Haiku | n8n comparison run in this document; Scoring pending |
| n8n | n8n workflow automation | GPT-4o-mini (Reader/Auditor) + GPT-4o (Critic/Summariser) | n8n comparison run in this document; Done, F1=0.172 |
## Key Architecture

```
data/processed/
  reviews_parsed.json         ← 100 papers from OpenReview
  critique_dicts/             ← LLM-extracted ground-truth critique points

src/
  baseline/baseline_critique.py    ← single Claude Sonnet call
  agents/orchestrator.py           ← 4-agent Claude debate pipeline
  platforms/n8n_critique.py        ← Python adapter: POSTs papers to n8n webhook
  platforms/n8n_workflow.json      ← importable n8n workflow (6 nodes)
  evaluation/scorer.py             ← cosine similarity F1 scoring
  evaluation/llm_judge.py          ← LLM-based qualitative scoring

results/
  baseline/scores.json
  agents/scores.json
  n8n/scores.json
```

## Evaluation Method
- Ground truth: human peer reviews from OpenReview, LLM-extracted into atomic critique points
- Metric: Precision/Recall/F1 via cosine similarity (MiniLM embeddings, threshold=0.55)
- Limitation (acknowledge in report): critique dicts are LLM-distilled, not manually verified

## n8n Setup
- Docker: `docker run -p 5678:5678 -e OPENAI_API_KEY=... -e N8N_BLOCK_ENV_ACCESS_IN_NODE=false n8nio/n8n`
- Webhook: `http://localhost:5678/webhook/paper-critique` (production, requires workflow to be Active/Published)
- Key fix: Critic node uses `$('Receive Paper').first().json.paper_text` to access paper text directly from webhook (not via ctx, which drops large strings)

## Running the Pipeline
```bash
# Run a platform
python -m src.platforms.n8n_critique
python -m src.baseline.baseline_critique
python -m src.agents.orchestrator

# Score results
python -m src.evaluation.scorer baseline
python -m src.evaluation.scorer agents
python -m src.evaluation.scorer n8n
```

## Known Issues / Findings
- n8n recall is low (0.123) because it generates 3-5 weaknesses vs ground truth's 10-14
- n8n precision is reasonable (0.335) — points it generates are on-target
- ~30% of n8n papers score 0.000 (likely paper_text not reaching some nodes)
- Claude handles noisy PDF text better than GPT-4o in this task

## Report Framing
- This is a **platform comparison**, not just a model comparison
- Same pipeline design, different execution environments
- Cross-model comparison (Claude vs GPT-4o) is a secondary finding
- Methodology: "structured extraction from OpenReview peer reviews" for ground truth
