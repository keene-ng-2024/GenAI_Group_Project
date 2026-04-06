# GenAI Final Group Project — Vertex AI Multi-Agent Paper Critique System

## Project Overview
SMU IS469 group project comparing agentic platforms for automated academic paper critique.
Bryan's delegation: **Vertex AI platform integration** (Role: Multi-Agent Pipeline Lead).

## Goal
Compare multiple agentic platforms building the same multi-agent paper critique pipeline:
- Reader → Critic → Auditor → Summariser (4-agent debate, up to 3 rounds)
- Evaluate each platform's output against human peer reviews from OpenReview

## Systems Being Compared

| System | Platform | Models | Status |
|--------|----------|--------|--------|
| Baseline | Custom Python (direct API) | GPT-4o | Done, F1=0.331 |
| n8n | n8n workflow automation | GPT-4o-mini / GPT-4o | Done, F1=0.382 |
| Vertex AI | Google Cloud Vertex AI | Gemini 2.5 Flash / Flash-Lite | **Complete (100/100), F1=0.342** |

## Key Architecture

```
src/
  agents/
    vertex_orchestrator.py    ← main pipeline: Reader→Critic→Auditor→Summariser debate
    vertex_client.py          ← google-genai SDK wrapper with rate limiting + circuit breaker
    grounding_verifier.py     ← verifies critique points are grounded in paper text
    personas.py               ← AgentRole enum + BaseAgent class
    state.py                  ← AgentState TypedDict + state management helpers

results/
  vertexai/
    paper_0001.json ... paper_0100.json   ← per-paper critique results
    scores.json                           ← aggregate P/R/F1 scores

config.yaml                   ← vertex_ai section: project, location, model names
```

## Model Configuration

| Agent Role | Model | Purpose |
|------------|-------|---------|
| Reader | `gemini-2.5-flash-lite` | Summarises paper into structured sections |
| Critic | `gemini-2.5-flash` | Generates substantive critique points |
| Auditor | `gemini-2.5-flash-lite` | Challenges weak points, requests evidence |
| Summariser | `gemini-2.5-flash` | Consolidates debate into final structured JSON |
| Grounding Verifier | `gemini-2.5-flash-lite` | Verifies critique points are paper-grounded |

`gemini-2.5-flash-lite` = cheapest GA model (cost-optimised for high-volume roles)
`gemini-2.5-flash` = balanced quality/cost (used for quality-critical roles)

## Vertex AI Setup

### Prerequisites
1. Google Cloud project with Vertex AI API enabled
2. Application Default Credentials (ADC) configured:
   ```bash
   gcloud auth application-default login
   ```
3. Install dependencies:
   ```bash
   pip install google-genai google-cloud-aiplatform
   ```

### Critical Configuration Notes
- **Location must be a real region** — `global` causes SDK to hang indefinitely. Use `us-central1`.
- **SDK choice** — use `google-genai` (`google.genai.Client(vertexai=True, ...)`), NOT the `vertexai` package directly. The `vertexai` package hangs on import due to gRPC initialisation on Windows.
- **Model names** — use short names like `gemini-2.5-flash`, not full resource paths.

### config.yaml
```yaml
vertex_ai:
  project: "your-gcp-project-id"
  location: "us-central1"
  reader_model: "gemini-2.5-flash-lite"
  critic_model: "gemini-2.5-flash"
  auditor_model: "gemini-2.5-flash-lite"
  summariser_model: "gemini-2.5-flash"
```

## Running the Pipeline

```bash
# Run full Vertex AI pipeline (all 100 papers, skips already-done)
python src/agents/vertex_orchestrator.py

# Score results
python src/evaluation/scorer.py vertexai

# Test connection
python -c "
import google.genai as genai
client = genai.Client(vertexai=True, project='your-project-id', location='us-central1')
r = client.models.generate_content(model='gemini-2.5-flash-lite', contents='Hello')
print(r.text)
"
```

## Pipeline Design

Each paper goes through a structured multi-agent debate:

```
1. Reader    → structured summary (Problem, Method, Results, Contributions)
2. Critic    → initial critique points (novelty, methodology, evaluation, clarity, reproducibility)
3. [Debate loop, up to 3 rounds]:
     Auditor → challenges weak points, requests evidence, flags missed issues
     Critic  → revises/defends points based on Auditor feedback
     (early stop if Auditor says "no further concerns" / "i am satisfied")
4. Summariser → final structured JSON (summary, strengths, weaknesses, questions, scores)
5. Grounding Verifier → checks each weakness point is supported by paper text
```

Output per paper is saved as JSON with:
- `critique_points` — flat dict of weakness points for evaluation
- `structured` — full review (summary, strengths, weaknesses, questions, scores)
- `transcript` — full agent conversation log
- `latency_seconds`, `token_usage`, `grounding_verifier_scores`, `run_metadata`

## Known Issues / Findings

- **Slow throughput** — ~4-5 min per paper due to multi-round debate + API latency. Full 100-paper run takes ~7-8 hours.
- **v1beta1 error rate ~43%** — visible in GCP dashboard. These are 429 rate-limit retries handled automatically by the SDK with exponential backoff. Not actual failures — the circuit breaker and rate limiter in `vertex_client.py` manage this.
- **Occasional structured-output extraction mismatches** — Summariser sometimes wraps JSON in extra markdown, and `_parse_structured_output()` recovers some of these cases. However, in the current committed results there are still papers where the transcript contains a JSON review but the saved `structured` and `critique_points` are empty. These should be treated as parse/persistence failures in the saved artifact, and affected papers should be rerun if exact field consistency is required.
- **Gemini 2.5 Flash thinking tokens** — `gemini-2.5-flash` uses internal reasoning tokens that count toward billing but aren't in the output. Actual cost is higher than token counts suggest.
- **Known affected outputs** — `paper_0043`, `paper_0044`, `paper_0059`, and `paper_0083` have transcript/saved-output mismatches; they should not be described as cleanly handled by the fallback.

## Final Results (100/100 papers)

| Metric | Baseline (GPT-4o) | n8n (GPT-4o) | Vertex AI (Gemini 2.5) |
|--------|-------------------|--------------|------------------------|
| Precision | 0.463 | 0.386 | **0.421** |
| Recall | 0.272 | 0.401 | 0.302 |
| F1 | 0.331 | 0.382 | 0.342 |
| Latency | ~9s | ~44s | ~214s |
| Papers | 100 | 99 | 100 |

Vertex AI achieves higher precision than n8n (0.421 vs 0.386) but remains below the baseline (0.463) — the multi-round debate produces fewer, more selective critique points. Recall is lower than n8n because the debate loop tends to consolidate points rather than enumerate many. The trade-off is significant latency (~214s vs ~9s baseline).

Vertex AI does not lead on precision or F1 overall, but it outperforms the baseline on F1 (0.342 vs 0.331) while trailing n8n (0.382). The multi-round debate produces fewer but higher-quality critique points compared to single-shot approaches.

## Report Framing

- This is a **platform comparison**: same 4-agent pipeline design, different execution environments (Python/OpenAI vs n8n/OpenAI vs Vertex AI/Gemini)
- Vertex AI adds **grounding verification** as an extra quality layer not present in other platforms
- The debate loop (Critic ↔ Auditor) is the key differentiator vs single-shot baseline
- Cross-model comparison (GPT-4o vs Gemini 2.5) is a secondary finding
- Methodology: "structured extraction from OpenReview peer reviews" for ground truth
