# Requirements Document: AI Research Paper Critique Assistant

## 1. Functional Requirements

### 1.1 Multi-Agent Orchestration

**FR-1.1.1**: The system MUST orchestrate four distinct agent roles: Reader, Critic, Auditor, and Summarizer.

**FR-1.1.2**: The system MUST use LangGraph for stateful orchestration of agent interactions.

**FR-1.1.3**: The system MUST maintain conversation history for each agent across multiple rounds.

**FR-1.1.4**: The system MUST support configurable agent models (Gemini 1.5 Flash for loops, Gemini 1.5 Pro for final generation).

### 1.2 Paper Processing

**FR-1.2.1**: The system MUST accept papers with the following fields: paper_id, title, abstract, full_text, and optional reviews.

**FR-1.2.2**: The system MUST truncate full text to a configurable maximum (default 12000 characters) to respect model context windows.

**FR-1.2.3**: The system MUST provide the Reader agent with the complete paper for section-by-section summarization.

### 1.3 Iterative Debate Loop

**FR-1.3.1**: The system MUST execute an iterative debate loop between Critic and Auditor agents.

**FR-1.3.2**: The system MUST cap the maximum number of debate rounds at 5.

**FR-1.3.3**: The system MUST allow early termination of the debate loop when the Auditor expresses satisfaction.

**FR-1.3.4**: The system MUST support configurable early-stop phrases (e.g., "no further concerns", "i am satisfied", "well-supported").

**FR-1.3.5**: The system MUST detect negation in early-stop phrases to prevent premature termination.

### 1.4 Grounding Verification

**FR-1.4.1**: The system MUST implement a Grounding Verifier to enforce evidence-based critique generation.

**FR-1.4.2**: The system MUST verify each critique point against relevant paper sections.

**FR-1.4.3**: The system MUST use low-cost LLM sub-calls for grounding verification to minimize latency.

**FR-1.4.4**: The system MUST return grounding scores including: is_supported (boolean), confidence (0-1), and evidence_match_score (0-1).

**FR-1.4.5**: The system MUST aggregate grounding scores across all critique points.

### 1.5 Output Generation

**FR-1.5.1**: The system MUST produce structured output matching this schema:
```json
{
  "summary": "string",
  "strengths": [{"point": "string", "evidence": "string"}],
  "weaknesses": [{"point": "string", "evidence": "string"}],
  "questions": [{"question": "string", "motivation": "string"}],
  "scores": {
    "correctness": "integer 1-5",
    "novelty": "integer 1-5",
    "recommendation": "string",
    "confidence": "integer 1-5"
  }
}
```

**FR-1.5.2**: The system MUST include a confidence flag for each critique point.

**FR-1.5.3**: The system MUST flatten structured output to a critique_points dictionary for evaluation compatibility.

**FR-1.5.4**: The system MUST include a complete transcript of all agent messages.

### 1.6 Metadata Logging

**FR-1.6.1**: The system MUST save outputs to `results/agents/<paper_id>.json`.

**FR-1.6.2**: The system MUST log run metadata including latency in milliseconds.

**FR-1.6.3**: The system MUST track token usage for each agent.

**FR-1.6.4**: The system MUST record the number of debate rounds executed.

### 1.7 Baseline Comparison

**FR-1.7.1**: The system MUST implement a single-call baseline that produces output in the same schema.

**FR-1.7.2**: The system MUST compare efficacy metrics between multi-agent and baseline approaches.

**FR-1.7.3**: The system MUST support evaluation using existing metrics (precision, recall, F1, coverage, specificity, grounding).

## 2. Non-Functional Requirements

### 2.1 Performance

**NFR-2.1.1**: Single agentic critique MUST complete within 60 seconds for typical papers.

**NFR-2.1.2**: Baseline critique MUST complete within 30 seconds.

**NFR-2.1.3**: Grounding verification per critique point MUST complete within 5 seconds.

**NFR-2.1.4**: The system MUST handle papers up to 12000 characters of full text.

### 2.2 Reliability

**NFR-2.2.1**: The system MUST retry failed LLM API calls with exponential backoff (max 3 attempts).

**NFR-2.2.2**: The system MUST gracefully handle JSON parsing failures with fallback mechanisms.

**NFR-2.2.3**: The system MUST continue processing when individual papers fail.

**NFR-2.2.4**: The system MUST log all errors for debugging and monitoring.

### 2.3 Maintainability

**NFR-2.3.1**: The system MUST separate agent logic from orchestration logic.

**NFR-2.3.2**: The system MUST use dataclasses for all data structures.

**NFR-2.3.3**: The system MUST provide comprehensive docstrings for all public functions.

**NFR-2.3.4**: The system MUST support configuration via YAML file.

### 2.4 Testability

**NFR-2.4.1**: The system MUST be unit testable with mock LLM responses.

**NFR-2.4.2**: The system MUST support property-based testing for invariants.

**NFR-2.4.3**: The system MUST provide test fixtures for sample papers.

**NFR-2.4.4**: The system MUST expose all key functions for direct testing.

## 3. Data Requirements

### 3.1 Input Data

**DR-3.1.1**: The system MUST accept papers in the following format:
```python
@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    full_text: str
    reviews: Optional[List[Dict[str, Any]]] = None
```

### 3.2 Output Data

**DR-3.2.1**: The system MUST produce output in the following format:
```python
@dataclass
class CritiqueResult:
    paper_id: str
    model: str
    rounds: int
    latency_seconds: float
    token_usage: Dict[str, int]
    transcript: List[AgentMessage]
    structured: StructuredReview
    critique_points: Dict[str, str]
    grounding_verifier_scores: Dict[str, float]
    run_metadata: Dict[str, Any]
```

### 3.3 Configuration Data

**DR-3.3.1**: The system MUST support configuration via YAML file with the following keys:
- `models`: strong, fast, baseline
- `agent`: max_rounds, truncate_body_chars, use_tools, max_tool_calls, early_stop_phrases, model_map
- `data`: raw_dir, jsonl_file, processed_dir, reviews_file, critique_dicts_dir
- `critique_dict`: min_points, max_points, similarity_threshold
- `evaluation`: similarity_threshold, embedding_model
- `results`: baseline_dir, agents_dir

## 4. Integration Requirements

### 4.1 LLM Integration

**IR-4.1.1**: The system MUST integrate with Google Cloud Vertex AI.

**IR-4.1.2**: The system MUST support Gemini 1.5 Flash for iterative rounds.

**IR-4.1.3**: The system MUST support Gemini 1.5 Pro for final generation.

**IR-4.1.4**: The system MUST handle API rate limiting and backoff.

### 4.2 Evaluation Integration

**IR-4.2.1**: The system MUST be compatible with existing evaluation metrics (precision, recall, F1).

**IR-4.2.2**: The system MUST support LLM-as-judge evaluation (coverage, specificity, grounding, overall).

**IR-4.2.3**: The system MUST produce output compatible with existing scorer.py format.

### 4.3 Baseline Integration

**IR-4.3.1**: The system MUST produce output compatible with baseline_critique.py format.

**IR-4.3.2**: The system MUST support direct comparison with baseline results.

## 5. Constraints

### 5.1 Technical Constraints

**C-5.1.1**: The system MUST use Python 3.9+.

**C-5.1.2**: The system MUST use Pydantic for data validation.

**C-5.1.3**: The system MUST use LangGraph for stateful orchestration.

**C-5.1.4**: The system MUST use Google Cloud Vertex AI for LLM calls.

### 5.2 Resource Constraints

**C-5.2.1**: The system MUST limit total token usage per paper to avoid excessive costs.

**C-5.2.2**: The system MUST implement circuit breakers for LLM API calls.

**C-5.2.3**: The system MUST queue requests when rate limits are approached.

### 5.3 Deployment Constraints

**C-5.3.1**: The system MUST run in the existing project structure under `src/`.

**C-5.3.2**: The system MUST use existing configuration in `config.yaml`.

**C-5.3.3**: The system MUST save outputs to existing directories (`results/agents/`).

## 6. Acceptance Criteria

### 6.1 Core Functionality

**AC-6.1.1**: Given a valid paper, the system produces a structured critique with summary, strengths, weaknesses, questions, and scores.

**AC-6.1.2**: Given a valid paper, the system executes at least one debate round between Critic and Auditor.

**AC-6.1.3**: Given a valid paper, the system verifies grounding for all critique points.

**AC-6.1.4**: Given a valid paper, the system saves output to `results/agents/<paper_id>.json`.

**AC-6.1.5**: Given a valid paper, the system logs latency in milliseconds to run_metadata.json.

### 6.2 Quality Requirements

**AC-6.2.1**: Given a valid paper, the system completes within 60 seconds.

**AC-6.2.2**: Given a valid paper, the system produces output compatible with existing evaluation metrics.

**AC-6.2.3**: Given a valid paper, the system handles errors gracefully without crashing.

**AC-6.2.4**: Given a valid paper, the system produces consistent output across multiple runs.

### 6.3 Comparison Requirements

**AC-6.3.1**: Given a valid paper, the multi-agent approach produces output with higher coverage than baseline.

**AC-6.3.2**: Given a valid paper, the multi-agent approach produces output with higher specificity than baseline.

**AC-6.3.3**: Given a valid paper, the multi-agent approach produces output with higher grounding than baseline.

**AC-6.3.4**: Given a valid paper, the latency difference between multi-agent and baseline is acceptable (< 2x).

## 7. Requirements Traceability

| Requirement | Design Section | Implementation Task |
|-------------|----------------|---------------------|
| FR-1.1.1 | Section 1.1 | Task 1.1 |
| FR-1.1.2 | Section 1.1 | Task 1.2 |
| FR-1.1.3 | Section 1.1 | Task 1.3 |
| FR-1.1.4 | Section 1.1 | Task 1.4 |
| FR-1.2.1 | Section 1.2 | Task 2.1 |
| FR-1.2.2 | Section 1.2 | Task 2.2 |
| FR-1.2.3 | Section 1.2 | Task 2.3 |
| FR-1.3.1 | Section 1.3 | Task 3.1 |
| FR-1.3.2 | Section 1.3 | Task 3.2 |
| FR-1.3.3 | Section 1.3 | Task 3.3 |
| FR-1.3.4 | Section 1.3 | Task 3.4 |
| FR-1.3.5 | Section 1.3 | Task 3.5 |
| FR-1.4.1 | Section 1.4 | Task 4.1 |
| FR-1.4.2 | Section 1.4 | Task 4.2 |
| FR-1.4.3 | Section 1.4 | Task 4.3 |
| FR-1.4.4 | Section 1.4 | Task 4.4 |
| FR-1.4.5 | Section 1.4 | Task 4.5 |
| FR-1.5.1 | Section 1.5 | Task 5.1 |
| FR-1.5.2 | Section 1.5 | Task 5.2 |
| FR-1.5.3 | Section 1.5 | Task 5.3 |
| FR-1.5.4 | Section 1.5 | Task 5.4 |
| FR-1.6.1 | Section 1.6 | Task 6.1 |
| FR-1.6.2 | Section 1.6 | Task 6.2 |
| FR-1.6.3 | Section 1.6 | Task 6.3 |
| FR-1.6.4 | Section 1.6 | Task 6.4 |
| FR-1.7.1 | Section 1.7 | Task 7.1 |
| FR-1.7.2 | Section 1.7 | Task 7.2 |
| FR-1.7.3 | Section 1.7 | Task 7.3 |
| NFR-2.1.1 | Section 1.1 | Task 1.5 |
| NFR-2.1.2 | Section 1.7 | Task 7.4 |
| NFR-2.2.1 | Section 1.4 | Task 4.6 |
| NFR-2.3.1 | Section 1.1 | Task 1.6 |
| NFR-2.4.1 | Section 1.1 | Task 1.7 |
| IR-4.1.1 | Section 1.1 | Task 1.8 |
| IR-4.2.1 | Section 1.7 | Task 7.5 |
| C-5.1.1 | Section 1.1 | Task 1.9 |
| C-5.2.1 | Section 1.4 | Task 4.7 |
| AC-6.1.1 | Section 1.5 | Task 5.5 |
| AC-6.2.1 | Section 1.3 | Task 3.6 |
| AC-6.3.1 | Section 1.7 | Task 7.6 |