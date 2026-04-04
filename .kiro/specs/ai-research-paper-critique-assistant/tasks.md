# Implementation Tasks: AI Research Paper Critique Assistant

## Phase 1: Core Infrastructure

### 1.1 Agent Role Definitions

- [ ] 1.1.1 Create `src/agents/personas.py` with Vertex AI client initialization using `google-cloud-vertexai` (FR-1.1.4, IR-4.1.1)
- [ ] 1.1.2 Define `AgentRole` enum (READER, CRITIC, AUDITOR, SUMMARIZER) in `src/agents/personas.py` (FR-1.1.1)
- [ ] 1.1.3 Update `BaseAgent` class to support configurable models per role with Vertex AI integration (FR-1.1.4)
- [ ] 1.1.4 Add `AgentMessage` dataclass with role, content, timestamp, tool_calls, tool_results fields (FR-1.5.4)
- [ ] 1.1.5 Add `AgentRole` field to `BaseAgent` for role identification in transcript tracking (FR-1.1.3)

### 1.2 Data Structures

- [ ] 1.2.1 Create `src/agents/data_models.py` with core dataclasses (FR-1.2.1, DR-3.1.1)
- [ ] 1.2.2 Define `Paper` dataclass with paper_id, title, abstract, full_text, reviews fields (FR-1.2.1, DR-3.1.1)
- [ ] 1.2.3 Define `CritiquePoint` dataclass with point, evidence, low_confidence fields (FR-1.5.2)
- [ ] 1.2.4 Define `StructuredReview` dataclass with summary, strengths, weaknesses, questions, scores (FR-1.5.1)
- [ ] 1.2.5 Define `CritiqueResult` dataclass with all required fields including grounding_verifier_scores (FR-1.4.5, DR-3.2.1)

### 1.3 Configuration

- [ ] 1.3.1 Update `config.yaml` with Vertex AI model mappings (models.vertex_fast, models.vertex_pro) (FR-1.1.4)
- [ ] 1.3.2 Add `agent.vertex_ai` section with project and location configuration (IR-4.1.1)
- [ ] 1.3.3 Add `grounding_verifier` section with settings (model, max_tokens, early_stop_phrases) (FR-1.4.1)
- [ ] 1.3.4 Add `output` section with results_dir and metadata_file configuration (FR-1.6.1, FR-1.6.2)
- [ ] 1.3.5 Add `cost_control` section with max_tokens_per_paper and rate_limit settings (NFR-2.1.3)

## Phase 2: Grounding Verification

### 2.1 Grounding Verifier Implementation

- [ ] 2.1.1 Create `src/agents/grounding_verifier.py` with `verify_grounding()` function (FR-1.4.1, FR-1.4.2)
- [ ] 2.1.2 Implement `verify_grounding(critique_point, paper_section, config)` returning is_supported, confidence, evidence_match_score (FR-1.4.4)
- [ ] 2.1.3 Implement `verify_all_grounding(critique_text, paper, config)` for batch verification (FR-1.4.5)
- [ ] 2.1.4 Add grounding score calculation with LLM sub-call using Gemini 1.5 Flash (NFR-2.1.3)
- [ ] 2.1.5 Implement early stopping detection with negation handling (FR-1.3.5)

### 2.2 Vertex AI Integration

- [ ] 2.2.1 Install `google-cloud-vertexai>=1.143.0` and `langchain-google-vertexai>=3.2.2` packages (IR-4.1.1)
- [ ] 2.2.2 Create `src/agents/vertex_client.py` with `get_vertex_ai_client()` function (IR-4.1.1)
- [ ] 2.2.3 Implement `generate_content()` method with structured output support (FR-1.5.1)
- [ ] 2.2.4 Add retry logic with exponential backoff for 429 errors (NFR-2.2.1)
- [ ] 2.2.5 Add rate limiting and circuit breaker for API calls (NFR-2.2.1, C-5.2.2)

## Phase 3: Orchestration

### 3.1 Main Orchestrator

- [ ] 3.1.1 Create `src/agents/state.py` with `AgentState` TypedDict and state management (FR-1.1.2)
- [ ] 3.1.2 Implement `run_pipeline(paper_id, paper_text)` function as main entry point (IR-4.3.1)
- [ ] 3.1.3 Implement iterative debate loop with configurable max_rounds=5 (FR-1.3.2)
- [ ] 3.1.4 Implement early stopping detection with configurable phrases (FR-1.3.3, FR-1.3.4)
- [ ] 3.1.5 Integrate grounding verification into debate loop (FR-1.4.3)

### 3.2 Output Generation

- [ ] 3.2.1 Implement `parse_structured_output(raw_text)` with JSON schema validation (FR-1.5.1)
- [ ] 3.2.2 Implement `flatten_to_critique_points(structured)` for evaluation compatibility (FR-1.5.3)
- [ ] 3.2.3 Implement `build_debate_context(transcript, config)` for Summarizer input (FR-1.1.3)
- [ ] 3.2.4 Implement `save_result(result, output_dir)` to `results/agents/<paper_id>.json` (FR-1.6.1)
- [ ] 3.2.5 Implement `log_run_metadata(metadata, output_dir)` with latency_ms (FR-1.6.2)

### 3.3 Baseline Integration

- [ ] 3.3.1 Update `src/baseline/baseline_critique.py` to use Vertex AI (IR-4.1.1)
- [ ] 3.3.2 Ensure baseline output matches agentic output schema (FR-1.7.1)
- [ ] 3.3.3 Add compatibility layer for evaluation metrics (IR-4.2.1)
- [ ] 3.3.4 Implement `compare_with_baseline(agentic_result, baseline_result)` (FR-1.7.2)

## Phase 4: Testing

### 4.1 Unit Tests

- [ ] 4.1.1 Create `tests/test_agents.py` with agent role tests using mock LLM responses (NFR-2.4.1)
- [ ] 4.1.2 Create `tests/test_grounding_verifier.py` with grounding tests (NFR-2.4.1)
- [ ] 4.1.3 Create `tests/test_orchestrator.py` with orchestration tests (NFR-2.4.1)
- [ ] 4.1.4 Create `tests/test_data_models.py` with data model validation tests (NFR-2.4.1)
- [ ] 4.1.5 Add mock LLM responses for all agent roles (NFR-2.4.1)

### 4.2 Property-Based Tests

- [ ] 4.2.1 Create `tests/test_properties.py` with property-based tests (NFR-2.4.2)
- [ ] 4.2.2 Test transcript length invariant: `|transcript| = 3 + 2 * rounds` (Correctness Property 1)
- [ ] 4.2.3 Test latency consistency: `latency_ms = latency_seconds * 1000` (Correctness Property 2)
- [ ] 4.2.4 Test grounding score bounds: `0.0 <= score <= 1.0` (Correctness Property 3)
- [ ] 4.2.5 Test round count bounds: `0 <= rounds <= max_rounds` (Correctness Property 4)
- [ ] 4.2.6 Test output schema compliance with Pydantic validation (Correctness Property 5)

### 4.3 Integration Tests

- [ ] 4.3.1 Create `tests/test_integration.py` with end-to-end tests (NFR-2.4.3)
- [ ] 4.3.2 Test full pipeline on dev_split.jsonl (5 papers) (NFR-2.4.3)
- [ ] 4.3.3 Test baseline compatibility with evaluation metrics (IR-4.2.1)
- [ ] 4.3.4 Test grounding verification integration with actual LLM calls (FR-1.4.3)
- [ ] 4.3.5 Test multi-paper batch processing with metadata logging (FR-1.6.4)

## Phase 5: Documentation

### 5.1 Code Documentation

- [ ] 5.1.1 Add docstrings to all public functions in `src/agents/` (NFR-2.3.3)
- [ ] 5.1.2 Add type hints to all function signatures in data_models.py (NFR-2.3.2)
- [ ] 5.1.3 Add inline comments for complex logic in orchestrator.py (NFR-2.3.3)
- [ ] 5.1.4 Add error handling documentation in grounding_verifier.py (NFR-2.2.1)
- [ ] 5.1.5 Add usage examples in docstrings for public API (NFR-2.3.3)

### 5.2 User Documentation

- [ ] 5.2.1 Create `docs/ai-research-paper-critique-assistant.md` (NFR-2.3.4)
- [ ] 5.2.2 Document system architecture with state flow diagram (FR-1.1.2)
- [ ] 5.2.3 Document agent roles and responsibilities (FR-1.1.1)
- [ ] 5.2.4 Document configuration options in config.yaml (NFR-2.3.4)
- [ ] 5.2.5 Document output format with JSON schema (FR-1.5.1)

### 5.3 API Documentation

- [ ] 5.3.1 Document public API functions in `src/agents/` (NFR-2.3.1)
- [ ] 5.3.2 Document data structures with Pydantic models (NFR-2.3.2)
- [ ] 5.3.3 Document configuration schema with examples (NFR-2.3.4)
- [ ] 5.3.4 Document error codes and recovery procedures (NFR-2.2.2)
- [ ] 5.3.5 Document performance characteristics (NFR-2.1.1)

## Phase 6: Evaluation

### 6.1 Efficacy Metrics

- [ ] 6.1.1 Run evaluation on dev_split.jsonl (5 papers) using existing scorer.py (IR-4.2.1)
- [ ] 6.1.2 Compare multi-agent vs baseline coverage (IR-4.2.1)
- [ ] 6.1.3 Compare multi-agent vs baseline specificity (IR-4.2.2)
- [ ] 6.1.4 Compare multi-agent vs baseline grounding (IR-4.2.2)
- [ ] 6.1.5 Compare multi-agent vs baseline latency (NFR-2.1.1, NFR-2.1.2)

### 6.2 Performance Tuning

- [ ] 6.2.1 Profile latency bottlenecks using cProfile (NFR-2.1.1)
- [ ] 6.2.2 Optimize grounding verification with caching (NFR-2.1.3)
- [ ] 6.2.3 Optimize early stopping detection (FR-1.3.3)
- [ ] 6.2.4 Optimize token usage per paper (NFR-2.1.4, C-5.2.1)
- [ ] 6.2.5 Implement caching for repeated LLM calls (NFR-2.1.1)

## Phase 7: Deployment

### 7.1 Environment Setup

- [ ] 7.1.1 Set up Google Cloud Vertex AI credentials via `gcloud auth application-default login` (IR-4.1.1)
- [ ] 7.1.2 Configure project and location in config.yaml (IR-4.1.1)
- [ ] 7.1.3 Install required packages: `google-cloud-aiplatform>=1.143.0`, `langchain-google-vertexai>=3.2.2`, `langgraph>=1.2.6`, `pydantic>=2.0` (IR-4.1.1)
- [ ] 7.1.4 Set up environment variables: `GOOGLE_APPLICATION_CREDENTIALS` (IR-4.1.1)
- [ ] 7.1.5 Test Vertex AI connection with sample paper (IR-4.1.1)

### 7.2 CI/CD

- [ ] 7.2.1 Set up GitHub Actions workflow for testing (NFR-2.4.4)
- [ ] 7.2.2 Add unit test step with pytest (NFR-2.4.1)
- [ ] 7.2.3 Add integration test step (NFR-2.4.3)
- [ ] 7.2.4 Add performance test step (NFR-2.1.1)
- [ ] 7.2.5 Add coverage report step (NFR-2.4.4)

### 7.3 Monitoring

- [ ] 7.3.1 Set up logging for LLM API calls (NFR-2.2.4)
- [ ] 7.3.2 Set up error tracking with Sentry or similar (NFR-2.2.4)
- [ ] 7.3.3 Set up performance monitoring with latency metrics (NFR-2.1.1)
- [ ] 7.3.4 Set up cost tracking per paper (C-5.2.1)
- [ ] 7.3.5 Set up alerting for failures (NFR-2.2.3)

## Tasks Summary

| Task ID | Description | Priority | Estimated Time |
|---------|-------------|----------|----------------|
| 1.1.1-1.1.5 | Agent Role Definitions | High | 5 hours |
| 1.2.1-1.2.5 | Data Structures | High | 4 hours |
| 1.3.1-1.3.5 | Configuration | Medium | 2 hours |
| 2.1.1-2.1.5 | Grounding Verifier | High | 6 hours |
| 2.2.1-2.2.5 | Vertex AI Integration | High | 4 hours |
| 3.1.1-3.1.5 | Main Orchestrator | High | 6 hours |
| 3.2.1-3.2.5 | Output Generation | High | 4 hours |
| 3.3.1-3.3.4 | Baseline Integration | Medium | 3 hours |
| 4.1.1-4.1.5 | Unit Tests | High | 6 hours |
| 4.2.1-4.2.6 | Property-Based Tests | Medium | 4 hours |
| 4.3.1-4.3.5 | Integration Tests | Medium | 4 hours |
| 5.1.1-5.1.5 | Code Documentation | Medium | 3 hours |
| 5.2.1-5.2.5 | User Documentation | Low | 4 hours |
| 5.3.1-5.3.5 | API Documentation | Low | 3 hours |
| 6.1.1-6.1.5 | Efficacy Metrics | Medium | 4 hours |
| 6.2.1-6.2.5 | Performance Tuning | Medium | 4 hours |
| 7.1.1-7.1.5 | Environment Setup | High | 2 hours |
| 7.2.1-7.2.5 | CI/CD | Medium | 4 hours |
| 7.3.1-7.3.5 | Monitoring | Medium | 3 hours |

**Total Estimated Time**: 72 hours