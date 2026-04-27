"""
test_vertex_pr_fixes.py
-----------------------
Tests for the three PR fixes:
1. _parse_structured_output() robustness
2. run_pipeline() includes platform field
3. patch script: platform field + zero-point re-parse
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from patch_vertexai_results import (
    _parse_structured_output,
    _flatten_to_critique_points,
    patch_results,
)

# ── shared fixture ─────────────────────────────────────────────────────────────

VALID_STRUCTURED = {
    "summary": "A paper about X.",
    "strengths": [{"point": "Good eval", "evidence": "Table 1"}],
    "weaknesses": [
        {"point": "Missing ablation", "evidence": "Section 4"},
        {"point": "No baseline", "evidence": "Section 3"},
    ],
    "questions": [{"question": "Why?", "motivation": "Clarity"}],
    "scores": {"correctness": 4, "novelty": 3, "recommendation": "borderline", "confidence": 3},
}

VALID_JSON_STR = json.dumps(VALID_STRUCTURED)


# ── Task 5.1: _parse_structured_output unit tests ─────────────────────────────

class TestParseStructuredOutput:

    def test_fenced_with_json_tag(self):
        raw = f"```json\n{VALID_JSON_STR}\n```"
        result = _parse_structured_output(raw)
        assert result["summary"] == VALID_STRUCTURED["summary"]
        assert len(result["weaknesses"]) == 2

    def test_fenced_without_tag(self):
        raw = f"```\n{VALID_JSON_STR}\n```"
        result = _parse_structured_output(raw)
        assert result["summary"] == VALID_STRUCTURED["summary"]

    def test_fenced_with_trailing_newline(self):
        raw = f"```json\n{VALID_JSON_STR}\n```\n"
        result = _parse_structured_output(raw)
        assert result["summary"] == VALID_STRUCTURED["summary"]

    def test_plain_json_preserved(self):
        result = _parse_structured_output(VALID_JSON_STR)
        assert result["summary"] == VALID_STRUCTURED["summary"]
        assert len(result["weaknesses"]) == 2

    def test_malformed_returns_fallback(self):
        result = _parse_structured_output("this is not json at all !!!")
        assert result["summary"] == ""
        assert result["weaknesses"] == []
        assert result["scores"]["correctness"] == 3

    def test_fenced_with_latex_escapes(self):
        # Summarizer sometimes includes LaTeX like \alpha in JSON strings
        latex_structured = dict(VALID_STRUCTURED)
        latex_structured["summary"] = r"Uses \alpha and \beta parameters."
        raw = "```json\n" + json.dumps(latex_structured) + "\n```"
        result = _parse_structured_output(raw)
        assert "alpha" in result["summary"]

    def test_empty_string_returns_fallback(self):
        result = _parse_structured_output("")
        assert result["weaknesses"] == []


# ── Task 5.2: run_pipeline() includes platform field ─────────────────────────

class TestRunPipelinePlatformField:

    def test_platform_field_present(self):
        """run_pipeline() return dict must include platform='vertexai'."""
        # We mock the Vertex AI client to avoid real API calls
        mock_response = {
            "text": json.dumps(VALID_STRUCTURED),
            "input_tokens": 100,
            "output_tokens": 50,
        }

        with patch("agents.vertex_client.VertexAIClient.generate_content",
                   return_value=mock_response):
            with patch("agents.grounding_verifier.verify_all_grounding",
                       return_value={"avg_confidence": 0.8, "grounding_rate": 0.9,
                                     "points_verified": 2, "points_unsupported": 0}):
                from agents.vertex_orchestrator import run_pipeline
                cfg = {
                    "vertex_ai": {
                        "project": "test-project",
                        "location": "us-central1",
                        "reader_model": "gemini-2.5-flash-lite",
                        "critic_model": "gemini-2.5-flash",
                        "auditor_model": "gemini-2.5-flash-lite",
                        "summariser_model": "gemini-2.5-flash",
                    },
                    "agent": {"max_rounds": 1, "early_stop_phrases": ["i am satisfied"]},
                }
                result = run_pipeline("paper_test", "Some paper text.", config=cfg)

        assert "platform" in result, "platform field missing from run_pipeline() output"
        assert result["platform"] == "vertexai"


# ── Task 5.3: patch script unit tests ────────────────────────────────────────

class TestPatchScript:

    def _make_result_file(self, tmp_dir: Path, name: str, critique_points: dict,
                          summarizer_content: str, include_platform: bool = False) -> Path:
        data = {
            "paper_id": name,
            "model": "gemini-2.5-flash",
            "rounds": 1,
            "latency_seconds": 10.0,
            "token_usage": {"input": 100, "output": 50},
            "transcript": [
                {"role": "Reader", "content": "summary"},
                {"role": "Summarizer", "content": summarizer_content},
            ],
            "structured": {"summary": "", "strengths": [], "weaknesses": [],
                           "questions": [], "scores": {"correctness": 3, "novelty": 3,
                                                        "recommendation": "borderline", "confidence": 1}},
            "critique_points": critique_points,
            "grounding_verifier_scores": {},
            "run_metadata": {},
        }
        if include_platform:
            data["platform"] = "vertexai"
        fpath = tmp_dir / f"{name}.json"
        fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return fpath

    def test_adds_platform_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._make_result_file(tmp_dir, "paper_0001", {"point_001": "x"}, VALID_JSON_STR)
            patch_results(str(tmp_dir))
            data = json.loads((tmp_dir / "paper_0001.json").read_text())
            assert data["platform"] == "vertexai"

    def test_does_not_overwrite_existing_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._make_result_file(tmp_dir, "paper_0001", {"point_001": "x"},
                                   VALID_JSON_STR, include_platform=True)
            patch_results(str(tmp_dir))
            data = json.loads((tmp_dir / "paper_0001.json").read_text())
            assert data["platform"] == "vertexai"

    def test_repairs_zero_point_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            fenced = f"```json\n{VALID_JSON_STR}\n```"
            self._make_result_file(tmp_dir, "paper_0043", {}, fenced)
            patch_results(str(tmp_dir))
            data = json.loads((tmp_dir / "paper_0043.json").read_text())
            assert len(data["critique_points"]) == 2
            assert data["structured"]["summary"] == VALID_STRUCTURED["summary"]

    def test_does_not_modify_nonzero_critique_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            original_points = {"point_001": "original point text"}
            self._make_result_file(tmp_dir, "paper_0001", original_points, VALID_JSON_STR)
            patch_results(str(tmp_dir))
            data = json.loads((tmp_dir / "paper_0001.json").read_text())
            assert data["critique_points"] == original_points


# ── Task 5.4: property-based tests ───────────────────────────────────────────

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

if HYPOTHESIS_AVAILABLE:

    # Strategy: generate a valid structured dict as JSON, wrap in various fence formats
    _fence_formats = st.sampled_from([
        lambda s: f"```json\n{s}\n```",
        lambda s: f"```json\n{s}\n```\n",
        lambda s: f"```\n{s}\n```",
        lambda s: s,  # plain JSON, no fences
    ])

    @given(
        summary=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"))),
        n_weaknesses=st.integers(min_value=0, max_value=8),
        fence_fn=_fence_formats,
    )
    @settings(max_examples=100)
    def test_pbt_parse_always_extracts_weaknesses(summary, n_weaknesses, fence_fn):
        """Fixed parser always extracts the correct number of weaknesses from any fence format."""
        structured = {
            "summary": summary,
            "strengths": [],
            "weaknesses": [{"point": f"p{i}", "evidence": f"e{i}"} for i in range(n_weaknesses)],
            "questions": [],
            "scores": {"correctness": 3, "novelty": 3, "recommendation": "borderline", "confidence": 3},
        }
        raw = fence_fn(json.dumps(structured))
        result = _parse_structured_output(raw)
        assert len(result.get("weaknesses", [])) == n_weaknesses

    @given(
        summary=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"))),
    )
    @settings(max_examples=50)
    def test_pbt_plain_json_no_regression(summary):
        """Plain JSON (no fences) is always parsed correctly — no regression."""
        structured = dict(VALID_STRUCTURED)
        structured["summary"] = summary
        raw = json.dumps(structured)
        result = _parse_structured_output(raw)
        assert result["summary"] == summary
