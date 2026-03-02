"""
tools.py
--------
Tool definitions that agents can invoke during the agentic loop.

Each tool is a plain Python function.  The tool registry maps tool names
(as the LLM will call them) to their implementations.

Available tools
  - summarise_section   : return a short summary of a text chunk
  - extract_claims      : pull out empirical claims from text
  - check_citation      : placeholder for citation-existence check
  - flag_missing_baselines : highlight missing baseline comparisons
"""

from __future__ import annotations

import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


# ── Tool implementations ───────────────────────────────────────────────────────

def summarise_section(text: str, max_sentences: int = 3) -> str:
    """Return a concise summary of the provided text."""
    client = _get_client()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Summarise the following text in at most {max_sentences} sentences:\n\n{text}"
                ),
            }
        ],
    )
    return response.content[0].text.strip()


def extract_claims(text: str) -> list[str]:
    """Extract a bullet list of empirical claims from text."""
    client = _get_client()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    "List every empirical claim in the text below, one per line, "
                    "prefixed with '- '.\n\n" + text
                ),
            }
        ],
    )
    raw = response.content[0].text.strip()
    return [line.lstrip("- ").strip() for line in raw.splitlines() if line.strip()]


def check_citation(claim: str, bibliography: str) -> str:
    """
    Placeholder: check whether a claim is supported by a citation in the bibliography.
    Returns 'SUPPORTED', 'UNSUPPORTED', or 'UNCLEAR'.
    """
    lowered = claim.lower()
    # Very naive heuristic — replace with a real retrieval step if needed
    if re.search(r"\[\d+\]|\(.*\d{4}\)", claim):
        return "SUPPORTED"
    if any(kw in lowered for kw in ["we show", "we demonstrate", "we prove"]):
        return "UNCLEAR"
    return "UNSUPPORTED"


def flag_missing_baselines(methods_section: str, results_section: str) -> list[str]:
    """
    Ask the LLM to identify baselines that appear in the methods but are absent
    from the results tables.
    """
    client = _get_client()
    prompt = (
        "Methods section:\n"
        f"{methods_section}\n\n"
        "Results section:\n"
        f"{results_section}\n\n"
        "List any baselines or comparisons mentioned in the methods that are "
        "missing from the results. One per line, prefixed with '- '."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    return [line.lstrip("- ").strip() for line in raw.splitlines() if line.strip()]


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "summarise_section": summarise_section,
    "extract_claims": extract_claims,
    "check_citation": check_citation,
    "flag_missing_baselines": flag_missing_baselines,
}


def call_tool(name: str, **kwargs) -> str:
    """Dispatch a tool call by name and return the result as a string."""
    if name not in TOOL_REGISTRY:
        return f"[ERROR] Unknown tool: {name}"
    try:
        result = TOOL_REGISTRY[name](**kwargs)
        if isinstance(result, list):
            return "\n".join(result) if result else "(none found)"
        return str(result)
    except Exception as exc:
        return f"[ERROR] Tool '{name}' raised: {exc}"


# ── Anthropic tool-use schema ──────────────────────────────────────────────────
# Passed to client.messages.create(tools=TOOL_SCHEMAS) when using tool_use.

TOOL_SCHEMAS = [
    {
        "name": "summarise_section",
        "description": "Summarise a section of text in a few sentences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarise."},
                "max_sentences": {"type": "integer", "default": 3},
            },
            "required": ["text"],
        },
    },
    {
        "name": "extract_claims",
        "description": "Extract empirical claims from a passage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Passage to analyse."}
            },
            "required": ["text"],
        },
    },
    {
        "name": "check_citation",
        "description": "Check whether a claim is cited in the bibliography.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "bibliography": {"type": "string"},
            },
            "required": ["claim", "bibliography"],
        },
    },
    {
        "name": "flag_missing_baselines",
        "description": "Find baselines in methods that are absent from results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "methods_section": {"type": "string"},
                "results_section": {"type": "string"},
            },
            "required": ["methods_section", "results_section"],
        },
    },
]
