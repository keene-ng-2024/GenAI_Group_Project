"""
agents.py
---------
Agent role definitions for the paper-critique multi-agent system.

Roles
  Reader     – reads & summarises the paper section by section
  Critic     – proposes critique points based on the Reader's summary
  Auditor    – challenges the Critic's points and requests evidence
  Summariser – consolidates the debate into a final structured review

Each agent is a dataclass holding its system prompt and a method to
generate a response given a conversation history.  Agents that have
use_tools=True will dispatch tool calls via the Anthropic tool-use API
and loop until the model returns a final text response.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

from src.agents.tools import TOOL_SCHEMAS, call_tool

load_dotenv()


# ── Shared client ──────────────────────────────────────────────────────────────

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


# ── Base agent ─────────────────────────────────────────────────────────────────

@dataclass
class BaseAgent:
    name: str
    system_prompt: str
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.2
    use_tools: bool = False
    max_tool_calls: int = 5
    history: list[dict] = field(default_factory=list)

    # Accumulated token usage across all calls
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def chat(self, user_message: str) -> str:
        """Send a message and get a reply, updating internal history.

        If use_tools is True and the model requests tool calls, this method
        dispatches them via call_tool() and loops until the model produces a
        final text response (or max_tool_calls is reached).
        """
        self.history.append({"role": "user", "content": user_message})
        client = _get_client()

        tool_calls_remaining = self.max_tool_calls

        while True:
            api_kwargs: dict = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=self.history,
            )
            if self.use_tools and tool_calls_remaining > 0:
                api_kwargs["tools"] = TOOL_SCHEMAS

            response = client.messages.create(**api_kwargs)

            # Track token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            # If the model wants to call tools, dispatch and loop
            if response.stop_reason == "tool_use" and tool_calls_remaining > 0:
                self.history.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = call_tool(block.name, **block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                        tool_calls_remaining -= 1

                self.history.append({"role": "user", "content": tool_results})
                continue

            # Extract text from the final response
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            reply = "\n".join(text_parts).strip()

            if not reply and response.stop_reason == "tool_use":
                reply = "[Agent hit tool call limit without producing a text response]"
                print(f"  [WARN] {self.name} exhausted tool calls without final text.")

            self.history.append({"role": "assistant", "content": reply})
            return reply

    def reset(self) -> None:
        self.history.clear()

    def get_token_usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }


# ── Role definitions ───────────────────────────────────────────────────────────

class ReaderAgent(BaseAgent):
    """Reads the paper and produces a structured summary with labelled sections."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        super().__init__(
            name="Reader",
            system_prompt=(
                "You are a careful academic reader. "
                "When given a paper (or section of a paper), produce a structured "
                "summary with the following clearly labelled sections:\n\n"
                "## Problem & Motivation\n"
                "## Proposed Method\n"
                "## Methods\n"
                "## Results\n"
                "## Claimed Contributions\n\n"
                "Be factual and concise. Include specific numbers from experiments."
            ),
            model=model,
            **kwargs,
        )

    def summarise_paper(self, paper_text: str) -> str:
        return self.chat(f"Please summarise the following paper:\n\n{paper_text}")


class CriticAgent(BaseAgent):
    """Generates critique points from a paper summary."""

    def __init__(self, model: str = "claude-sonnet-4-6", use_tools: bool = False, **kwargs):
        tool_instructions = ""
        if use_tools:
            tool_instructions = (
                "\n\nYou have access to tools. Use `extract_claims` to identify "
                "empirical claims before critiquing them. Use `flag_missing_baselines` "
                "to check for missing comparisons (pass the Methods and Results "
                "sections separately)."
            )
        super().__init__(
            name="Critic",
            system_prompt=(
                "You are a rigorous peer reviewer for a top-tier ML/AI venue. "
                "Given a paper summary, identify substantive weaknesses in: "
                "novelty, methodology, evaluation, clarity, and reproducibility. "
                "For each point give: (a) the issue, (b) why it matters, "
                "(c) what evidence from the paper supports your concern. "
                "Be specific and actionable."
                + tool_instructions
            ),
            model=model,
            use_tools=use_tools,
            **kwargs,
        )

    def generate_critique(self, paper_summary: str) -> str:
        return self.chat(
            f"Here is the paper summary:\n\n{paper_summary}\n\n"
            "Now list your critique points."
        )

    def revise_critique(self, auditor_feedback: str) -> str:
        return self.chat(
            f"The Auditor has challenged some of your points:\n\n{auditor_feedback}\n\n"
            "Revise or defend your critique points accordingly."
        )


class AuditorAgent(BaseAgent):
    """Challenges the Critic's points and asks for evidence."""

    def __init__(self, model: str = "claude-sonnet-4-6", use_tools: bool = False, **kwargs):
        tool_instructions = ""
        if use_tools:
            tool_instructions = (
                "\n\nYou have access to tools. Use `extract_claims` and "
                "`check_citation` to verify whether the Critic's references are "
                "supported before deciding if a point is well-grounded."
            )
        super().__init__(
            name="Auditor",
            system_prompt=(
                "You are a senior programme committee member auditing a peer review. "
                "Your job is to challenge poorly-supported critique points: "
                "ask for concrete evidence, flag over-interpretations, and identify "
                "any points that are actually addressed in the paper. "
                "Also highlight genuine issues the Critic may have missed. "
                "Be constructive but demanding."
                + tool_instructions
            ),
            model=model,
            use_tools=use_tools,
            **kwargs,
        )

    def audit(self, critique: str, paper_summary: str) -> str:
        return self.chat(
            f"Paper summary:\n{paper_summary}\n\n"
            f"Critic's points:\n{critique}\n\n"
            "Challenge weak points and identify anything important that was missed."
        )


STRUCTURED_OUTPUT_SCHEMA = """\
{
  "summary": "<2-3 sentence paper summary>",
  "strengths": [
    {"point": "<strength>", "evidence": "<supporting evidence from paper>"}
  ],
  "weaknesses": [
    {"point": "<weakness>", "evidence": "<supporting evidence from paper>"}
  ],
  "questions": [
    {"question": "<question for authors>", "motivation": "<why this matters>"}
  ],
  "scores": {
    "correctness": <int 1-5>,
    "novelty": <int 1-5>,
    "recommendation": "<accept|borderline|reject>",
    "confidence": <int 1-5>
  }
}"""


class SummariserAgent(BaseAgent):
    """Consolidates the debate into a final structured review JSON."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        super().__init__(
            name="Summariser",
            max_tokens=4096,
            system_prompt=(
                "You are a senior editor. Given a debate between a Critic and Auditor "
                "about a paper, synthesise their discussion into a final structured review.\n\n"
                "Output ONLY valid JSON matching this exact schema:\n"
                f"{STRUCTURED_OUTPUT_SCHEMA}\n\n"
                "Rules:\n"
                "- Include 3-8 strengths and 3-8 weaknesses (deduplicated).\n"
                "- Include 2-5 questions for the authors.\n"
                "- Base scores on the reviewer score context provided and the debate.\n"
                "- Output nothing except the JSON object. No markdown fences, no commentary."
            ),
            model=model,
            **kwargs,
        )

    def summarise(self, debate_transcript: str) -> str:
        return self.chat(
            f"Here is the full debate transcript:\n\n{debate_transcript}\n\n"
            "Produce the final structured review JSON."
        )


# ── Factory ────────────────────────────────────────────────────────────────────

def build_agents(cfg: dict) -> dict[str, BaseAgent]:
    """Build agents with per-role models and tool-use settings from config."""
    model_map = cfg["agent"].get("model_map", {})
    default_model = cfg["models"]["strong"]
    use_tools = cfg["agent"].get("use_tools", False)
    max_tool_calls = cfg["agent"].get("max_tool_calls", 5)

    return {
        "reader": ReaderAgent(
            model=model_map.get("reader", default_model),
            max_tool_calls=max_tool_calls,
        ),
        "critic": CriticAgent(
            model=model_map.get("critic", default_model),
            use_tools=use_tools,
            max_tool_calls=max_tool_calls,
        ),
        "auditor": AuditorAgent(
            model=model_map.get("auditor", default_model),
            use_tools=use_tools,
            max_tool_calls=max_tool_calls,
        ),
        "summariser": SummariserAgent(
            model=model_map.get("summariser", default_model),
            max_tool_calls=max_tool_calls,
        ),
    }
