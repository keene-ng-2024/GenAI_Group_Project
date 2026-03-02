"""
agents.py
---------
Agent role definitions for the paper-critique multi-agent system.

Roles
  Reader     – reads & summarises the paper section by section
  Critic     – proposes critique points based on the Reader's summary
  Auditor    – challenges the Critic's points and requests evidence
  Summariser – consolidates the debate into a final critique dict

Each agent is a dataclass holding its system prompt and a method to
generate a response given a conversation history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

load_dotenv()


# ── Shared client ──────────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Base agent ─────────────────────────────────────────────────────────────────

@dataclass
class BaseAgent:
    name: str
    system_prompt: str
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    temperature: float = 0.2
    history: list[dict] = field(default_factory=list)

    def chat(self, user_message: str) -> str:
        """Send a message and get a reply, updating internal history."""
        self.history.append({"role": "user", "content": user_message})
        client = _get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=self.history,
        )
        reply = response.content[0].text.strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self.history.clear()


# ── Role definitions ───────────────────────────────────────────────────────────

class ReaderAgent(BaseAgent):
    """Reads the paper and produces a structured summary."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        super().__init__(
            name="Reader",
            system_prompt=(
                "You are a careful academic reader. "
                "When given a paper (or section of a paper), you produce a structured "
                "summary covering: (1) Problem & motivation, (2) Proposed method, "
                "(3) Key experiments and results, (4) Claimed contributions. "
                "Be factual and concise."
            ),
            model=model,
            **kwargs,
        )

    def summarise_paper(self, paper_text: str) -> str:
        return self.chat(f"Please summarise the following paper:\n\n{paper_text}")


class CriticAgent(BaseAgent):
    """Generates critique points from a paper summary."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        super().__init__(
            name="Critic",
            system_prompt=(
                "You are a rigorous peer reviewer for a top-tier ML/AI venue. "
                "Given a paper summary, identify substantive weaknesses in: "
                "novelty, methodology, evaluation, clarity, and reproducibility. "
                "For each point give: (a) the issue, (b) why it matters, "
                "(c) what evidence from the paper supports your concern. "
                "Be specific and actionable."
            ),
            model=model,
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

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        super().__init__(
            name="Auditor",
            system_prompt=(
                "You are a senior programme committee member auditing a peer review. "
                "Your job is to challenge poorly-supported critique points: "
                "ask for concrete evidence, flag over-interpretations, and identify "
                "any points that are actually addressed in the paper. "
                "Also highlight genuine issues the Critic may have missed. "
                "Be constructive but demanding."
            ),
            model=model,
            **kwargs,
        )

    def audit(self, critique: str, paper_summary: str) -> str:
        return self.chat(
            f"Paper summary:\n{paper_summary}\n\n"
            f"Critic's points:\n{critique}\n\n"
            "Challenge weak points and identify anything important that was missed."
        )


class SummariserAgent(BaseAgent):
    """Consolidates the debate into a final JSON critique dictionary."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        super().__init__(
            name="Summariser",
            system_prompt=(
                "You are a senior editor. Given a debate between a Critic and Auditor "
                "about the weaknesses of a paper, synthesise their discussion into a "
                "final, deduplicated dictionary of critique points. "
                "Output ONLY valid JSON: a dict mapping 'point_NNN' keys to "
                "concise critique strings. Include 5–12 points."
            ),
            model=model,
            **kwargs,
        )

    def summarise(self, debate_transcript: str) -> str:
        return self.chat(
            f"Here is the full debate transcript:\n\n{debate_transcript}\n\n"
            "Produce the final JSON critique dictionary."
        )


# ── Factory ────────────────────────────────────────────────────────────────────

def build_agents(cfg: dict) -> dict[str, BaseAgent]:
    model = cfg["models"]["strong"]
    return {
        "reader": ReaderAgent(model=model),
        "critic": CriticAgent(model=model),
        "auditor": AuditorAgent(model=model),
        "summariser": SummariserAgent(model=model),
    }
