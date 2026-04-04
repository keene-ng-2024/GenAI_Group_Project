"""
langgraph_critique.py
---------------------
LangGraph-based multi-agent critique workflow.

Architecture
  Uses LangGraph's StateGraph to orchestrate the same
  Reader → Critic ↔ Auditor → Summariser pipeline as the other platforms,
  with conditional edges for the debate loop and early stopping.

  ┌────────┐      ┌────────┐      ┌─────────┐
  │ Reader │─────►│ Critic │─────►│ Auditor │
  └────────┘      └────────┘      └─────────┘
                       ▲               │
                       └───────────────┘  (conditional: continue or stop)
                                       │
                                  ┌────▼──────┐
                                  │Summariser │──► END
                                  └───────────┘

Output per paper: results/langgraph/<paper_id>.json  (same schema as other platforms)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Graph state ────────────────────────────────────────────────────────────────

class CritiqueState(TypedDict):
    paper_id: str
    title: str
    paper_text: str
    summary: str
    critique: str
    audit_feedback: str
    transcript: list[dict]
    round_num: int
    max_rounds: int
    early_stop_phrases: list[str]
    structured: dict
    critique_points: dict[str, str]
    token_usage: dict[str, int]


# ── System prompts (same as agents.py for fair comparison) ─────────────────────

READER_SYSTEM = (
    "You are a careful academic reader. "
    "When given a paper (or section of a paper), produce a structured "
    "summary with the following clearly labelled sections:\n\n"
    "## Problem & Motivation\n"
    "## Proposed Method\n"
    "## Methods\n"
    "## Results\n"
    "## Claimed Contributions\n\n"
    "Be factual and concise. Include specific numbers from experiments."
)

CRITIC_SYSTEM = (
    "You are a rigorous peer reviewer for a top-tier ML/AI venue. "
    "Given a paper summary, identify substantive weaknesses in: "
    "novelty, methodology, evaluation, clarity, and reproducibility. "
    "For each point give: (a) the issue, (b) why it matters, "
    "(c) what evidence from the paper supports your concern. "
    "Be specific and actionable."
)

AUDITOR_SYSTEM = (
    "You are a senior programme committee member auditing a peer review. "
    "Your job is to challenge poorly-supported critique points: "
    "ask for concrete evidence, flag over-interpretations, and identify "
    "any points that are actually addressed in the paper. "
    "Also highlight genuine issues the Critic may have missed. "
    "Be constructive but demanding."
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

SUMMARISER_SYSTEM = (
    "You are a senior editor. Given a debate between a Critic and Auditor "
    "about a paper, synthesise their discussion into a final structured review.\n\n"
    "Output ONLY valid JSON matching this exact schema:\n"
    f"{STRUCTURED_OUTPUT_SCHEMA}\n\n"
    "Rules:\n"
    "- Include 3-8 strengths and 3-8 weaknesses (deduplicated).\n"
    "- Include 2-5 questions for the authors.\n"
    "- Base scores on the debate content.\n"
    "- Output nothing except the JSON object. No markdown fences, no commentary."
)


# ── LLM helper ────────────────────────────────────────────────────────────────

def _call_llm(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> tuple[str, int, int]:
    """Call the LLM and return (text, input_tokens, output_tokens)."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])
    usage = response.usage_metadata or {}
    return (
        response.content,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )


def _add_tokens(state: CritiqueState, input_t: int, output_t: int) -> dict[str, int]:
    """Return updated token_usage dict."""
    prev = state.get("token_usage", {"input": 0, "output": 0})
    return {
        "input": prev["input"] + input_t,
        "output": prev["output"] + output_t,
    }


# ── Node functions ─────────────────────────────────────────────────────────────

def reader_node(state: CritiqueState) -> dict:
    """Reader: summarise the paper."""
    text, in_t, out_t = _call_llm(
        system_prompt=READER_SYSTEM,
        user_message=f"Please summarise the following paper:\n\nTitle: {state['title']}\n\n{state['paper_text']}",
        model=state.get("_model_reader", "gpt-4o-mini"),
    )
    print(f"    [Reader] {text[:200]}{'…' if len(text) > 200 else ''}")
    return {
        "summary": text,
        "transcript": state["transcript"] + [{"role": "Reader", "content": text}],
        "token_usage": _add_tokens(state, in_t, out_t),
    }


def critic_node(state: CritiqueState) -> dict:
    """Critic: generate initial critique or revise based on auditor feedback."""
    if state["round_num"] == 0:
        user_msg = (
            f"Here is the paper summary:\n\n{state['summary']}\n\n"
            "Now list your critique points."
        )
        role_label = "Critic (initial)"
    else:
        user_msg = (
            f"The Auditor has challenged some of your points:\n\n{state['audit_feedback']}\n\n"
            "Revise or defend your critique points accordingly."
        )
        role_label = f"Critic (round {state['round_num']})"

    text, in_t, out_t = _call_llm(
        system_prompt=CRITIC_SYSTEM,
        user_message=user_msg,
        model=state.get("_model_critic", "gpt-4o"),
    )
    print(f"    [{role_label}] {text[:200]}{'…' if len(text) > 200 else ''}")
    return {
        "critique": text,
        "transcript": state["transcript"] + [{"role": role_label, "content": text}],
        "token_usage": _add_tokens(state, in_t, out_t),
    }


def auditor_node(state: CritiqueState) -> dict:
    """Auditor: challenge the critic's points."""
    round_num = state["round_num"] + 1
    user_msg = (
        f"Paper summary:\n{state['summary']}\n\n"
        f"Critic's points:\n{state['critique']}\n\n"
        "Challenge weak points and identify anything important that was missed."
    )
    text, in_t, out_t = _call_llm(
        system_prompt=AUDITOR_SYSTEM,
        user_message=user_msg,
        model=state.get("_model_auditor", "gpt-4o-mini"),
    )
    print(f"    [Auditor (round {round_num})] {text[:200]}{'…' if len(text) > 200 else ''}")
    return {
        "audit_feedback": text,
        "round_num": round_num,
        "transcript": state["transcript"] + [
            {"role": f"Auditor (round {round_num})", "content": text}
        ],
        "token_usage": _add_tokens(state, in_t, out_t),
    }


def summariser_node(state: CritiqueState) -> dict:
    """Summariser: consolidate the debate into structured JSON."""
    full_debate = "\n\n".join(
        f"=== {entry['role']} ===\n{entry['content']}" for entry in state["transcript"]
    )
    text, in_t, out_t = _call_llm(
        system_prompt=SUMMARISER_SYSTEM,
        user_message=(
            f"Here is the full debate transcript:\n\n{full_debate}\n\n"
            "Produce the final structured review JSON."
        ),
        model=state.get("_model_summariser", "gpt-4o"),
    )
    print(f"    [Summariser] {text[:200]}{'…' if len(text) > 200 else ''}")

    structured = _parse_structured_output(text)
    critique_points = _flatten_to_critique_points(structured)

    return {
        "structured": structured,
        "critique_points": critique_points,
        "transcript": state["transcript"] + [{"role": "Summariser", "content": text}],
        "token_usage": _add_tokens(state, in_t, out_t),
    }


# ── Conditional edge: should the debate continue? ─────────────────────────────

def should_continue(state: CritiqueState) -> str:
    """Route after Auditor: back to Critic or on to Summariser."""
    if state["round_num"] >= state["max_rounds"]:
        print(f"    [STOP] Max rounds ({state['max_rounds']}) reached.")
        return "summariser"

    feedback_lower = state["audit_feedback"].lower()
    for phrase in state.get("early_stop_phrases", []):
        idx = feedback_lower.find(phrase)
        if idx >= 0:
            prefix = feedback_lower[max(0, idx - 15):idx]
            if not any(neg in prefix for neg in ["not ", "no ", "don't ", "isn't ", "hardly "]):
                print(f"    [STOP] Auditor satisfied ('{phrase}') after round {state['round_num']}.")
                return "summariser"

    return "critic"


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph critique workflow."""
    graph = StateGraph(CritiqueState)

    graph.add_node("reader", reader_node)
    graph.add_node("critic", critic_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("summariser", summariser_node)

    graph.add_edge(START, "reader")
    graph.add_edge("reader", "critic")
    graph.add_edge("critic", "auditor")
    graph.add_conditional_edges("auditor", should_continue)
    graph.add_edge("summariser", END)

    return graph.compile()


# ── Output parsing (shared with other platforms) ───────────────────────────────

def _parse_structured_output(raw: str) -> dict:
    """Parse the Summariser's structured JSON output with fallbacks."""
    text = raw.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print("    [WARN] Failed to parse structured output, returning empty structure")
    return {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "scores": {
            "correctness": 3,
            "novelty": 3,
            "recommendation": "borderline",
            "confidence": 1,
        },
    }


def _flatten_to_critique_points(structured: dict) -> dict[str, str]:
    """Convert structured output to flat critique_points dict for scorer compat."""
    points = {}
    idx = 1
    for item in structured.get("weaknesses", []):
        if isinstance(item, str):
            full = item
        elif isinstance(item, dict):
            point_text = item.get("point", "")
            evidence = item.get("evidence", "")
            full = f"{point_text}. {evidence}".strip(" .") if evidence else point_text
        else:
            continue
        points[f"point_{idx:03d}"] = full
        idx += 1
    return points


# ── Single paper critique ──────────────────────────────────────────────────────

def critique_paper(
    paper_id: str,
    paper: dict,
    cfg: dict,
) -> dict:
    """Run the LangGraph critique workflow for one paper."""
    truncate_chars = cfg["agent"].get("truncate_body_chars", 12000)
    title = paper.get("title", paper_id)
    full_text = paper.get("full_text", "")
    paper_text = full_text[:truncate_chars] if full_text else paper.get("abstract", title)

    lg_cfg = cfg.get("langgraph", {})

    app = build_graph()
    start_time = time.perf_counter()

    result = app.invoke({
        "paper_id": paper_id,
        "title": title,
        "paper_text": paper_text,
        "summary": "",
        "critique": "",
        "audit_feedback": "",
        "transcript": [],
        "round_num": 0,
        "max_rounds": cfg["agent"]["max_rounds"],
        "early_stop_phrases": cfg["agent"].get("early_stop_phrases", []),
        "structured": {},
        "critique_points": {},
        "token_usage": {"input": 0, "output": 0},
        # Model config from langgraph section of config.yaml
        "_model_reader": lg_cfg.get("reader_model", "gpt-4o-mini"),
        "_model_critic": lg_cfg.get("critic_model", "gpt-4o"),
        "_model_auditor": lg_cfg.get("auditor_model", "gpt-4o-mini"),
        "_model_summariser": lg_cfg.get("summariser_model", "gpt-4o"),
    })

    latency_seconds = round(time.perf_counter() - start_time, 2)

    return {
        "paper_id": paper_id,
        "title": title,
        "platform": "langgraph",
        "model": lg_cfg.get("critic_model", "gpt-4o"),
        "rounds": result["round_num"],
        "latency_seconds": latency_seconds,
        "token_usage": result["token_usage"],
        "transcript": result["transcript"],
        "structured": result["structured"],
        "critique_points": result["critique_points"],
    }


# ── Batch pipeline ─────────────────────────────────────────────────────────────

def run_all_papers(reviews_path: str, output_dir: str, cfg: dict) -> None:
    with open(reviews_path) as f:
        all_papers: dict = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for paper_id, paper in all_papers.items():
        out_file = out_dir / f"{paper_id}.json"
        if out_file.exists():
            print(f"  [SKIP] {paper_id}")
            continue

        print(f"\n{'='*60}\n  [PAPER] {paper_id} — {paper.get('title', '')[:50]}\n{'='*60}")

        try:
            result = critique_paper(
                paper_id=paper_id,
                paper=paper,
                cfg=cfg,
            )
        except Exception as exc:
            print(f"\n  [ERROR] {paper_id} failed: {exc}")
            continue

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        n_points = len(result["critique_points"])
        print(f"\n  [SAVED] {n_points} weakness points → {out_file}")
        print(f"  [COST]  {result['token_usage']['input']:,} in / "
              f"{result['token_usage']['output']:,} out tokens  "
              f"({result['latency_seconds']}s)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_all_papers(
        reviews_path=cfg["data"]["reviews_file"],
        output_dir=cfg["results"]["langgraph_dir"],
        cfg=cfg,
    )
