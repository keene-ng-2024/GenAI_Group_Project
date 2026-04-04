"""
langgraph_critique.py
---------------------
LangGraph-based multi-agent critique workflow with ablation support.

Architecture
  Uses LangGraph's StateGraph to orchestrate the same
  Reader → Critic ↔ Auditor → Summariser pipeline as the other platforms,
  with three loop modes for controlled ablation:

  - none:    Reader → Critic → Summariser  (no debate)
  - fixed:   Reader → Critic → (Auditor → Critic) × N → Summariser  (no early exit)
  - dynamic: Reader → Critic ↔ Auditor → Summariser  (conditional early exit)

Output per paper: results/langgraph_{mode}/<paper_id>.json
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


# ── LLM helper with retry ───────────────────────────────────────────────────

_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds


def _call_llm(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> tuple[str, int, int]:
    """Call the LLM with exponential backoff retry. Returns (text, input_tokens, output_tokens)."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
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
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY * (2 ** attempt)
                print(f"    [RETRY] Attempt {attempt + 1} failed: {exc}. Retrying in {delay}s...")
                time.sleep(delay)
    raise RuntimeError(f"LLM call failed after {_MAX_RETRIES} attempts: {last_exc}") from last_exc


def _add_tokens(state: CritiqueState, input_t: int, output_t: int) -> dict[str, int]:
    """Return updated token_usage dict."""
    prev = state.get("token_usage", {"input": 0, "output": 0})
    return {
        "input": prev["input"] + input_t,
        "output": prev["output"] + output_t,
    }


# ── Node factory functions ───────────────────────────────────────────────────
# Models are passed via closures (not state) to keep CritiqueState clean.

def make_reader_node(model: str):
    def reader_node(state: CritiqueState) -> dict:
        text, in_t, out_t = _call_llm(
            system_prompt=READER_SYSTEM,
            user_message=f"Please summarise the following paper:\n\nTitle: {state['title']}\n\n{state['paper_text']}",
            model=model,
        )
        print(f"    [Reader] {text[:200]}{'…' if len(text) > 200 else ''}")
        return {
            "summary": text,
            "transcript": state["transcript"] + [{"role": "Reader", "content": text}],
            "token_usage": _add_tokens(state, in_t, out_t),
        }
    return reader_node


def make_critic_node(model: str):
    def critic_node(state: CritiqueState) -> dict:
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
            model=model,
        )
        print(f"    [{role_label}] {text[:200]}{'…' if len(text) > 200 else ''}")
        return {
            "critique": text,
            "transcript": state["transcript"] + [{"role": role_label, "content": text}],
            "token_usage": _add_tokens(state, in_t, out_t),
        }
    return critic_node


def make_auditor_node(model: str):
    def auditor_node(state: CritiqueState) -> dict:
        round_num = state["round_num"] + 1
        user_msg = (
            f"Paper summary:\n{state['summary']}\n\n"
            f"Critic's points:\n{state['critique']}\n\n"
            "Challenge weak points and identify anything important that was missed."
        )
        text, in_t, out_t = _call_llm(
            system_prompt=AUDITOR_SYSTEM,
            user_message=user_msg,
            model=model,
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
    return auditor_node


def make_summariser_node(model: str):
    def summariser_node(state: CritiqueState) -> dict:
        full_debate = "\n\n".join(
            f"=== {entry['role']} ===\n{entry['content']}" for entry in state["transcript"]
        )
        text, in_t, out_t = _call_llm(
            system_prompt=SUMMARISER_SYSTEM,
            user_message=(
                f"Here is the full debate transcript:\n\n{full_debate}\n\n"
                "Produce the final structured review JSON."
            ),
            model=model,
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
    return summariser_node


# ── Conditional edge: should the debate continue? ─────────────────────────────

def should_continue(state: CritiqueState) -> str:
    """Route after Auditor: back to Critic or on to Summariser."""
    if state["round_num"] >= state["max_rounds"]:
        print(f"    [STOP] Max rounds ({state['max_rounds']}) reached.")
        return "summariser"

    feedback_lower = state["audit_feedback"].lower()
    for phrase in state.get("early_stop_phrases", []):
        phrase_lower = phrase.lower()
        # Check every occurrence of the phrase, not just the first
        start = 0
        while True:
            idx = feedback_lower.find(phrase_lower, start)
            if idx < 0:
                break
            # Extract the full sentence containing this phrase for negation check
            sent_start = max(
                feedback_lower.rfind(".", 0, idx) + 1,
                feedback_lower.rfind("!", 0, idx) + 1,
                feedback_lower.rfind("?", 0, idx) + 1,
                0,
            )
            sentence = feedback_lower[sent_start:idx]
            negations = ["not ", "no ", "don't ", "doesn't ", "isn't ", "aren't ",
                         "hardly ", "never ", "cannot ", "can't ", "won't "]
            if not any(neg in sentence for neg in negations):
                print(f"    [STOP] Auditor satisfied ('{phrase}') after round {state['round_num']}.")
                return "summariser"
            start = idx + 1

    return "critic"


def _always_summariser(state: CritiqueState) -> str:
    """Fixed-loop routing: always go back to critic until max_rounds, then summarise."""
    if state["round_num"] >= state["max_rounds"]:
        print(f"    [STOP] Fixed rounds ({state['max_rounds']}) completed.")
        return "summariser"
    return "critic"


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph(
    loop_mode: str,
    models: dict[str, str],
) -> StateGraph:
    """Construct and compile the LangGraph critique workflow.

    Args:
        loop_mode: One of "none", "fixed", "dynamic".
        models: Dict with keys reader, critic, auditor, summariser mapping to model names.
    """
    graph = StateGraph(CritiqueState)

    graph.add_node("reader", make_reader_node(models["reader"]))
    graph.add_node("critic", make_critic_node(models["critic"]))
    graph.add_node("summariser", make_summariser_node(models["summariser"]))

    graph.add_edge(START, "reader")
    graph.add_edge("reader", "critic")

    if loop_mode == "none":
        # No auditor at all: Reader → Critic → Summariser
        graph.add_edge("critic", "summariser")
    else:
        # Both fixed and dynamic use the auditor
        graph.add_node("auditor", make_auditor_node(models["auditor"]))
        graph.add_edge("critic", "auditor")

        if loop_mode == "fixed":
            graph.add_conditional_edges("auditor", _always_summariser)
        else:  # dynamic
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
    app,
    loop_mode: str,
) -> dict:
    """Run the LangGraph critique workflow for one paper using a pre-compiled graph."""
    truncate_chars = cfg["agent"].get("truncate_body_chars", 12000)
    title = paper.get("title", paper_id)
    full_text = paper.get("full_text", "")
    paper_text = full_text[:truncate_chars] if full_text else paper.get("abstract", title)

    lg_cfg = cfg.get("langgraph", {})

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
        "max_rounds": cfg["agent"]["max_rounds"] if loop_mode != "none" else 0,
        "early_stop_phrases": lg_cfg.get("early_stop_phrases",
                                          cfg["agent"].get("early_stop_phrases", [])),
        "structured": {},
        "critique_points": {},
        "token_usage": {"input": 0, "output": 0},
    })

    latency_seconds = round(time.perf_counter() - start_time, 2)

    return {
        "paper_id": paper_id,
        "title": title,
        "platform": "langgraph",
        "loop_mode": loop_mode,
        "model": lg_cfg.get("critic_model", "gpt-4o"),
        "rounds": result["round_num"],
        "latency_seconds": latency_seconds,
        "token_usage": result["token_usage"],
        "transcript": result["transcript"],
        "structured": result["structured"],
        "critique_points": result["critique_points"],
    }


# ── Batch pipeline ─────────────────────────────────────────────────────────────

def run_all_papers(
    reviews_path: str,
    output_dir: str,
    cfg: dict,
    loop_mode: str,
) -> None:
    with open(reviews_path) as f:
        all_papers: dict = json.load(f)

    lg_cfg = cfg.get("langgraph", {})
    models = {
        "reader": lg_cfg.get("reader_model", "gpt-4o-mini"),
        "critic": lg_cfg.get("critic_model", "gpt-4o"),
        "auditor": lg_cfg.get("auditor_model", "gpt-4o-mini"),
        "summariser": lg_cfg.get("summariser_model", "gpt-4o"),
    }

    # Compile graph ONCE and reuse for all papers
    app = build_graph(loop_mode=loop_mode, models=models)

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
                app=app,
                loop_mode=loop_mode,
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
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph critique pipeline")
    parser.add_argument(
        "--mode",
        choices=["none", "fixed", "dynamic", "all"],
        default=None,
        help="Loop mode to run. 'all' runs all three. Defaults to config.yaml langgraph.loop_mode.",
    )
    args = parser.parse_args()

    cfg = load_config()
    lg_cfg = cfg.get("langgraph", {})

    if args.mode == "all":
        modes = ["none", "fixed", "dynamic"]
    elif args.mode:
        modes = [args.mode]
    else:
        modes = [lg_cfg.get("loop_mode", "dynamic")]

    for mode in modes:
        out_dir = cfg["results"].get(f"langgraph_{mode}_dir",
                                     f"results/langgraph_{mode}")
        print(f"\n{'#'*60}")
        print(f"  Running LangGraph ablation: loop_mode={mode}")
        print(f"  Output → {out_dir}")
        print(f"{'#'*60}")

        run_all_papers(
            reviews_path=cfg["data"]["reviews_file"],
            output_dir=out_dir,
            cfg=cfg,
            loop_mode=mode,
        )
