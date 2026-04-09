"""
state.py
--------
State management for the Vertex AI multi-agent orchestration.

This module provides:
- AgentState TypedDict for type-safe state management
- State initialization and update functions
- State persistence utilities
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
import time


class AgentState(TypedDict):
    """
    TypedDict for agent state management.
    
    Attributes:
        paper_id: Unique identifier for the paper
        title: Paper title
        abstract: Paper abstract
        full_text: Full paper text (truncated)
        reviews: Optional list of existing reviews
        transcript: List of agent messages
        rounds: Number of debate rounds completed
        max_rounds: Maximum allowed rounds
        early_stop_phrases: Phrases that trigger early stopping
        start_time: Unix timestamp when processing started
        token_usage: Accumulated token usage per agent
    """
    paper_id: str
    title: str
    abstract: str
    full_text: str
    reviews: Optional[List[Dict[str, Any]]]
    transcript: List[Dict[str, Any]]
    rounds: int
    max_rounds: int
    early_stop_phrases: List[str]
    start_time: float
    token_usage: Dict[str, Dict[str, int]]


def create_initial_state(
    paper_id: str,
    paper: Dict[str, Any],
    cfg: Dict[str, Any],
) -> AgentState:
    """
    Create initial agent state for a paper.
    
    Args:
        paper_id: Unique identifier for the paper
        paper: Full paper dict from reviews_parsed.json
        cfg: Config dict loaded from config.yaml
        
    Returns:
        Initial AgentState
    """
    truncate_chars = cfg["agent"].get("truncate_body_chars", 12000)
    title = paper.get("title", paper_id)
    full_text = paper.get("body_text", paper.get("full_text", ""))
    paper_text = full_text[:truncate_chars] if full_text else paper.get("abstract", title)
    
    return AgentState(
        paper_id=paper_id,
        title=title,
        abstract=paper.get("abstract", ""),
        full_text=paper_text,
        reviews=paper.get("reviews"),
        transcript=[],
        rounds=0,
        max_rounds=cfg["agent"].get("max_rounds", 5),
        early_stop_phrases=cfg["agent"].get("early_stop_phrases", []),
        start_time=time.time(),
        token_usage={},
    )


def update_transcript(
    state: AgentState,
    role: str,
    content: str,
) -> AgentState:
    """
    Update state with a new transcript entry.
    
    Args:
        state: Current AgentState
        role: Agent role (reader, critic, auditor, summarizer)
        content: Message content
        
    Returns:
        Updated AgentState
    """
    state["transcript"].append({
        "role": role,
        "content": content,
        "timestamp": time.time(),
    })
    return state


def update_token_usage(
    state: AgentState,
    agent_name: str,
    input_tokens: int,
    output_tokens: int,
) -> AgentState:
    """
    Update state with token usage for an agent.
    
    Args:
        state: Current AgentState
        agent_name: Name of the agent
        input_tokens: Input token count
        output_tokens: Output token count
        
    Returns:
        Updated AgentState
    """
    if agent_name not in state["token_usage"]:
        state["token_usage"][agent_name] = {"input": 0, "output": 0}
    
    state["token_usage"][agent_name]["input"] += input_tokens
    state["token_usage"][agent_name]["output"] += output_tokens
    return state


def increment_rounds(state: AgentState) -> AgentState:
    """
    Increment the round counter.
    
    Args:
        state: Current AgentState
        
    Returns:
        Updated AgentState
    """
    state["rounds"] += 1
    return state


def should_early_stop(state: AgentState, message: str) -> bool:
    """
    Check if the debate should stop early based on early_stop_phrases.
    
    Args:
        state: Current AgentState
        message: Message to check for early stop phrases
        
    Returns:
        True if early stopping should occur
    """
    message_lower = message.lower()
    
    for phrase in state["early_stop_phrases"]:
        idx = message_lower.find(phrase)
        if idx >= 0:
            # Check for negation in context
            prefix = message_lower[max(0, idx - 15):idx]
            negations = ["not ", "no ", "don't ", "isn't ", "hardly "]
            if not any(neg in prefix for neg in negations):
                return True
    
    return False


def get_latency_seconds(state: AgentState) -> float:
    """
    Calculate elapsed time since state creation.
    
    Args:
        state: Current AgentState
        
    Returns:
        Elapsed time in seconds
    """
    return time.time() - state["start_time"]
