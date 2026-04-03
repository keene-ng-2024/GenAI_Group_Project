"""
data_models.py
--------------
Core data models for the AI Research Paper Critique Assistant.

This module defines Pydantic-compatible dataclasses for:
- Paper: Input paper representation
- CritiquePoint: Individual critique point with evidence
- StructuredReview: Final structured output format
- CritiqueResult: Complete critique result with metadata
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Paper:
    """
    Represents a research paper to be critiqued.
    
    Attributes:
        paper_id: Unique identifier for the paper
        title: Paper title
        abstract: Paper abstract
        full_text: Full paper text (may be truncated)
        reviews: Optional list of existing reviews for context
    """
    paper_id: str
    title: str
    abstract: str
    full_text: str
    reviews: Optional[List[Dict[str, Any]]] = None


@dataclass
class CritiquePoint:
    """
    Represents a single critique point with supporting evidence.
    
    Attributes:
        point: The critique point or claim
        evidence: Supporting evidence from the paper
        low_confidence: Whether this point has low confidence
    """
    point: str
    evidence: str
    low_confidence: bool = False


@dataclass
class StructuredReview:
    """
    Represents the final structured review output.
    
    Attributes:
        summary: 2-3 sentence summary of the paper
        strengths: List of strengths with evidence
        weaknesses: List of weaknesses with evidence
        questions: List of questions for authors with motivation
        scores: Review scores dictionary
    """
    summary: str
    strengths: List[CritiquePoint]
    weaknesses: List[CritiquePoint]
    questions: List[Dict[str, str]]
    scores: Dict[str, Any]


@dataclass
class AgentMessage:
    """
    Represents a message in the agent conversation transcript.
    
    Attributes:
        role: Agent role (reader, critic, auditor, summarizer)
        content: Message content
        timestamp: Unix timestamp when message was created
        tool_calls: Optional list of tool calls made
        tool_results: Optional list of tool call results
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class CritiqueResult:
    """
    Represents the complete critique result for a paper.
    
    Attributes:
        paper_id: Unique identifier for the paper
        model: Model name used for generation
        rounds: Number of debate rounds executed
        latency_seconds: Total execution time in seconds
        token_usage: Token usage dictionary with input/output counts
        transcript: List of all agent messages
        structured: Structured review output
        critique_points: Flat dictionary of critique points for evaluation
        grounding_verifier_scores: Grounding verification metrics
        run_metadata: Additional run metadata
    """
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
