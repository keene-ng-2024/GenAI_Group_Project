"""
grounding_verifier.py
---------------------
Grounding verification for critique points using Vertex AI.

This module provides:
- verify_grounding(): Verify a single critique point
- verify_all_grounding(): Batch verification of multiple critique points
- Grounding score calculation with LLM sub-calls
"""

from __future__ import annotations

import re
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from vertex_client import get_vertex_ai_client


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class GroundingResult:
    """Result of grounding verification for a single critique point."""
    is_supported: bool
    confidence: float
    evidence_match_score: float
    supporting_evidence: Optional[str] = None


# ── Core verification ────────────────────────────────────────────────────────

def verify_grounding(
    critique_point: Dict[str, str],
    paper_section: str,
    config: Optional[Dict[str, Any]] = None,
) -> GroundingResult:
    """
    Verify if a critique point is supported by evidence in the paper.
    
    Args:
        critique_point: Dict with "point" and "evidence" fields
        paper_section: Relevant section of the paper text
        config: Config dict with vertex_ai settings
        
    Returns:
        GroundingResult with is_supported, confidence, and evidence_match_score
    """
    if config is None:
        config = load_config()
    
    vertex_config = config.get("vertex_ai", {})
    model = vertex_config.get("grounding_verifier", {}).get("model", "gemini-2.5-flash-lite")
    max_tokens = vertex_config.get("grounding_verifier", {}).get("max_tokens", 1024)
    
    point = critique_point.get("point", "")
    evidence = critique_point.get("evidence", "")
    
    paper_section_truncated = paper_section[:2000]
    prompt = f"""Analyze whether the critique point is supported by the provided paper section.

Critique Point: {point}
Claimed Evidence: {evidence}

Paper Section:
{paper_section_truncated}

Please evaluate:
1. Is the claimed evidence actually present in the paper section?
2. Does the evidence actually support the critique point?
3. What is your confidence level (0-1)?

Respond with JSON:
{{
  "is_supported": true/false,
  "confidence": 0.0-1.0,
  "evidence_match_score": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
    
    client = get_vertex_ai_client(config=config)
    
    try:
        response = client.generate_content(
            prompt=prompt,
            model_name=model,
            max_tokens=max_tokens,
        )
        
        # Parse response
        text = response.get("text", "")
        
        # Try to parse JSON
        try:
            result = _parse_verification_response(text)
        except Exception:
            # Fallback parsing
            result = _fallback_parse_verification(text)
        
        return GroundingResult(
            is_supported=result.get("is_supported", False),
            confidence=result.get("confidence", 0.0),
            evidence_match_score=result.get("evidence_match_score", 0.0),
            supporting_evidence=result.get("reasoning"),
        )
        
    except Exception as e:
        print(f"  [ERROR] Grounding verification failed: {e}")
        return GroundingResult(
            is_supported=False,
            confidence=0.0,
            evidence_match_score=0.0,
        )


def _parse_verification_response(text: str) -> Dict[str, Any]:
    """Parse JSON response from grounding verification."""
    # Try to find JSON in the response
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        import json
        return json.loads(match.group())
    raise ValueError("No JSON found in response")


def _fallback_parse_verification(text: str) -> Dict[str, Any]:
    """Fallback parsing for grounding verification response."""
    result = {
        "is_supported": False,
        "confidence": 0.0,
        "evidence_match_score": 0.0,
    }
    
    text_lower = text.lower()
    
    # Check for explicit indicators
    if "supported" in text_lower and "not supported" not in text_lower:
        result["is_supported"] = True
    
    # Try to extract confidence score
    confidence_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', text_lower)
    if confidence_match:
        result["confidence"] = min(1.0, float(confidence_match.group(1)))
    
    # Try to extract evidence match score
    match_match = re.search(r'match[:\s]+(\d+(?:\.\d+)?)', text_lower)
    if match_match:
        result["evidence_match_score"] = min(1.0, float(match_match.group(1)))
    
    return result


# ── Batch verification ───────────────────────────────────────────────────────

def verify_all_grounding(
    critique_input: Any,
    paper: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Verify grounding for all critique points.

    Args:
        critique_input: Either a list of weakness dicts ({"point": ..., "evidence": ...})
                        from the structured Summarizer output, or a raw text string
                        (falls back to line-based extraction for backward compat).
        paper: Paper dict with full_text
        config: Config dict with vertex_ai settings

    Returns:
        Dict with aggregate grounding scores
    """
    if config is None:
        config = load_config()

    # Fast path: structured weaknesses list passed directly
    if isinstance(critique_input, list):
        critique_points = []
        for item in critique_input:
            if isinstance(item, dict):
                critique_points.append({
                    "point": item.get("point", ""),
                    "evidence": item.get("evidence", ""),
                })
            elif isinstance(item, str):
                critique_points.append({"point": item, "evidence": ""})
    else:
        # Legacy: raw text — extract via line-based heuristic
        critique_points = _extract_critique_points(str(critique_input))
    
    total_confidence = 0.0
    supported_count = 0
    points_verified = 0
    
    for point in critique_points:
        # Find relevant paper section (simplified: use full text)
        paper_section = paper.get("full_text", "")[:5000]
        
        result = verify_grounding(point, paper_section, config)
        
        total_confidence += result.confidence
        if result.is_supported:
            supported_count += 1
        points_verified += 1
    
    # Calculate aggregates
    if points_verified > 0:
        avg_confidence = total_confidence / points_verified
        grounding_rate = supported_count / points_verified
    else:
        avg_confidence = 0.0
        grounding_rate = 0.0
    
    return {
        "avg_confidence": round(avg_confidence, 4),
        "grounding_rate": round(grounding_rate, 4),
        "points_verified": points_verified,
        "points_unsupported": points_verified - supported_count,
    }


def _extract_critique_points(critique_text: str) -> List[Dict[str, str]]:
    """Extract individual critique points from critique text."""
    points = []
    
    # Split by common separators
    lines = critique_text.split('\n')
    
    current_point = {"point": "", "evidence": ""}
    
    for line in lines:
        line = line.strip()
        
        # Check for point markers
        if re.match(r'^[-*]\s+', line) or re.match(r'^\d+\.\s+', line):
            # Save previous point if exists
            if current_point["point"]:
                points.append(current_point.copy())
            
            # Start new point
            current_point = {"point": line, "evidence": ""}
        elif line and current_point["point"]:
            # This is evidence for the current point
            if current_point["evidence"]:
                current_point["evidence"] += " " + line
            else:
                current_point["evidence"] = line
    
    # Save last point
    if current_point["point"]:
        points.append(current_point.copy())
    
    return points


# ── Early stopping detection ─────────────────────────────────────────────────

def should_stop_debate(
    audit_feedback: str,
    early_stop_phrases: List[str],
) -> bool:
    """
    Check if the debate should stop early based on auditor feedback.
    
    Args:
        audit_feedback: Auditor's feedback message
        early_stop_phrases: List of phrases that trigger early stopping
        
    Returns:
        True if early stopping should occur
    """
    feedback_lower = audit_feedback.lower()
    
    for phrase in early_stop_phrases:
        idx = feedback_lower.find(phrase)
        if idx >= 0:
            # Check for negation in context
            prefix = feedback_lower[max(0, idx - 15):idx]
            negations = ["not ", "no ", "don't ", "isn't ", "hardly "]
            if not any(neg in prefix for neg in negations):
                return True
    
    return False
