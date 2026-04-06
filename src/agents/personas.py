"""
personas.py
-----------
Agent role definitions for the multi-agent critique system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentRole(Enum):
    READER = "reader"
    CRITIC = "critic"
    AUDITOR = "auditor"
    SUMMARIZER = "summarizer"


@dataclass
class AgentMessage:
    role: AgentRole
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


class BaseAgent:
    def __init__(self, name: str, role: AgentRole, system_prompt: str,
                 model: str = "gemini-2.5-flash-lite", max_tokens: int = 4096,
                 temperature: float = 0.2, use_tools: bool = False,
                 max_tool_calls: int = 5, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_tools = use_tools
        self.max_tool_calls = max_tool_calls
        self.config = config or {}
        self.history: List[Dict[str, Any]] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def chat(self, user_message: str) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        self.history.clear()

    def get_token_usage(self) -> Dict[str, int]:
        return {"input_tokens": self.total_input_tokens, "output_tokens": self.total_output_tokens}
