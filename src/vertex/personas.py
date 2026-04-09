"""
personas.py
-----------
Agent role definitions and Vertex AI client initialization for the
AI Research Paper Critique Assistant.

Roles
  Reader     – reads & summarises the paper section by section
  Critic     – proposes critique points based on the Reader's summary
  Auditor    – challenges the Critic's points and requests evidence
  Summarizer – consolidates the debate into a final structured review

This module provides:
- AgentRole enum for role identification
- Vertex AI client initialization
- BaseAgent class with configurable models per role
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from google.cloud import aiplatform
from google.cloud.aiplatform import initializer as aiplatform_initializer
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.cloud.aiplatform_v1.types import (
    Content,
    GenerateContentRequest,
    Part,
    Schema,
    Type,
)


class AgentRole(Enum):
    """Enum for agent roles in the multi-agent critique system."""
    READER = "reader"
    CRITIC = "critic"
    AUDITOR = "auditor"
    SUMMARIZER = "summarizer"


@dataclass
class AgentMessage:
    """Represents a message in the agent conversation transcript."""
    role: AgentRole
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


class BaseAgent:
    """
    Base agent class with Vertex AI integration.
    
    Each agent has a role, system prompt, and configurable model.
    The model can be specified per role (Flash for loops, Pro for final generation).
    """
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        system_prompt: str,
        model: str = "gemini-1.5-flash",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        use_tools: bool = False,
        max_tool_calls: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a BaseAgent with Vertex AI client.
        
        Args:
            name: Human-readable agent name
            role: AgentRole enum value for role identification
            system_prompt: System prompt defining agent behavior
            model: Vertex AI model name (default: gemini-1.5-flash)
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            use_tools: Whether to enable tool use
            max_tool_calls: Maximum tool calls per turn
            config: Optional config dict with vertex_ai settings
        """
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_tools = use_tools
        self.max_tool_calls = max_tool_calls
        self.config = config or {}
        
        # Vertex AI client
        self._client: Optional[PredictionServiceClient] = None
        
        # Conversation history
        self.history: List[Dict[str, Any]] = []
        
        # Token usage tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
    
    def _get_vertex_client(self) -> PredictionServiceClient:
        """Get or create Vertex AI PredictionServiceClient."""
        if self._client is None:
            project = self.config.get("vertex_ai", {}).get("project", "your-project-id")
            location = self.config.get("vertex_ai", {}).get("location", "us-central1")
            
            aiplatform.init(project=project, location=location)
            self._client = PredictionServiceClient(
                client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
            )
        return self._client
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a reply, updating internal history.
        
        Args:
            user_message: User message content
            
        Returns:
            Model response text
        """
        self.history.append({"role": "user", "content": user_message})
        client = self._get_vertex_client()
        
        # Build content for Vertex AI
        contents = [
            Content(
                role="user",
                parts=[Part(text=msg["content"]) for msg in self.history if msg["role"] == "user"],
            ),
            Content(
                role="model",
                parts=[Part(text=msg["content"]) for msg in self.history if msg["role"] == "model"],
            ),
        ]
        
        # Filter out None contents
        contents = [c for c in contents if c.parts]
        
        request = GenerateContentRequest(
            model=self.model,
            contents=contents,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        
        try:
            response = client.generate_content(request=request)
            
            # Extract text from response
            if response.candidates and response.candidates[0].content.parts:
                reply = response.candidates[0].content.parts[0].text
            else:
                reply = "[No response generated]"
            
            # Track token usage (approximate for now)
            if response.usage_metadata:
                self.total_input_tokens += response.usage_metadata.prompt_token_count
                self.total_output_tokens += response.usage_metadata.candidates_token_count
            
            self.history.append({"role": "model", "content": reply})
            return reply
            
        except Exception as e:
            print(f"  [ERROR] Vertex AI API call failed: {e}")
            return f"[Error: {str(e)}]"
    
    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get accumulated token usage."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }
