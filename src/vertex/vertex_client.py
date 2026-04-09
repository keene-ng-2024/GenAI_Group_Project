"""
vertex_client.py
----------------
Vertex AI client using the google-genai SDK (avoids gRPC hang on import).

Uses google.genai.Client with vertexai=True — no vertexai.init() needed.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    failures: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: Optional[float] = None
    half_open_calls: int = 0

    def record_success(self) -> None:
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def is_allowed(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        return False


class RateLimiter:
    def __init__(self, rate: float = 10.0, burst: int = 20):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.time()

    def acquire(self) -> None:
        while True:
            now = time.time()
            self.tokens = min(self.burst, self.tokens + (now - self.last_update) * self.rate)
            self.last_update = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            time.sleep(0.1)


class VertexAIClient:
    """Vertex AI client using google-genai SDK (no gRPC hang)."""

    def __init__(self, project: str, location: str, config: dict):
        self.project = project
        self.location = location
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("failure_threshold", 5),
            recovery_timeout=config.get("recovery_timeout", 30.0),
        )
        self.rate_limiter = RateLimiter(
            rate=config.get("rate_limit", 5.0),
            burst=config.get("burst", 10),
        )
        self.model_map = config.get("models", {})
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.genai as genai
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                )
            except ImportError:
                raise ImportError("google-genai not installed. Run: pip install google-genai")
        return self._client

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.circuit_breaker.is_allowed():
            raise RuntimeError("Circuit breaker is open")

        self.rate_limiter.acquire()

        if model_name is None:
            model_name = self.model_map.get("default", "gemini-2.5-flash-lite")

        import google.genai.types as types

        client = self._get_client()

        gen_config = {}
        if max_tokens:
            gen_config["max_output_tokens"] = max_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature

        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        # Retry with exponential backoff on 429 rate-limit errors
        max_retries = 6
        backoff = 30.0
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(**gen_config) if gen_config else None,
                )
                self.circuit_breaker.record_success()
                usage = response.usage_metadata
                return {
                    "text": response.text,
                    "input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                    "output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
                }
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        wait = backoff * (2 ** attempt)
                        print(f"  [429] Rate limited. Waiting {wait:.0f}s before retry {attempt + 1}/{max_retries - 1}…")
                        time.sleep(wait)
                        continue
                self.circuit_breaker.record_failure()
                raise


def get_vertex_ai_client(config: dict) -> VertexAIClient:
    vertex_config = config.get("vertex_ai", {})
    return VertexAIClient(
        project=vertex_config.get("project", "genai-vertexai-492302"),
        location=vertex_config.get("location", "us-central1"),
        config=vertex_config,
    )
