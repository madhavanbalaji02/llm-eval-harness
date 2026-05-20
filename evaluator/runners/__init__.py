"""LLM runners — async wrappers around provider SDKs."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class RunResult(BaseModel):
    """Raw output from a single LLM call."""

    response: str
    latency_ms: float
    ttft_ms: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    raw_finish_reason: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


class BaseRunner(ABC):
    """Abstract base class for LLM runners."""

    def __init__(self, model: str, api_key: Optional[str] = None, max_tokens: int = 512) -> None:
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    @abstractmethod
    async def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> RunResult:
        """Execute a single prompt and return structured results."""

    def _elapsed_ms(self, start: float) -> float:
        return (time.perf_counter() - start) * 1000


from .openai_runner import OpenAIRunner
from .anthropic_runner import AnthropicRunner
from .groq_runner import GroqRunner

__all__ = ["BaseRunner", "RunResult", "OpenAIRunner", "AnthropicRunner", "GroqRunner"]
