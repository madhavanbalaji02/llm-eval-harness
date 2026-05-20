"""Anthropic Messages API runner with streaming TTFT measurement."""

from __future__ import annotations

import os
import time
from typing import Optional

from . import BaseRunner, RunResult


class AnthropicRunner(BaseRunner):
    """Wraps Anthropic Messages API with streaming.

    Uses anthropic>=0.30 async streaming interface.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(model=model, api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), max_tokens=max_tokens)
        self.temperature = temperature
        self._client: Optional[object] = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> RunResult:
        client = self._get_client()

        start = time.perf_counter()
        ttft_ms: Optional[float] = None
        chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if text:
                    if ttft_ms is None:
                        ttft_ms = self._elapsed_ms(start)
                    chunks.append(text)

            final_message = await stream.get_final_message()
            if final_message.usage:
                prompt_tokens = final_message.usage.input_tokens or 0
                completion_tokens = final_message.usage.output_tokens or 0

        latency_ms = self._elapsed_ms(start)

        if prompt_tokens == 0:
            prompt_tokens = _estimate_tokens(prompt + (system_prompt or ""))
        if completion_tokens == 0:
            completion_tokens = _estimate_tokens("".join(chunks))

        return RunResult(
            response="".join(chunks),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
        )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)
