"""Groq inference runner — OpenAI-compatible API with ultra-fast inference."""

from __future__ import annotations

import os
import time
from typing import Optional

from . import BaseRunner, RunResult


class GroqRunner(BaseRunner):
    """Wraps Groq's OpenAI-compatible API.

    Groq provides dramatically faster inference via its LPU hardware, making it
    an excellent latency baseline for benchmarking other providers.
    """

    def __init__(
        self,
        model: str = "llama3-70b-8192",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(model=model, api_key=api_key or os.getenv("GROQ_API_KEY"), max_tokens=max_tokens)
        self.temperature = temperature
        self._client: Optional[object] = None

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self.api_key)
        return self._client

    async def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> RunResult:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        ttft_ms: Optional[float] = None
        chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                if ttft_ms is None:
                    ttft_ms = self._elapsed_ms(start)
                chunks.append(content)
            # Groq sends usage in the last chunk (x_groq.usage)
            if hasattr(chunk, "x_groq") and chunk.x_groq and hasattr(chunk.x_groq, "usage"):
                usage = chunk.x_groq.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        latency_ms = self._elapsed_ms(start)

        # Fallback token counting
        if prompt_tokens == 0:
            prompt_tokens = _count_tokens_tiktoken(messages)
        if completion_tokens == 0:
            completion_tokens = _count_tokens_tiktoken([{"content": "".join(chunks)}])

        return RunResult(
            response="".join(chunks),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
        )


def _count_tokens_tiktoken(messages: list[dict]) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        text = " ".join(m.get("content", "") for m in messages)
        return len(enc.encode(text))
    except Exception:
        text = " ".join(m.get("content", "") for m in messages)
        return max(1, len(text.split()) * 4 // 3)
