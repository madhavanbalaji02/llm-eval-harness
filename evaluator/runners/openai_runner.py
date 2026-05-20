"""OpenAI chat completions runner with streaming TTFT measurement."""

from __future__ import annotations

import os
import time
from typing import Optional

from . import BaseRunner, RunResult


class OpenAIRunner(BaseRunner):
    """Wraps OpenAI chat completions API with streaming for accurate TTFT.

    Uses openai>=1.35 async streaming interface.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(model=model, api_key=api_key or os.getenv("OPENAI_API_KEY"), max_tokens=max_tokens)
        self.temperature = temperature
        self._client: Optional[object] = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)
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
        finish_reason: Optional[str] = None

        async with client.chat.completions.stream(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) as stream:
            async for event in stream:
                # event is a ChatCompletionChunk
                if not hasattr(event, "choices") or not event.choices:
                    continue
                delta = event.choices[0].delta
                content = delta.content or ""
                if content:
                    if ttft_ms is None:
                        ttft_ms = self._elapsed_ms(start)
                    chunks.append(content)
                if event.choices[0].finish_reason:
                    finish_reason = event.choices[0].finish_reason

            # Retrieve token usage from the final completion
            try:
                final = await stream.get_final_completion()
                if final.usage:
                    prompt_tokens = final.usage.prompt_tokens or 0
                    completion_tokens = final.usage.completion_tokens or 0
            except Exception:
                pass

        latency_ms = self._elapsed_ms(start)

        # Fallback token counting via tiktoken
        if prompt_tokens == 0 or completion_tokens == 0:
            prompt_tokens, completion_tokens = _count_tokens_tiktoken(
                self.model, messages, "".join(chunks)
            )

        return RunResult(
            response="".join(chunks),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
            raw_finish_reason=finish_reason,
        )


def _count_tokens_tiktoken(model: str, messages: list[dict], response: str) -> tuple[int, int]:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens = len(enc.encode(prompt_text))
        completion_tokens = len(enc.encode(response))
        return prompt_tokens, completion_tokens
    except Exception:
        prompt_text = " ".join(m.get("content", "") for m in messages)
        return len(prompt_text.split()) * 4 // 3, len(response.split()) * 4 // 3
