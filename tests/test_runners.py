"""Unit tests for LLM runners.

All API calls are mocked — no real API keys are needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluator.runners import RunResult


# ── Async iterator helpers ─────────────────────────────────────────────────────


def make_async_iter(items):
    """Return an async iterable that yields from *items*."""

    async def _gen():
        for item in items:
            yield item

    return _gen()


class AsyncContextManagerMock:
    """A reusable async context manager that wraps an async iterable."""

    def __init__(self, items, *, get_final_completion=None, get_final_message=None):
        self._items = items
        self._get_final_completion = get_final_completion
        self._get_final_message = get_final_message

    def __aiter__(self):
        return make_async_iter(self._items).__aiter__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get_final_completion(self):
        if self._get_final_completion:
            return self._get_final_completion()
        raise RuntimeError("no final completion configured")

    async def get_final_message(self):
        if self._get_final_message:
            return self._get_final_message()
        raise RuntimeError("no final message configured")

    @property
    def text_stream(self):
        """Anthropic-style text_stream attribute."""
        return make_async_iter(self._items)


# ── OpenAI Runner ──────────────────────────────────────────────────────────────


class TestOpenAIRunner:
    @pytest.fixture
    def runner(self):
        from evaluator.runners.openai_runner import OpenAIRunner

        return OpenAIRunner(model="gpt-4o-mini", api_key="sk-test", max_tokens=100)

    def _make_chunk(self, content: str, finish: str | None = None):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].finish_reason = finish
        return chunk

    def _make_final(self, prompt_tokens: int, completion_tokens: int):
        final = MagicMock()
        final.usage.prompt_tokens = prompt_tokens
        final.usage.completion_tokens = completion_tokens
        return final

    def test_default_model(self):
        from evaluator.runners.openai_runner import OpenAIRunner

        r = OpenAIRunner()
        assert r.model == "gpt-4o-mini"

    def test_max_tokens_stored(self, runner):
        assert runner.max_tokens == 100

    def test_api_key_stored(self, runner):
        assert runner.api_key == "sk-test"

    @pytest.mark.asyncio
    async def test_run_returns_run_result(self, runner):
        chunk = self._make_chunk("Paris", "stop")
        final = self._make_final(15, 5)
        stream = AsyncContextManagerMock([chunk], get_final_completion=lambda: final)
        runner._client = MagicMock()
        runner._client.chat.completions.stream = MagicMock(return_value=stream)

        result = await runner.run("What is the capital of France?")

        assert isinstance(result, RunResult)
        assert result.response == "Paris"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 5
        assert result.latency_ms >= 0
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_run_with_system_prompt(self, runner):
        chunk = self._make_chunk("Answer", "stop")
        final = self._make_final(20, 3)
        captured: dict = {}

        def capture_stream(**kwargs):
            captured.update(kwargs)
            return AsyncContextManagerMock([chunk], get_final_completion=lambda: final)

        runner._client = MagicMock()
        runner._client.chat.completions.stream = capture_stream

        await runner.run("Question", system_prompt="Be concise.")
        messages = captured["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_ttft_measured(self, runner):
        chunk1 = self._make_chunk("token", None)
        chunk2 = self._make_chunk("", "stop")
        final = self._make_final(10, 1)
        stream = AsyncContextManagerMock([chunk1, chunk2], get_final_completion=lambda: final)
        runner._client = MagicMock()
        runner._client.chat.completions.stream = MagicMock(return_value=stream)

        result = await runner.run("prompt")
        assert result.ttft_ms is not None
        assert result.ttft_ms >= 0

    @pytest.mark.asyncio
    async def test_multi_chunk_response(self, runner):
        chunks = [
            self._make_chunk("The ", None),
            self._make_chunk("Eiffel ", None),
            self._make_chunk("Tower", "stop"),
        ]
        final = self._make_final(12, 3)
        stream = AsyncContextManagerMock(chunks, get_final_completion=lambda: final)
        runner._client = MagicMock()
        runner._client.chat.completions.stream = MagicMock(return_value=stream)

        result = await runner.run("Famous Paris landmark?")
        assert result.response == "The Eiffel Tower"


# ── Anthropic Runner ───────────────────────────────────────────────────────────


class TestAnthropicRunner:
    @pytest.fixture
    def runner(self):
        from evaluator.runners.anthropic_runner import AnthropicRunner

        return AnthropicRunner(model="claude-haiku-4-5-20251001", api_key="sk-ant-test")

    def _make_final_message(self, input_tokens: int, output_tokens: int):
        msg = MagicMock()
        msg.usage.input_tokens = input_tokens
        msg.usage.output_tokens = output_tokens
        return msg

    def test_default_model(self):
        from evaluator.runners.anthropic_runner import AnthropicRunner

        r = AnthropicRunner()
        assert "claude" in r.model

    @pytest.mark.asyncio
    async def test_run_returns_run_result(self, runner):
        final_msg = self._make_final_message(20, 10)
        stream = AsyncContextManagerMock(
            ["The", " answer"],
            get_final_message=lambda: final_msg,
        )
        runner._client = MagicMock()
        runner._client.messages.stream = MagicMock(return_value=stream)

        result = await runner.run("Hello")
        assert isinstance(result, RunResult)
        assert "The" in result.response
        assert " answer" in result.response
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 10

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self, runner):
        final_msg = self._make_final_message(10, 5)
        captured: dict = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return AsyncContextManagerMock(["ok"], get_final_message=lambda: final_msg)

        runner._client = MagicMock()
        runner._client.messages.stream = capture

        await runner.run("Q", system_prompt="Sys")
        assert captured.get("system") == "Sys"

    @pytest.mark.asyncio
    async def test_ttft_recorded_on_first_text(self, runner):
        final_msg = self._make_final_message(10, 5)
        stream = AsyncContextManagerMock(
            ["first", " second"],
            get_final_message=lambda: final_msg,
        )
        runner._client = MagicMock()
        runner._client.messages.stream = MagicMock(return_value=stream)

        result = await runner.run("prompt")
        assert result.ttft_ms is not None
        assert result.ttft_ms >= 0


# ── Groq Runner ────────────────────────────────────────────────────────────────


class TestGroqRunner:
    @pytest.fixture
    def runner(self):
        from evaluator.runners.groq_runner import GroqRunner

        return GroqRunner(model="llama3-70b-8192", api_key="gsk_test")

    def _make_chunk(self, content: str, with_usage: bool = False):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        if with_usage:
            chunk.x_groq = MagicMock()
            chunk.x_groq.usage = MagicMock()
            chunk.x_groq.usage.prompt_tokens = 12
            chunk.x_groq.usage.completion_tokens = 2
        else:
            chunk.x_groq = None
        return chunk

    def test_default_model(self):
        from evaluator.runners.groq_runner import GroqRunner

        r = GroqRunner()
        assert r.model == "llama3-70b-8192"

    @pytest.mark.asyncio
    async def test_run_returns_run_result(self, runner):
        chunks = [self._make_chunk("42"), self._make_chunk("", with_usage=True)]

        async def fake_stream():
            for c in chunks:
                yield c

        runner._client = AsyncMock()
        runner._client.chat.completions.create = AsyncMock(return_value=fake_stream())

        result = await runner.run("What is 6x7?")
        assert isinstance(result, RunResult)
        assert result.model == "llama3-70b-8192"
        assert "42" in result.response

    @pytest.mark.asyncio
    async def test_groq_token_counts_from_usage_chunk(self, runner):
        chunks = [self._make_chunk("Answer"), self._make_chunk("", with_usage=True)]

        async def fake_stream():
            for c in chunks:
                yield c

        runner._client = AsyncMock()
        runner._client.chat.completions.create = AsyncMock(return_value=fake_stream())

        result = await runner.run("Q")
        assert result.prompt_tokens == 12
        assert result.completion_tokens == 2


# ── RunResult ──────────────────────────────────────────────────────────────────


class TestRunResult:
    def test_basic_construction(self):
        r = RunResult(
            response="hello",
            latency_ms=123.4,
            ttft_ms=55.0,
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4o-mini",
        )
        assert r.response == "hello"
        assert r.latency_ms == pytest.approx(123.4)
        assert r.ttft_ms == pytest.approx(55.0)

    def test_optional_fields_default(self):
        r = RunResult(response="ok", latency_ms=0.0)
        assert r.ttft_ms is None
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.model == ""
