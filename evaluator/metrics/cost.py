"""Cost estimation: token counting and per-model pricing tables."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Pricing in USD per 1 million tokens (input / output)
PRICING_TABLE: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    # Anthropic
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # Groq (hosted open-source) — current active models
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.2-3b-preview": {"input": 0.06, "output": 0.06},
    "qwen/qwen3-32b": {"input": 0.29, "output": 0.59},
    # Groq legacy (decommissioned — kept for historical cost lookups)
    "llama3-70b-8192": {"input": 0.59, "output": 0.79},
    "llama3-8b-8192": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}

# Fallback pricing for unknown models
_DEFAULT_PRICING = {"input": 1.00, "output": 3.00}


def get_pricing(model: str) -> dict[str, float]:
    """Look up pricing for a model, with fuzzy prefix matching."""
    if model in PRICING_TABLE:
        return PRICING_TABLE[model]
    # Prefix match (e.g. "gpt-4o-mini-..." -> "gpt-4o-mini")
    for key, pricing in PRICING_TABLE.items():
        if model.startswith(key) or key.startswith(model):
            return pricing
    logger.warning("Unknown model %r — using default pricing $1/$3 per 1M tokens", model)
    return _DEFAULT_PRICING


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single API call.

    Args:
        model: Model identifier string.
        prompt_tokens: Number of input (prompt) tokens.
        completion_tokens: Number of output (completion) tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = get_pricing(model)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def count_tokens_tiktoken(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken (OpenAI tokenizer).

    Falls back to character-based estimate if tiktoken is unavailable.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)


def cost_efficiency_score(cost_usd: float, accuracy: float, max_cost: float = 0.01) -> float:
    """Composite score balancing cost and accuracy.

    Normalized so that a perfect-accuracy, zero-cost model scores 1.0.

    Args:
        cost_usd: Per-query cost in USD.
        accuracy: Accuracy metric in [0, 1].
        max_cost: Normalization ceiling for cost (USD). Queries above this score 0 on cost.

    Returns:
        Score in [0, 1].
    """
    cost_score = max(0.0, 1.0 - (cost_usd / max_cost))
    return 0.5 * accuracy + 0.5 * cost_score


def format_cost(cost_usd: float) -> str:
    """Format a cost value for human-readable display."""
    if cost_usd < 0.0001:
        return f"${cost_usd * 1_000_000:.2f}µ"
    if cost_usd < 0.01:
        return f"${cost_usd * 1000:.3f}m"
    return f"${cost_usd:.4f}"
