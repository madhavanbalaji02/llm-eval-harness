"""Latency metrics: p50/p95/p99, mean, tokens/sec, TTFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LatencyStats:
    """Percentile-based latency statistics over N calls."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    n: int
    mean_ttft_ms: Optional[float] = None
    mean_tokens_per_sec: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "n": self.n,
            "mean_ttft_ms": round(self.mean_ttft_ms, 2) if self.mean_ttft_ms else None,
            "mean_tokens_per_sec": round(self.mean_tokens_per_sec, 2) if self.mean_tokens_per_sec else None,
        }


def compute_latency_stats(
    latencies_ms: list[float],
    ttft_ms: Optional[list[float]] = None,
    tokens_per_sec: Optional[list[float]] = None,
) -> LatencyStats:
    """Compute percentile latency statistics from a list of wall-clock measurements.

    Args:
        latencies_ms: List of end-to-end latency values in milliseconds.
        ttft_ms: Optional list of time-to-first-token values.
        tokens_per_sec: Optional list of tokens/second measurements.

    Returns:
        LatencyStats with p50/p95/p99 and ancillary stats.
    """
    if not latencies_ms:
        raise ValueError("latencies_ms must be non-empty")

    arr = np.array(latencies_ms, dtype=float)
    mean_ttft = float(np.mean(ttft_ms)) if ttft_ms else None
    mean_tps = float(np.mean(tokens_per_sec)) if tokens_per_sec else None

    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(np.mean(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        std_ms=float(np.std(arr)),
        n=len(latencies_ms),
        mean_ttft_ms=mean_ttft,
        mean_tokens_per_sec=mean_tps,
    )


def compute_tokens_per_second(completion_tokens: int, latency_ms: float) -> Optional[float]:
    """Convert completion token count and latency into tokens/second."""
    if latency_ms <= 0 or completion_tokens <= 0:
        return None
    return (completion_tokens / latency_ms) * 1000.0
