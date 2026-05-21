"""Top-level worker functions for HuggingFace model inference.

These run inside a ProcessPoolExecutor (spawn mode on macOS), so they start in
a clean process with no inherited mutex state — avoiding the MPS/OMP deadlocks
that occur when HF models are loaded inside a forked/threaded asyncio context.

All functions must be picklable (module-level, no closures).
"""

import os
# Must be set before any HuggingFace import in this worker process.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ── Module-level singletons (loaded once per worker process) ──────────────────

_sentence_model = None
_nli_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _sentence_model


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", device="cpu")
    return _nli_model


# ── Public worker functions (picklable, called via ProcessPoolExecutor) ────────


def compute_semantic_similarity_worker(prediction: str, reference: str) -> float:
    """Cosine similarity between sentence embeddings. Runs in a worker process."""
    import numpy as np

    model = _get_sentence_model()
    embeddings = model.encode(
        [prediction, reference],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, min(1.0, similarity))


_NLI_LABELS = ["contradiction", "entailment", "neutral"]
_NLI_THRESHOLD = 0.5


def check_nli_worker(context: str, answer: str) -> tuple[str | None, float | None, bool]:
    """NLI hallucination check. Runs in a worker process."""
    if not context or not answer:
        return None, None, False
    try:
        import numpy as np

        model = _get_nli_model()
        scores = model.predict([(context, answer)])
        arr = scores[0].astype(float)
        arr -= arr.max()
        import math
        exp_arr = [math.exp(x) for x in arr]
        total = sum(exp_arr)
        probs = [x / total for x in exp_arr]
        label_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        label = _NLI_LABELS[label_idx]
        score = float(probs[label_idx])
        is_hall = label == "contradiction" and score >= _NLI_THRESHOLD
        return label, score, is_hall
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("NLI worker failed: %s", exc)
        return None, None, False


def evaluate_ragas_batch_worker(samples: list) -> list:
    """RAGAS evaluation. Runs in a worker process (avoids asyncio/MPS conflict)."""
    from evaluator.metrics.ragas_metrics import evaluate_ragas_batch
    return evaluate_ragas_batch(samples)
