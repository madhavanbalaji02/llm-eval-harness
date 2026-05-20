"""Dataset loader: reads JSONL files into typed DatasetItem objects."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class DatasetItem(BaseModel):
    """A single evaluation example.

    Fields:
        id: Unique identifier for this item.
        question: The question/prompt to send to the model.
        ground_truth: The reference answer for accuracy comparison.
        context: Background passage used for faithfulness / RAGAS evaluation.
        metadata: Arbitrary extra fields (topic, difficulty, source, etc.).
    """

    id: str
    question: str
    ground_truth: str
    context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question", "ground_truth")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question and ground_truth must be non-empty")
        return v.strip()

    @field_validator("context")
    @classmethod
    def strip_context(cls, v: str) -> str:
        return v.strip()


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_no, exc)


def load_dataset(
    path: str | Path,
    max_items: Optional[int] = None,
    filter_topic: Optional[str] = None,
) -> list[DatasetItem]:
    """Load a JSONL dataset file into a list of DatasetItem objects.

    Args:
        path: Path to the .jsonl file.
        max_items: Optional limit on the number of items to load.
        filter_topic: If set, only include items whose metadata.topic matches.

    Returns:
        List of validated DatasetItem objects.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the file contains no valid items.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    items: list[DatasetItem] = []
    skipped = 0

    for raw in _iter_jsonl(path):
        try:
            item = DatasetItem.model_validate(raw)
        except Exception as exc:
            logger.warning("Skipping invalid item (id=%s): %s", raw.get("id", "?"), exc)
            skipped += 1
            continue

        if filter_topic and item.metadata.get("topic") != filter_topic:
            continue

        items.append(item)

        if max_items and len(items) >= max_items:
            break

    if skipped:
        logger.warning("Skipped %d invalid items during load", skipped)

    if not items:
        raise ValueError(f"No valid items found in {path}")

    logger.info("Loaded %d items from %s", len(items), path)
    return items


def save_dataset(items: list[DatasetItem], path: str | Path) -> None:
    """Write DatasetItem objects to a JSONL file.

    Useful for creating subsets or synthetic datasets.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    logger.info("Saved %d items to %s", len(items), path)
