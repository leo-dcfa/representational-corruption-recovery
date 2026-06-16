"""Convert TrainExample corpora into a chat dataset for the SFT trainer."""

from __future__ import annotations

from rcr.datagen.schema import TrainExample
from rcr.utils.io import read_jsonl


def examples_to_messages(examples: list[TrainExample]) -> list[dict]:
    return [{"messages": ex.to_chat(), "id": ex.id, "domain": ex.domain} for ex in examples]


def load_corpus_dataset(path: str):
    """Load a corpus JSONL into a datasets.Dataset of {messages, id, domain}."""
    from datasets import Dataset

    examples = [TrainExample.from_dict(d) for d in read_jsonl(path)]
    return Dataset.from_list(examples_to_messages(examples))
