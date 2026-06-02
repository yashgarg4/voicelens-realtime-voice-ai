"""Shared helpers for the fine-tuning scripts.

Importing this module pulls in heavy ML deps (transformers/torch), so it is used
ONLY by the finetune/ scripts (run on a GPU / Colab), never by the FastAPI
backend. Run the scripts from the project root, e.g.:

    python finetune/prepare_dataset.py
    python finetune/train.py
    python finetune/evaluate.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the finetune/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml as a nested dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_like: str) -> Path:
    """Resolve a config path (relative ones are relative to the project root)."""
    p = Path(path_like)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads Whisper input features and label sequences to batch shape.

    Extra dataset columns (e.g. the raw "reference" text we keep for eval) are
    ignored here — only `input_features` and `labels` are consumed.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch  # local import: keeps module import light until used

        # Audio features are already fixed-length log-mels, but pad anyway.
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        # Replace padding with -100 so it is ignored by the loss.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # If the BOS/decoder-start token was prepended during tokenisation, drop
        # it — the model adds it itself.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        del torch
        return batch
