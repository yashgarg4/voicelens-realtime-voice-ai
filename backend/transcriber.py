"""Whisper transcription wrapper (B2 layer).

Loads Whisper-small and, if present, the QLoRA fine-tuned adapter from
finetune/output/adapter, transcribing 16 kHz PCM16 audio to text. If the adapter
is missing or fails to load, it falls back to the base model — so the app keeps
working before any fine-tuning is done.

IMPORTANT: torch/transformers/peft are imported lazily inside `load()`, NOT at
module import time. That keeps the FastAPI backend importable in the lightweight
voice-only environment (which has no torch). Only call `load()` where those
heavy deps are installed. This wrapper is not wired into the live Gemini flow
yet; it exists to demonstrate the fine-tuned model and back the WER comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import settings

logger = logging.getLogger("voicelens.transcriber")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ADAPTER = _PROJECT_ROOT / "finetune" / "output" / "adapter"


class WhisperTranscriber:
    """Lazily-loaded Whisper transcriber with optional fine-tuned adapter."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        adapter_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.adapter_dir = Path(adapter_dir) if adapter_dir else _DEFAULT_ADAPTER
        self._model = None
        self._processor = None
        self._device = "cpu"
        self.using_adapter = False

    def load(self) -> "WhisperTranscriber":
        """Load model + processor (and adapter if available). Idempotent."""
        if self._model is not None:
            return self

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        self._processor = WhisperProcessor.from_pretrained(
            self.model_name, language="english", task="transcribe"
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=dtype
        )

        # Try to attach the fine-tuned adapter; fall back to base on any error.
        if self.adapter_dir.exists():
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(self.adapter_dir))
                self.using_adapter = True
                logger.info("Loaded fine-tuned adapter from %s", self.adapter_dir)
            except Exception as exc:
                logger.warning(
                    "Failed to load adapter (%s); using base model.", exc
                )
        else:
            logger.info(
                "No adapter at %s; using base %s.", self.adapter_dir, self.model_name
            )

        self._model = model.to(self._device).eval()
        return self

    def transcribe_pcm(self, pcm_bytes: bytes) -> str:
        """Transcribe raw 16 kHz mono PCM16 bytes to text."""
        import numpy as np

        # PCM16 -> float32 in [-1, 1] at the rate Whisper expects.
        audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        return self.transcribe_array(audio, sample_rate=settings.SAMPLE_RATE_IN)

    def transcribe_array(self, audio, sample_rate: int = 16_000) -> str:
        """Transcribe a float32 mono waveform (any rate; resampled to 16 kHz)."""
        import torch

        if self._model is None:
            self.load()

        features = self._processor.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self._device, dtype=self._model.dtype)

        forced = self._processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )
        with torch.no_grad():
            generated = self._model.generate(
                input_features=features, forced_decoder_ids=forced, max_new_tokens=128
            )
        return self._processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()


# Lazily-shared singleton (only constructed; not loaded until first use).
_transcriber: Optional[WhisperTranscriber] = None


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber
