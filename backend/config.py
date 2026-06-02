"""Central configuration for the VoiceLens backend.

All audio-format constants live here so every module agrees on the contract
that the Gemini Live API enforces. Getting these wrong is the single most common
cause of a "silent failure" (no error, just no audio coming back), so they are
defined in exactly one place and imported everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# The .env file lives in the project root (one level above this `backend/` dir).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    """Runtime configuration, read once at import time."""

    # --- Credentials -------------------------------------------------------
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()

    # --- Gemini Live model -------------------------------------------------
    # The live (full-duplex, streaming) variant. NOT the same as the standard
    # request/response `gemini-2.0-flash` model.
    GEMINI_LIVE_MODEL: str = os.getenv(
        "GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"
    )

    # Standard (text) model used to grade answers with structured output.
    GRADER_MODEL: str = os.getenv("GRADER_MODEL", "gemini-3.5-flash")

    # --- Audio format contract (do not change without reading the docs) ----
    # Gemini Live INPUT  : raw PCM, 16-bit signed LE, mono, 16 kHz.
    # Gemini Live OUTPUT : raw PCM, 16-bit signed LE, mono, 24 kHz.
    SAMPLE_RATE_IN: int = 16_000
    SAMPLE_RATE_OUT: int = 24_000
    CHANNELS: int = 1
    SAMPLE_WIDTH_BYTES: int = 2  # 16-bit signed integers

    # Audio chunk size in *samples* used for client-side capture buffering.
    CHUNK_SIZE: int = 1024

    # B2: also transcribe the candidate's answer with the fine-tuned Whisper
    # (requires torch/transformers installed; no-ops gracefully if not).
    ENABLE_FINETUNED_STT: bool = os.getenv(
        "ENABLE_FINETUNED_STT", "true"
    ).strip().lower() in ("1", "true", "yes")

    # --- Server / CORS -----------------------------------------------------
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

    @property
    def input_audio_mime_type(self) -> str:
        """MIME type string the Live API expects for streamed input audio."""
        return f"audio/pcm;rate={self.SAMPLE_RATE_IN}"

    def require_api_key(self) -> str:
        if not self.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add "
                "your key from https://aistudio.google.com/apikey"
            )
        return self.GOOGLE_API_KEY


settings = Settings()
