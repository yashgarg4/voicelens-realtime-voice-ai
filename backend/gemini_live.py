"""Thin async wrapper around the Gemini Live API WebSocket session.

The google-genai SDK exposes the Live API as an async context manager:

    async with client.aio.live.connect(model=..., config=...) as session:
        await session.send_realtime_input(audio=Blob(...))
        async for message in session.receive():
            ...

We wrap that in a small class with an explicit lifecycle (connect / send_audio
/ receive / close) so `main.py` can drive both directions as independent tasks
and tear down cleanly when the browser disconnects. `receive()` normalises the
SDK's rich message objects into flat, JSON-serialisable typed dicts so the rest
of the app never has to import google-genai types.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Literal, Optional, TypedDict

from google import genai
from google.genai import errors, types
from websockets.exceptions import ConnectionClosed

from .audio import pcm_to_base64
from .config import settings

logger = logging.getLogger("voicelens.gemini")


# --- Typed messages emitted by receive() -----------------------------------
class AudioMessage(TypedDict):
    type: Literal["audio"]
    data: str  # base64-encoded 24 kHz PCM16


class TextMessage(TypedDict):
    type: Literal["text"]
    text: str


class TranscriptMessage(TypedDict):
    type: Literal["input_transcript", "output_transcript"]
    text: str


class StatusMessage(TypedDict):
    type: Literal["turn_complete", "interrupted", "generation_complete", "setup_complete"]


GeminiMessage = AudioMessage | TextMessage | TranscriptMessage | StatusMessage


class GeminiLiveSession:
    """Manages a single full-duplex Gemini Live conversation."""

    def __init__(self, response_modalities: Optional[list[str]] = None) -> None:
        # v1beta is the API surface that serves the *-live-* models.
        self._client = genai.Client(
            api_key=settings.require_api_key(),
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self._response_modalities = response_modalities or ["AUDIO"]
        self._cm = None  # the async context manager returned by connect()
        self._session = None  # the live AsyncSession once entered

    async def connect(self, system_prompt: str) -> None:
        """Open the Live WebSocket and wait until the session is usable."""
        config = types.LiveConnectConfig(
            response_modalities=self._response_modalities,
            system_instruction=types.Content(
                parts=[types.Part(text=system_prompt)]
            ),
            # Ask Gemini to transcribe both sides — useful for the UI and, in
            # later phases, for routing answers through the fine-tuned Whisper.
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
        self._cm = self._client.aio.live.connect(
            model=settings.GEMINI_LIVE_MODEL, config=config
        )
        self._session = await self._cm.__aenter__()
        logger.info("Gemini Live session connected (model=%s)", settings.GEMINI_LIVE_MODEL)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Stream a chunk of raw 16 kHz PCM16 mic audio to Gemini."""
        if self._session is None:
            raise RuntimeError("send_audio called before connect()")
        await self._session.send_realtime_input(
            audio=types.Blob(
                data=pcm_bytes,
                mime_type=settings.input_audio_mime_type,  # audio/pcm;rate=16000
            )
        )

    async def send_text(self, text: str) -> None:
        """Send a text turn (used to kick off / steer the conversation)."""
        if self._session is None:
            raise RuntimeError("send_text called before connect()")
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True,
        )

    async def receive(self) -> AsyncGenerator[GeminiMessage, None]:
        """Yield normalised messages from Gemini for the whole conversation.

        The SDK's `session.receive()` generator ends after each `turn_complete`
        (one turn per call), so we re-enter it in an outer loop to keep the
        conversation alive across turns. The inner iteration blocks on the
        socket between turns, so there is no busy-loop. The loop ends only when
        the underlying Live connection actually closes (APIError / closed WS).
        """
        if self._session is None:
            raise RuntimeError("receive called before connect()")

        try:
            while True:
                async for message in self._session.receive():
                    if message.setup_complete is not None:
                        yield {"type": "setup_complete"}

                    content = message.server_content
                    if content is None:
                        continue

                    # Model audio (and any inline text) is in model_turn.parts.
                    if content.model_turn is not None:
                        for part in content.model_turn.parts or []:
                            inline = getattr(part, "inline_data", None)
                            if inline is not None and inline.data:
                                yield {
                                    "type": "audio",
                                    "data": pcm_to_base64(inline.data),
                                }
                            if getattr(part, "text", None):
                                yield {"type": "text", "text": part.text}

                    if content.input_transcription and content.input_transcription.text:
                        yield {
                            "type": "input_transcript",
                            "text": content.input_transcription.text,
                        }
                    if (
                        content.output_transcription
                        and content.output_transcription.text
                    ):
                        yield {
                            "type": "output_transcript",
                            "text": content.output_transcription.text,
                        }

                    if content.interrupted:
                        yield {"type": "interrupted"}
                    if content.generation_complete:
                        yield {"type": "generation_complete"}
                    if content.turn_complete:
                        yield {"type": "turn_complete"}
                # Turn finished; loop to await the next turn.
        except (errors.APIError, ConnectionClosed) as exc:
            logger.info("Gemini Live connection ended: %s", exc)
            return

    async def close(self) -> None:
        """Tear down the Live session cleanly (idempotent)."""
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Error closing Gemini Live session: %s", exc)
            finally:
                self._cm = None
                self._session = None
                logger.info("Gemini Live session closed")
