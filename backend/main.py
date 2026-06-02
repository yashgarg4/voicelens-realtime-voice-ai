"""VoiceLens FastAPI backend.

Exposes:
  GET  /api/health             — liveness probe + key/config status
  GET  /api/questions          — the AI-engineer question bank
  POST /api/session/start      — begin a session (pick a question)
  POST /api/session/end        — end a session
  GET  /api/sessions           — past sessions + scores (for the dashboard)
  GET  /api/wer                — base vs fine-tuned Whisper WER comparison
  WS   /ws/session             — full-duplex audio relay between browser+Gemini

Relay protocol on /ws/session:
  Browser -> Backend : binary frames of raw 16 kHz PCM16 mic audio.
  Backend -> Browser : JSON text frames (audio base64 under "data"), plus a
                       {"type":"feedback", "scores": {...}} frame when the
                       coach finishes an evaluation turn.

Query params on /ws/session:
  question_id (int, optional) — which question to interview on.
  session_id  (str, optional) — links parsed scores back to a started session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import coach, db, transcriber
from .audio import simple_vad
from .config import settings
from .gemini_live import GeminiLiveSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicelens.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Pre-warm the fine-tuned Whisper in the background so the first answer isn't
    # delayed by the ~25s model load. Non-blocking: the server starts immediately.
    if settings.ENABLE_FINETUNED_STT and transcriber.is_available():
        logger.info("Pre-warming fine-tuned Whisper (background)…")
        asyncio.create_task(asyncio.to_thread(transcriber.get_transcriber().load))
    yield


app = FastAPI(title="VoiceLens", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WER comparison written by finetune/evaluate.py (Phase 3).
_WER_RESULT_PATH = (
    Path(__file__).resolve().parent.parent / "finetune" / "output" / "wer_result.json"
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- REST models ------------------------------------------------------------
class StartSessionRequest(BaseModel):
    question_id: Optional[int] = None
    category: Optional[str] = None


class EndSessionRequest(BaseModel):
    session_id: str


# --- REST endpoints ---------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": settings.GEMINI_LIVE_MODEL,
        "google_api_key_configured": bool(settings.GOOGLE_API_KEY),
        "sample_rate_in": settings.SAMPLE_RATE_IN,
        "sample_rate_out": settings.SAMPLE_RATE_OUT,
        "questions": len(coach.QUESTIONS),
        "finetuned_stt": settings.ENABLE_FINETUNED_STT and transcriber.is_available(),
    }


@app.get("/api/questions")
async def list_questions() -> dict:
    return {"questions": [q.model_dump() for q in coach.QUESTIONS]}


@app.post("/api/session/start")
async def start_session(req: StartSessionRequest) -> dict:
    try:
        question = coach.get_question(req.question_id, req.category)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    session_id = uuid.uuid4().hex
    await asyncio.to_thread(
        db.create_session,
        session_id,
        question.id,
        question.category,
        question.question,
        _utcnow(),
    )
    logger.info("Session %s started on question %s", session_id, question.id)
    return {"session_id": session_id, "question": question.model_dump()}


@app.post("/api/session/end")
async def end_session(req: EndSessionRequest) -> dict:
    await asyncio.to_thread(db.end_session, req.session_id, _utcnow())
    logger.info("Session %s ended", req.session_id)
    return {"status": "ended"}


@app.get("/api/sessions")
async def get_sessions() -> dict:
    sessions = await asyncio.to_thread(db.list_sessions)
    return {"sessions": sessions}


@app.get("/api/wer")
async def get_wer() -> dict:
    """Return the base-vs-fine-tuned WER comparison from Phase 3, if available."""
    if _WER_RESULT_PATH.exists():
        try:
            return json.loads(_WER_RESULT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"available": False}


# --- WebSocket relay --------------------------------------------------------
@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket) -> None:
    await websocket.accept()

    # Resolve which question to interview on.
    raw_qid = websocket.query_params.get("question_id")
    session_id = websocket.query_params.get("session_id")
    try:
        question = coach.get_question(int(raw_qid)) if raw_qid else coach.get_question()
    except (KeyError, ValueError):
        question = coach.get_question()

    try:
        gemini = GeminiLiveSession(response_modalities=["AUDIO"])
        await gemini.connect(coach.build_system_prompt(question))
    except Exception as exc:
        logger.exception("Failed to start Gemini Live session")
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    await websocket.send_json(
        {"type": "status", "status": "connected", "question": question.model_dump()}
    )
    # Kick off the interview: the coach greets and asks the question.
    await gemini.send_text(coach.START_INTERVIEW_TRIGGER)

    # B2: capture the candidate's answer audio and re-transcribe it with the
    # fine-tuned Whisper so the UI can show it next to Gemini's transcript.
    stt_on = settings.ENABLE_FINETUNED_STT and transcriber.is_available()
    answer_audio = bytearray()  # raw 16 kHz PCM16 of the current user turn
    # ~0.6 s minimum so we don't transcribe stray clicks/silence.
    MIN_ANSWER_BYTES = int(0.6 * settings.SAMPLE_RATE_IN) * settings.SAMPLE_WIDTH_BYTES

    async def transcribe_answer(pcm: bytes) -> None:
        try:
            text = await asyncio.to_thread(
                transcriber.get_transcriber().transcribe_pcm, pcm
            )
            if text:
                await websocket.send_json(
                    {
                        "type": "finetuned_transcript",
                        "text": text,
                        "using_adapter": transcriber.get_transcriber().using_adapter,
                    }
                )
        except Exception:
            logger.exception("fine-tuned transcription failed")

    async def uplink() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    await gemini.send_audio(data)
                    if stt_on:
                        answer_audio.extend(data)
        except WebSocketDisconnect:
            logger.info("Browser disconnected (uplink)")
        except Exception:
            logger.exception("uplink task error")

    async def downlink() -> None:
        # Per-turn transcripts: the candidate's answer (input) is graded by the
        # structured LLM grader; the coach's spoken text (output) is the regex
        # fallback if grading fails.
        answer_text = ""
        coach_text = ""
        coach_speaking = False  # has the coach started replying this turn?
        try:
            async for msg in gemini.receive():
                await websocket.send_json(msg)
                mtype = msg["type"]

                # Coach started replying => the user's answer just ended. Snapshot
                # the buffered answer audio and transcribe it with the fine-tune.
                if stt_on and not coach_speaking and mtype in ("audio", "output_transcript"):
                    coach_speaking = True
                    snapshot = bytes(answer_audio)
                    answer_audio.clear()
                    if len(snapshot) >= MIN_ANSWER_BYTES and simple_vad(snapshot, 300):
                        asyncio.create_task(transcribe_answer(snapshot))

                if mtype == "input_transcript":
                    answer_text += msg["text"]
                elif mtype == "output_transcript":
                    coach_text += msg["text"]
                elif mtype == "turn_complete":
                    # Grade the candidate's answer (skip the greeting turn, which
                    # has no answer). Fall back to parsing the coach's speech.
                    scores = None
                    if answer_text.strip():
                        scores = await coach.grade_answer(question, answer_text)
                    if scores is None:
                        scores = coach.parse_feedback(coach_text)
                    if scores is not None:
                        await websocket.send_json(
                            {"type": "feedback", "scores": scores.model_dump()}
                        )
                        if session_id:
                            await asyncio.to_thread(
                                db.update_scores, session_id, scores.model_dump()
                            )
                    answer_text = ""
                    coach_text = ""
                    coach_speaking = False
                    answer_audio.clear()  # drop any audio captured during the reply
        except WebSocketDisconnect:
            logger.info("Browser disconnected (downlink)")
        except Exception:
            logger.exception("downlink task error")

    up = asyncio.create_task(uplink())
    down = asyncio.create_task(downlink())
    try:
        _, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        await gemini.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass
        logger.info("Session ended (ws)")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True,
    )
