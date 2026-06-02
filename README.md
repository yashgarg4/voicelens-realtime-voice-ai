# VoiceLens

A real-time, full-duplex AI interview coach with an accent-aware transcription layer.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-unittest-success)

VoiceLens is an AI engineer interview coach you talk to. You pick a question, answer out loud, and a Gemini Live model listens while you speak and replies with spoken feedback plus a scored breakdown of content, technical depth, and structure. A second layer fine-tunes Whisper-small on Indian-accented English with QLoRA so the transcription that drives evaluation is faithful to how the candidate actually speaks. It is two systems in one: a low-latency realtime voice pipeline, and a measurable accent-adaptation result.

## Features

| B1 — Realtime voice | B2 — Accent-aware transcription |
|---|---|
| **Full-duplex audio** — speak and hear replies over a single WebSocket | **QLoRA fine-tuning** — Whisper-small adapted with 4-bit LoRA adapters |
| **Barge-in** — interrupt the coach mid-sentence; playback flushes instantly | **Indian English** — trained on the Svarah benchmark (117 Indian speakers) |
| **Interview brain** — 20 AI-engineer questions across 4 categories | **Measurable WER gain** — 16.98% → 13.97% on held-out clips (−17.8%) |
| **Spoken + scored feedback** — Content / Depth / Structure, parsed live | **Tiny adapter** — ~3 MB shipped instead of a multi-GB model |
| **Reconnection** — exponential backoff (3 retries) on dropped sockets | **Base fallback** — runs on the base model if no adapter is present |
| **Dashboard** — session history + a pure-SVG score radar chart | **Reproducible** — one Colab notebook runs prepare → train → evaluate |

## Architecture

```
                            BROWSER (React + TypeScript)
   mic ──► getUserMedia 16kHz ──► Float32→Int16 PCM ──► binary WS frames
                                                              │
   speakers ◄── 24kHz AudioBuffer queue ◄── JSON {audio,transcript,feedback} ◄┐
                                                              │               │
                                                              ▼               │
                            FastAPI  (/ws/session)                            │
                  uplink: mic bytes ─► Gemini      downlink: events ─► browser┘
                                  │                         ▲
                                  ▼                         │
                          GEMINI LIVE API  (full-duplex, server-side VAD)
                          speech-in ─► reasoning ─► speech-out + transcripts
                                  │
                  turn_complete ─► parse_feedback() ─► scores ─► SQLite
                                                                  │
                                                       GET /api/sessions, /api/wer

   FINE-TUNING PIPELINE (offline, GPU/Colab)
   Svarah (Indian English) ─► 16kHz log-mels ─► QLoRA train ─► adapter (~3MB)
                                                       │
                                          evaluate.py ─► wer_result.json ─► /api/wer
                                          transcriber.py loads adapter (base fallback)
```

Two layers, decoupled on purpose. **B1** is the online experience: the browser streams microphone PCM to FastAPI, which relays it to the Gemini Live API and streams spoken replies, transcripts, and parsed scores back. **B2** is an offline pipeline: Whisper-small is fine-tuned with QLoRA on Indian-accented English, emitting a small adapter and a WER comparison artifact that the backend serves. The realtime app never needs the ML stack loaded; the training job never needs the web server — they meet through small artifacts (`wer_result.json`, the adapter directory).

## Tech stack

| Component | Technology |
|---|---|
| **Backend** | Python 3.11+, FastAPI, uvicorn, `websockets` |
| | `google-genai` (Gemini Live API), Pydantic, NumPy |
| | SQLite (stdlib `sqlite3`) for session storage |
| **Frontend** | React 18, TypeScript, Vite |
| | Web Audio API (capture + playback), native WebSocket |
| | Tailwind CSS v4 (no component library) |
| **Fine-tuning** | PyTorch, Transformers, PEFT (LoRA), bitsandbytes (4-bit) |
| | Datasets, Evaluate + jiwer (WER), Whisper-small |
| **Infrastructure** | Vite dev proxy, Google Colab (T4) for training, Makefile |

## Quick start

```bash
# 1. Clone
git clone <your-fork-url> voicelens && cd voicelens

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 4. Configure your key
cp .env.example .env               # Windows: copy .env.example .env
#    then edit .env and set GOOGLE_API_KEY=<key from aistudio.google.com/apikey>

# 5. Start the backend (terminal 1)
uvicorn backend.main:app --reload  # or: make backend

# 6. Start the frontend (terminal 2)
cd frontend && npm run dev         # or: make frontend
```

7. Open **http://localhost:5173**, pick a question, click the mic, and answer out loud.

## Fine-tuning

**What QLoRA is, in plain English:** instead of retraining the whole model (expensive, needs a big GPU), you freeze it and compress it to 4-bit so it takes a quarter of the memory, then train a few small "adapter" matrices bolted onto it. You end up training well under 1% of the parameters, so it fits on a free GPU and you ship a tiny adapter file instead of a multi-gigabyte model — with almost no loss in quality.

A GPU is required (a free Google Colab T4 works). The easiest path is the notebook [`finetune/voicelens_finetune_colab.ipynb`](finetune/voicelens_finetune_colab.ipynb), which runs all three steps. To run the scripts directly:

```bash
python finetune/prepare_dataset.py   # stream Svarah, 16kHz log-mels -> finetune/data
python finetune/train.py             # 4-bit QLoRA; prints trainable-parameter %
python finetune/evaluate.py          # WER comparison + writes wer_result.json
```

Expected output: `Base WER: ~17% → Fine-tuned WER: ~14%` on the held-out set, with roughly **0.6%** of parameters trained. (The dataset, [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), is gated — accept its terms and set `HF_TOKEN`.)

## Project structure

```
voicelens/
├── backend/
│   ├── main.py           # FastAPI app: REST + WS audio relay + feedback parsing
│   ├── gemini_live.py    # Gemini Live WebSocket client wrapper (multi-turn)
│   ├── audio.py          # PCM helpers, energy VAD, resampling, format asserts
│   ├── coach.py          # interviewer system prompt, question bank, score parsing
│   ├── transcriber.py    # Whisper wrapper, loads QLoRA adapter (base fallback)
│   ├── db.py             # SQLite session storage
│   └── config.py         # env vars, audio-format constants, model name
├── finetune/
│   ├── prepare_dataset.py# download + prep Indian-English audio (Svarah)
│   ├── train.py          # QLoRA training (4-bit base + LoRA adapters)
│   ├── evaluate.py        # base vs fine-tuned WER comparison
│   ├── ft_utils.py       # shared config loader + data collator
│   ├── config.yaml       # training hyperparameters
│   └── voicelens_finetune_colab.ipynb  # one-click Colab runner
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── hooks/
│       │   ├── useVoiceStream.ts  # mic capture, WS audio, reconnection, playback
│       │   └── useSession.ts      # question bank, session + history + WER fetch
│       └── components/
│           ├── VoicePanel.tsx     # mic button, audio-level visualiser, status
│           ├── QuestionDisplay.tsx# question picker / current question
│           ├── FeedbackCard.tsx   # scored feedback + radar
│           ├── RadarChart.tsx     # pure-SVG Content/Depth/Structure radar
│           ├── ScoreHistory.tsx   # expandable past-session rows
│           └── WerBadge.tsx       # base→fine-tuned WER pill
├── questions/ai_engineer.json     # 20 interview questions (4 categories)
├── tests/                         # unittest suite (audio + score parsing)
├── INTERNAL_NOTES.md              # design decisions, debugging log, interview Q&A
├── requirements.txt
├── Makefile
└── README.md
```

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/ws/session` | WS | Full-duplex audio relay; `?question_id=&session_id=`. Binary PCM up, JSON down |
| `/api/session/start` | POST | Begin a session for a question; returns `session_id` + question |
| `/api/session/end` | POST | Mark a session ended |
| `/api/sessions` | GET | Past sessions with scores (for the dashboard) |
| `/api/wer` | GET | Base vs fine-tuned Whisper WER comparison |
| `/api/health` | GET | Liveness + config (key present, model, sample rates) |
| `/api/questions` | GET | The 20-question AI-engineer bank |

## Environment variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | — | Yes | Gemini API key from aistudio.google.com/apikey |
| `GEMINI_LIVE_MODEL` | `gemini-3.1-flash-live-preview` | No | Gemini Live model id, default in `backend/config.py` (stable alternative: `gemini-2.0-flash-live-001`) |
| `BACKEND_HOST` | `0.0.0.0` | No | Backend bind host |
| `BACKEND_PORT` | `8000` | No | Backend port |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | No | Allowed CORS origin |
| `HF_TOKEN` | — | Fine-tuning only | HuggingFace token for the gated Svarah dataset |

## Running tests

The Python unit tests (stdlib `unittest`, no extra deps) cover the audio helpers and the spoken-feedback score parser:

```bash
python -m unittest discover -s tests -t .   # or: make test
```
