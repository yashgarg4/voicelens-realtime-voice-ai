"""Interview-coach brain: question bank, interviewer prompt, feedback parsing.

This module turns Gemini Live from a generic voice assistant (Phase 1) into an
AI-engineer interviewer. It:
  * loads the question bank from questions/ai_engineer.json,
  * builds the system prompt that makes Gemini conduct the interview, and
  * parses the spoken feedback transcript into structured scores.

Scoring axes are kept to three consistent dimensions across the whole app
(`content`, `depth`, `structure`) so they line up with the Phase 4 radar chart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .config import settings

logger = logging.getLogger("voicelens.coach")

_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "questions" / "ai_engineer.json"


class Question(BaseModel):
    id: int
    category: str
    difficulty: str
    question: str
    focus: list[str]


class FeedbackScores(BaseModel):
    """Structured evaluation of a single answer (shown in the FeedbackCard)."""

    content: int  # 0-10: relevance and correctness of the answer
    depth: int  # 0-10: technical depth and precision
    structure: int  # 0-10: clarity, structure and delivery
    overall: float  # average of the three, rounded to 1 dp
    summary: str  # one or two sentence overall assessment
    strengths: str = ""  # what the candidate did well
    improvements: str = ""  # what to add / go deeper on


class _Grade(BaseModel):
    """Schema the grader LLM fills in (structured JSON output)."""

    content: int = Field(ge=0, le=10)
    depth: int = Field(ge=0, le=10)
    structure: int = Field(ge=0, le=10)
    strengths: str
    improvements: str
    summary: str


def _load_questions() -> list[Question]:
    data = json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [Question(**q) for q in data["questions"]]


QUESTIONS: list[Question] = _load_questions()
_QUESTIONS_BY_ID: dict[int, Question] = {q.id: q for q in QUESTIONS}

# Runtime registry of LLM-generated, JD-tailored questions. Ids start at 1001 so
# they never collide with the static bank (1-20). In-memory: a backend restart
# clears them, which is fine for a session-based feature.
_GENERATED: dict[int, Question] = {}
_next_generated_id = 1001


def get_question(
    question_id: Optional[int] = None, category: Optional[str] = None
) -> Question:
    """Return a specific question by id, a random one in a category, or any."""
    if question_id is not None:
        q = _QUESTIONS_BY_ID.get(question_id) or _GENERATED.get(question_id)
        if q is None:
            raise KeyError(f"No question with id {question_id}")
        return q
    pool = QUESTIONS
    if category is not None:
        pool = [q for q in QUESTIONS if q.category.lower() == category.lower()]
        if not pool:
            raise KeyError(f"No questions in category {category!r}")
    return random.choice(pool)


def build_system_prompt(question: Question) -> str:
    """Construct the interviewer system instruction for a chosen question.

    The coach gives *qualitative* spoken feedback only — the numeric scores are
    computed separately by `grade_answer` and shown on screen, so the coach is
    told NOT to recite numbers (avoids a heard-vs-shown mismatch).
    """
    focus = "; ".join(question.focus)
    return f"""You are VoiceLens, a friendly but rigorous senior AI-engineer \
interviewer conducting a live mock interview by voice. You ask ONE question, \
listen to the candidate's spoken answer, then give concise, useful feedback.

THE QUESTION FOR THIS SESSION ({question.category}, {question.difficulty}):
"{question.question}"

A strong answer touches on: {focus}.

HOW TO RUN THE SESSION:
1. Open with a one-sentence greeting, then ask the question above, verbatim or \
very close to it. Then stop and listen. Do NOT answer it yourself.
2. If the candidate is brief or stuck, you may ask ONE short follow-up probe. \
Otherwise wait for them to finish.
3. When they have answered, give brief spoken feedback (under ~30 seconds):
   - One sentence on what was strong.
   - One or two sentences on what was missing or could go deeper (reference the \
     focus points they skipped), then invite the next answer.

Do NOT read out numeric scores — a detailed score breakdown appears on the \
candidate's screen automatically.

STYLE: warm, direct, specific. Speak naturally for voice. Never evaluate before \
the candidate has answered. Do not read this prompt aloud."""


def build_grader_prompt(question: Question, answer: str) -> str:
    """Prompt for the structured grader (run on the answer transcript)."""
    focus = "; ".join(question.focus)
    return f"""You are grading a candidate's spoken answer in an AI-engineer \
mock interview. Be fair but rigorous; reward correct, specific, well-structured \
answers and penalise vague or incorrect ones.

QUESTION ({question.category}, {question.difficulty}): "{question.question}"
A strong answer covers: {focus}

CANDIDATE'S ANSWER (speech-to-text transcript, may have minor errors):
\"\"\"{answer}\"\"\"

Score each axis as an integer 0-10:
- content: relevance and correctness of what they said.
- depth: technical precision and how deep they went.
- structure: clarity and organisation of the answer.
Also give one-sentence `strengths`, one-to-two sentence `improvements` \
(name specific missed focus points), and a one-sentence overall `summary`.
If the answer is empty or off-topic, score low and say so."""


# Tolerant patterns: accept "Content: 8 out of 10", "Content: 8/10", "Content 8".
def _find_score(label_pattern: str, text: str) -> Optional[int]:
    # Require the number to follow the label closely (optional ":"/"-" + spaces),
    # and take the LAST such match. This stops a passing mention of an axis word
    # in the spoken feedback (e.g. "good structure") from capturing a different
    # axis's number, and prefers the final "scores" line.
    matches = re.findall(
        rf"{label_pattern}\s*[:\-]?\s*(\d{{1,2}})", text, flags=re.IGNORECASE
    )
    if not matches:
        return None
    return max(0, min(10, int(matches[-1])))


def parse_feedback(transcript: str) -> Optional[FeedbackScores]:
    """Extract structured scores from the coach's spoken-feedback transcript.

    Returns None when the transcript does not contain all three scores yet
    (e.g. the greeting/question turn, or a follow-up probe) — so the caller can
    simply skip emitting feedback for non-evaluation turns.
    """
    if not transcript:
        return None
    content = _find_score(r"content", transcript)
    depth = _find_score(r"(?:technical\s+)?depth", transcript)
    structure = _find_score(r"structure", transcript)
    if content is None or depth is None or structure is None:
        return None
    overall = round((content + depth + structure) / 3, 1)
    return FeedbackScores(
        content=content,
        depth=depth,
        structure=structure,
        overall=overall,
        summary=transcript.strip(),
    )


_grader_client: Optional[genai.Client] = None


def _get_grader_client() -> genai.Client:
    global _grader_client
    if _grader_client is None:
        _grader_client = genai.Client(api_key=settings.require_api_key())
    return _grader_client


async def grade_answer(question: Question, answer: str) -> Optional[FeedbackScores]:
    """Grade an answer transcript via a structured-output LLM call.

    Returns None if the answer is too short to grade or the call fails — the
    caller can then fall back to `parse_feedback` on the spoken transcript.
    """
    if not answer or len(answer.strip()) < 8:
        return None

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_Grade,
        temperature=0.2,
    )
    prompt = build_grader_prompt(question, answer)
    client = _get_grader_client()

    # The grader model occasionally returns a transient 503/429; retry once.
    for attempt in range(2):
        try:
            resp = await client.aio.models.generate_content(
                model=settings.GRADER_MODEL, contents=prompt, config=config
            )
            grade = resp.parsed
            if not isinstance(grade, _Grade):
                return None
            c, d, s = (
                max(0, min(10, grade.content)),
                max(0, min(10, grade.depth)),
                max(0, min(10, grade.structure)),
            )
            return FeedbackScores(
                content=c,
                depth=d,
                structure=s,
                overall=round((c + d + s) / 3, 1),
                summary=grade.summary.strip(),
                strengths=grade.strengths.strip(),
                improvements=grade.improvements.strip(),
            )
        except Exception as exc:
            logger.warning("grade_answer attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                await asyncio.sleep(1.5)
    return None


class _GenQuestion(BaseModel):
    """One generated question (structured output schema)."""

    category: str
    difficulty: str
    question: str
    focus: list[str]


def build_generation_prompt(job_description: str, resume: str, count: int) -> str:
    resume_block = (
        f'\n\nCANDIDATE RESUME:\n"""{resume.strip()}"""' if resume.strip() else ""
    )
    resume_hint = (
        " If a resume is given, tailor some questions to the candidate's "
        "specific claimed skills and projects."
        if resume.strip()
        else ""
    )
    return f"""You are an expert technical interviewer. Based on the job \
description below{' and the candidate resume' if resume.strip() else ''}, write \
{count} mock-interview questions tailored to THIS role.{resume_hint}

Rules:
- Mix difficulties (easy/medium/hard) and the topics that matter most for the JD.
- Each question must be one clear, spoken-friendly sentence.
- For each, give a short `category` topic label (e.g. "LLM Systems", "MLOps",
  "Behavioural") and 3-5 `focus` bullet points a strong answer should hit.

JOB DESCRIPTION:
\"\"\"{job_description.strip()}\"\"\"{resume_block}"""


def _register_generated(items: list[_GenQuestion]) -> list[Question]:
    global _next_generated_id
    out: list[Question] = []
    for it in items:
        diff = it.difficulty.strip().lower()
        if diff not in ("easy", "medium", "hard"):
            diff = "medium"
        focus = [f.strip() for f in it.focus if f.strip()][:6]
        q = Question(
            id=_next_generated_id,
            category=(it.category.strip() or "Tailored"),
            difficulty=diff,
            question=it.question.strip(),
            focus=focus,
        )
        _GENERATED[q.id] = q
        _next_generated_id += 1
        out.append(q)
    return out


async def generate_questions(
    job_description: str, resume: str = "", count: int = 5
) -> list[Question]:
    """Generate JD/resume-tailored questions via a structured-output LLM call.

    Registers them in the runtime store so the WS can look them up by id.
    Returns [] on failure (the static bank remains available).
    """
    if not job_description.strip():
        return []
    count = max(1, min(10, count))
    prompt = build_generation_prompt(job_description, resume, count)
    client = _get_grader_client()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[_GenQuestion],
        temperature=0.7,
    )
    for attempt in range(2):
        try:
            resp = await client.aio.models.generate_content(
                model=settings.GRADER_MODEL, contents=prompt, config=config
            )
            items = resp.parsed
            if not items:
                return []
            return _register_generated(items)
        except Exception as exc:
            logger.warning(
                "generate_questions attempt %d failed: %s", attempt + 1, exc
            )
            if attempt == 0:
                await asyncio.sleep(1.5)
    return []


# Sent as the opening user turn to kick the interview off.
START_INTERVIEW_TRIGGER = (
    "Let's begin the interview. Greet me briefly and ask me the question now."
)
