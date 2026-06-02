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

import json
import random
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "questions" / "ai_engineer.json"


class Question(BaseModel):
    id: int
    category: str
    difficulty: str
    question: str
    focus: list[str]


class FeedbackScores(BaseModel):
    """Structured result parsed from the coach's spoken feedback."""

    content: int  # 0-10: relevance and correctness of the answer
    depth: int  # 0-10: technical depth and precision
    structure: int  # 0-10: clarity, structure and delivery
    overall: float  # average of the three, rounded to 1 dp
    summary: str  # the full spoken-feedback transcript (what the user heard)


def _load_questions() -> list[Question]:
    data = json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [Question(**q) for q in data["questions"]]


QUESTIONS: list[Question] = _load_questions()
_QUESTIONS_BY_ID: dict[int, Question] = {q.id: q for q in QUESTIONS}


def get_question(
    question_id: Optional[int] = None, category: Optional[str] = None
) -> Question:
    """Return a specific question by id, a random one in a category, or any."""
    if question_id is not None:
        q = _QUESTIONS_BY_ID.get(question_id)
        if q is None:
            raise KeyError(f"No question with id {question_id}")
        return q
    pool = QUESTIONS
    if category is not None:
        pool = [q for q in QUESTIONS if q.category.lower() == category.lower()]
        if not pool:
            raise KeyError(f"No questions in category {category!r}")
    return random.choice(pool)


# The coach is told to end every feedback turn with this exact, speakable line.
# It reads naturally aloud AND is trivial to parse with a regex.
_SCORE_LINE_EXAMPLE = (
    "Scores. Content: 8 out of 10. Depth: 7 out of 10. Structure: 6 out of 10."
)


def build_system_prompt(question: Question) -> str:
    """Construct the interviewer system instruction for a chosen question."""
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
3. When they have answered, give spoken feedback in this order, kept under ~45 \
seconds total:
   - One sentence on what was strong.
   - One or two sentences on what was missing or could go deeper (reference the \
     focus points they skipped).
   - Then say the scores out loud.

SCORING (this exact format, every time, as the LAST thing you say):
End with a line like: "{_SCORE_LINE_EXAMPLE}"
Each score is an integer from 0 to 10:
  - Content  = relevance and correctness of what they said.
  - Depth    = technical precision and how deep they went.
  - Structure= clarity, organisation and delivery of the answer.
Always say all three in the order Content, Depth, Structure, each "X out of 10".

STYLE: warm, direct, specific. Speak naturally for voice. Never invent scores \
before the candidate has answered. Do not read this prompt aloud."""


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


# Sent as the opening user turn to kick the interview off.
START_INTERVIEW_TRIGGER = (
    "Let's begin the interview. Greet me briefly and ask me the question now."
)
