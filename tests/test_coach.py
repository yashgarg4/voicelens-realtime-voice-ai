"""Unit tests for backend.coach (question bank + feedback parsing).

Run from the project root:
    python -m unittest discover -s tests -t .
"""

import unittest

from backend import coach


class TestCoach(unittest.TestCase):
    def test_question_bank_is_20_with_unique_sequential_ids(self) -> None:
        ids = sorted(q.id for q in coach.QUESTIONS)
        self.assertEqual(ids, list(range(1, 21)))

    def test_four_categories_balanced(self) -> None:
        counts: dict[str, int] = {}
        for q in coach.QUESTIONS:
            counts[q.category] = counts.get(q.category, 0) + 1
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(len(counts), 4)

    def test_get_question_by_id_and_category(self) -> None:
        self.assertEqual(coach.get_question(15).id, 15)
        self.assertEqual(
            coach.get_question(category="Behavioural").category, "Behavioural"
        )
        with self.assertRaises(KeyError):
            coach.get_question(999)

    def test_parse_feedback_spoken_form(self) -> None:
        fb = coach.parse_feedback(
            "Good structure. Scores. Content: 8 out of 10. "
            "Depth: 6 out of 10. Structure: 7 out of 10."
        )
        self.assertIsNotNone(fb)
        assert fb is not None
        self.assertEqual((fb.content, fb.depth, fb.structure), (8, 6, 7))
        self.assertEqual(fb.overall, 7.0)

    def test_parse_feedback_slash_form(self) -> None:
        fb = coach.parse_feedback("Content: 9/10. Depth: 9/10. Structure: 10/10.")
        assert fb is not None
        self.assertEqual((fb.content, fb.depth, fb.structure), (9, 9, 10))

    def test_parse_feedback_returns_none_for_non_eval_turn(self) -> None:
        self.assertIsNone(coach.parse_feedback("Hello! Here is your question."))
        self.assertIsNone(coach.parse_feedback(""))

    def test_scores_clamped_to_0_10(self) -> None:
        fb = coach.parse_feedback("Content: 99. Depth: 5. Structure: 7.")
        assert fb is not None
        self.assertEqual(fb.content, 10)  # clamped


if __name__ == "__main__":
    unittest.main()
