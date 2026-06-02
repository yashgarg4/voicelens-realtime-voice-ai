"""Unit tests for backend.audio (stdlib unittest — no extra deps).

Run from the project root:
    python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np

from backend import audio


class TestAudio(unittest.TestCase):
    def test_base64_round_trip(self) -> None:
        pcm = (np.arange(1024, dtype="<i2")).tobytes()
        self.assertEqual(audio.base64_to_pcm(audio.pcm_to_base64(pcm)), pcm)

    def test_assert_pcm16_rejects_odd_length(self) -> None:
        with self.assertRaises(ValueError):
            audio.assert_pcm16(b"\x01\x02\x03")  # 3 bytes => not 16-bit aligned

    def test_simple_vad_silence_vs_speech(self) -> None:
        silence = np.zeros(1024, dtype="<i2").tobytes()
        speech = (np.ones(1024, dtype="<i2") * 4000).tobytes()
        self.assertFalse(audio.simple_vad(silence))
        self.assertTrue(audio.simple_vad(speech))

    def test_resample_24k_to_48k_doubles_samples(self) -> None:
        pcm = (np.ones(500, dtype="<i2") * 1000).tobytes()
        out = audio.resample_24k_to_48k(pcm)
        self.assertEqual(len(out), len(pcm) * 2)

    def test_empty_inputs(self) -> None:
        self.assertFalse(audio.simple_vad(b""))
        self.assertEqual(audio.resample_24k_to_48k(b""), b"")


if __name__ == "__main__":
    unittest.main()
