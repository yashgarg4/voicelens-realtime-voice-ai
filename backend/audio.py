"""PCM audio utilities for the VoiceLens pipeline.

Everything here operates on raw PCM as `bytes`: 16-bit signed integers,
little-endian, mono. That is the only format Gemini Live accepts (16 kHz in)
and emits (24 kHz out). We add cheap assertions at audio boundaries so a wrong
format fails loudly here instead of failing silently at the API.
"""

from __future__ import annotations

import base64

import numpy as np

# 16-bit signed PCM => each sample is 2 bytes.
_BYTES_PER_SAMPLE = 2


def assert_pcm16(pcm_bytes: bytes) -> None:
    """Cheap sanity check that a buffer is plausibly 16-bit mono PCM.

    16-bit samples are 2 bytes each, so a valid mono PCM buffer always has an
    even length. An odd length means we were handed Float32 data, a truncated
    frame, or text — all of which produce silent failure at the Live API.
    """
    if not isinstance(pcm_bytes, (bytes, bytearray)):
        raise TypeError(f"expected bytes, got {type(pcm_bytes).__name__}")
    if len(pcm_bytes) % _BYTES_PER_SAMPLE != 0:
        raise ValueError(
            f"PCM16 buffer must have an even byte length, got {len(pcm_bytes)}. "
            "This usually means the audio is not 16-bit signed integers."
        )


def pcm_to_base64(pcm_bytes: bytes) -> str:
    """Encode raw PCM bytes as an ASCII base64 string (for JSON transport)."""
    assert_pcm16(pcm_bytes)
    return base64.b64encode(pcm_bytes).decode("ascii")


def base64_to_pcm(b64_str: str) -> bytes:
    """Decode a base64 string back into raw PCM bytes."""
    pcm = base64.b64decode(b64_str)
    assert_pcm16(pcm)
    return pcm


def simple_vad(pcm_bytes: bytes, threshold: int = 500) -> bool:
    """Energy-based Voice Activity Detection.

    Returns True if the chunk appears to contain speech. We compute the RMS
    (root-mean-square) amplitude of the 16-bit samples; near-silence sits well
    below a few hundred, normal speech is in the thousands. `threshold=500` is
    a conservative gate that drops room tone without clipping quiet speech.

    This is deliberately simple: Gemini Live does its own server-side VAD for
    turn detection, so client-side VAD here is only used to avoid shipping pure
    silence over the wire.
    """
    assert_pcm16(pcm_bytes)
    if len(pcm_bytes) == 0:
        return False
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return rms >= threshold


def resample_24k_to_48k(pcm_bytes: bytes) -> bytes:
    """Upsample mono PCM16 from 24 kHz to 48 kHz (integer 2x).

    Gemini emits 24 kHz; some browser AudioContexts run at 48 kHz. A clean 2x
    ratio lets us upsample with linear interpolation and no anti-alias filter
    (upsampling does not introduce aliasing). Provided as a backend-side option;
    the frontend can alternatively let Web Audio resample by constructing the
    AudioBuffer at 24 kHz directly (the approach VoiceLens uses by default).
    """
    assert_pcm16(pcm_bytes)
    if len(pcm_bytes) == 0:
        return b""
    src = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
   
    src_idx = np.arange(src.size)
    dst_idx = np.arange(src.size * 2) / 2.0
    upsampled = np.interp(dst_idx, src_idx, src)
    return np.clip(np.round(upsampled), -32768, 32767).astype("<i2").tobytes()
