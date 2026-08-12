"""Tests for voice STT module (Phase 1).

CI-safe: tests that require real hardware or API keys are skipped by default.
"""

from __future__ import annotations

import io
import os
import wave

import pytest

# ── Pure-function imports (no audio hardware needed) ─────────────────────
from support.voice import SAMPLE_RATE, FRAME_SIZE, pcm_to_wav

# ── Audio imports — may fail if deps not installed ───────────────────────
try:
    import pyaudio  # noqa: F401
    import webrtcvad  # noqa: F401
    _AUDIO_DEPS_AVAILABLE = True
except ImportError:
    _AUDIO_DEPS_AVAILABLE = False

if _AUDIO_DEPS_AVAILABLE:
    from support.voice import VoiceRecorder, recognize_speech


# =============================================================================
# Unit tests — no hardware / API key needed
# =============================================================================

class TestPCMToWav:
    """In-memory PCM → WAV conversion (stdlib only)."""

    def test_empty_pcm(self):
        wav = pcm_to_wav(b"")
        assert len(wav) >= 44

    def test_roundtrip_format(self):
        """WAV header should declare 16-bit mono 16kHz."""
        pcm = b"\x00\x00" * SAMPLE_RATE
        wav = pcm_to_wav(pcm)
        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnframes() == SAMPLE_RATE

    def test_wav_magic_bytes(self):
        wav = pcm_to_wav(b"\x00\x00" * 480)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_small_pcm(self):
        wav = pcm_to_wav(b"\x00\x00" * 10)
        assert len(wav) >= 44
        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 10


class TestConstants:
    """Sanity-check audio constants."""

    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000

    def test_frame_size(self):
        assert FRAME_SIZE == 480


# =============================================================================
# Tests requiring audio deps
# =============================================================================

@pytest.mark.skipif(not _AUDIO_DEPS_AVAILABLE, reason="pyaudio/webrtcvad not installed")
class TestVADBoundaryDetection:
    """Voice boundary logic — needs pyaudio+webrtcvad (no mic input)."""

    def test_is_speaking_initially_false(self):
        rec = VoiceRecorder(vad_sensitivity=2)
        assert rec.is_speaking() is False

    def test_get_audio_none_before_start(self):
        rec = VoiceRecorder(vad_sensitivity=2)
        assert rec.get_audio() is None

    def test_cancel_on_non_running(self):
        rec = VoiceRecorder(vad_sensitivity=2)
        rec.cancel()  # should not raise


# =============================================================================
# Integration tests (need mic + API key)
# =============================================================================

_API_KEY = os.getenv("OPENAI_API_KEY", "")

@pytest.mark.skipif(
    not _AUDIO_DEPS_AVAILABLE or not _API_KEY,
    reason="requires audio deps + OPENAI_API_KEY",
)
class TestRecordAndTranscribe:
    """End-to-end: record → STT → Chinese text."""

    def test_record_and_transcribe(self):
        rec = VoiceRecorder(vad_sensitivity=2)
        rec.start()
        import time
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if not rec._running:
                break
            time.sleep(0.1)
        rec.stop()
        pcm = rec.get_audio()
        if not pcm:
            pytest.skip("No speech detected during test")
        text = recognize_speech(
            pcm,
            provider="openai_whisper",
            api_key=_API_KEY,
        )
        assert text
        assert text != "[未识别]"
        assert len(text) > 0
