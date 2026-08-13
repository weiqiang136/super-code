"""Tests for voiceprint module (Phase 1.5).

CI-safe: real embedding tests are skipped unless resemblyzer + model available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from support.voiceprint import VoiceprintManager, _to_wav_array


class _FakeEncoder:
    """Mock encoder：恒返回归一化的 256 维向量（模=1），避免加载 50MB 模型。"""

    def embed_utterance(self, wav):
        v = np.ones(256, dtype=np.float32)
        return v / np.linalg.norm(v)


def _make_tone_pcm(seconds: float = 0.5, freq: int = 440) -> bytes:
    """生成非静音测试音频（440Hz 正弦波）。"""
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    return (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16).tobytes()


# =============================================================================
# Pure logic tests — no resemblyzer model needed
# =============================================================================

class TestVoiceprintStorage:
    """Storage / listing / deletion logic (no model needed)."""

    def test_empty_library_verify_passes(self, tmp_path):
        """空声纹库 → verify 放行 (True, 1.0, None)。"""
        mgr = VoiceprintManager(tmp_path)
        passed, score, name = mgr.verify(b"\x00\x00" * 960)
        assert passed is True
        assert score == 1.0
        assert name is None

    def test_enroll_writes_npy_file(self, tmp_path, monkeypatch):
        """enroll 生成 .npy 文件（mock encoder 避免加载模型）。"""
        import support.voiceprint as vp
        monkeypatch.setattr(vp, "_get_encoder", lambda: _FakeEncoder())

        mgr = VoiceprintManager(tmp_path)
        path = mgr.enroll("alice", _make_tone_pcm())
        assert path is not None
        assert path.exists()
        assert path.name == "alice.npy"
        assert mgr.list_names() == ["alice"]

    def test_verify_with_stored_embedding(self, tmp_path, monkeypatch):
        """验证：存储的嵌入向量与输入一致 → 相似度 1.0 通过。"""
        import support.voiceprint as vp
        monkeypatch.setattr(vp, "_get_encoder", lambda: _FakeEncoder())
        monkeypatch.setattr(vp, "_to_wav_array",
                            lambda pcm, sr=16000: np.zeros(16000, dtype=np.float32))

        mgr = VoiceprintManager(tmp_path)
        mgr.enroll("alice", _make_tone_pcm())
        passed, score, name = mgr.verify(_make_tone_pcm(), threshold=0.75)
        assert passed is True
        assert score == pytest.approx(1.0)
        assert name == "alice"

    def test_delete_removes_file(self, tmp_path, monkeypatch):
        import support.voiceprint as vp
        monkeypatch.setattr(vp, "_get_encoder", lambda: _FakeEncoder())

        mgr = VoiceprintManager(tmp_path)
        mgr.enroll("bob", _make_tone_pcm())
        assert mgr.delete("bob") is True
        assert mgr.list_names() == []
        assert mgr.delete("nobody") is False  # 删除不存在的返回 False


# =============================================================================
# Integration tests — require real resemblyzer + model download (~50MB)
# =============================================================================

def _resemblyzer_available() -> bool:
    try:
        import resemblyzer  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _resemblyzer_available(), reason="resemblyzer not installed")
class TestVoiceprintRealModel:
    """真实模型集成测试（会下载 ~50MB 模型，首次较慢）。"""

    def test_to_wav_array_shape(self):
        """PCM → float32 数组，长度合理。"""
        wav = _to_wav_array(_make_tone_pcm(), 16000)
        assert wav.dtype == np.float32
        assert wav.size > 0

    def test_real_enroll_and_verify(self, tmp_path):
        """真实注册 + 验证同一段音频 → 通过。"""
        mgr = VoiceprintManager(tmp_path)
        pcm = _make_tone_pcm(seconds=1.0)

        path = mgr.enroll("carol", pcm)
        if path is None:
            pytest.skip("voiceprint model unavailable (download failed)")
        passed, score, name = mgr.verify(pcm, threshold=0.75)
        assert passed is True
        assert name == "carol"
