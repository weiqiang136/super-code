"""声纹注册 + 验证（Phase 1.5）。

基于 resemblyzer 本地提取 ECAPA-TDNN 嵌入向量（256 维），余弦相似度比对。
零网络依赖（模型首次下载后本地运行），与 STT provider 完全解耦。

用法::

    mgr = VoiceprintManager(storage_dir)
    mgr.enroll("songweiqiang", pcm_bytes)          # 注册
    passed, score, name = mgr.verify(pcm_bytes)    # 验证
    mgr.delete("songweiqiang")                     # 删除
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# resemblyzer 惰性导入：包未装 / 模型下载失败时降级（voiceprint 禁用），不拖垮启动
_encoder = None
_encoder_lock = threading.Lock()


def _get_encoder():
    """获取全局共享的 VoiceEncoder 实例（线程安全，首次调用下载 ~50MB 模型）。

    失败返回 None（调用方自行降级处理）。
    """
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        try:
            from resemblyzer import VoiceEncoder
            _encoder = VoiceEncoder()
            logger.info("resemblyzer VoiceEncoder 初始化完成（~50MB 模型已加载）")
        except Exception as exc:
            logger.warning(f"resemblyzer 初始化失败，声纹功能禁用: {exc}")
            _encoder = None
    return _encoder


class VoiceprintManager:
    """管理声纹的注册、存储和验证。

    存储目录结构::

        {storage_dir}/voiceprints/
            └── {name}.npy        # 256 维嵌入向量（~4KB）

    声纹库为空（首次使用）时 verify() 放行并返回 (True, 1.0, None)，
    由调用方引导用户注册。
    """

    def __init__(self, storage_dir: Path | str):
        self._storage = Path(storage_dir) / "voiceprints"
        self._storage.mkdir(parents=True, exist_ok=True)

    # ── 公共 API ────────────────────────────────────────────────────────

    def enroll(self, name: str, pcm: bytes, sample_rate: int = 16000) -> Path | None:
        """注册声纹：从语音中提取嵌入向量并保存为 .npy。

        name 用于区分不同用户（如 'songweiqiang'），后续可扩展多人场景。
        返回保存路径；模型不可用时返回 None（调用方提示降级）。
        """
        encoder = _get_encoder()
        if encoder is None:
            return None

        wav = _to_wav_array(pcm, sample_rate)
        if wav.size == 0:
            logger.warning(f"声纹注册失败: 音频无效（{len(pcm)} bytes 被裁剪为空）")
            return None

        embed = encoder.embed_utterance(wav)  # (256,) float32
        path = self._storage / f"{name}.npy"
        np.save(path, embed)
        logger.info(f"声纹已注册: {name} -> {path} ({embed.shape[0]} 维)")
        return path

    def verify(
        self,
        pcm: bytes,
        threshold: float = 0.75,
        sample_rate: int = 16000,
    ) -> tuple[bool, float, str | None]:
        """验证说话人：与所有已注册声纹比对，返回最匹配结果。

        返回 ``(是否通过, 最高相似度, 匹配用户名)``。

        - 声纹库为空（首次使用）→ ``(True, 1.0, None)`` 放行，提示注册
        - 模型不可用 → ``(True, 0.0, None)`` 放行（声纹禁用时不应拦用户）
        """
        if not list(self._storage.glob("*.npy")):
            return True, 1.0, None  # 无声纹库 → 首次使用，放行

        encoder = _get_encoder()
        if encoder is None:
            return True, 0.0, None  # 模型不可用 → 放行（降级）

        wav = _to_wav_array(pcm, sample_rate)
        if wav.size == 0:
            return False, 0.0, None

        embed = encoder.embed_utterance(wav)

        best_name: str | None = None
        best_score = 0.0
        for f in self._storage.glob("*.npy"):
            stored = np.load(f)
            sim = float(np.dot(embed, stored))  # 余弦相似度（向量已归一化）
            if sim > best_score:
                best_score, best_name = sim, f.stem

        passed = best_score >= threshold
        return passed, best_score, best_name

    def delete(self, name: str) -> bool:
        """删除已注册声纹。返回是否删除成功。"""
        path = self._storage / f"{name}.npy"
        if path.exists():
            path.unlink()
            logger.info(f"声纹已删除: {name}")
            return True
        return False

    def list_names(self) -> list[str]:
        """列出所有已注册的用户名。"""
        return sorted(f.stem for f in self._storage.glob("*.npy"))


def _to_wav_array(pcm_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """原始 PCM 16bit → float32 numpy 数组（resemblyzer 要求的输入格式）。

    使用 preprocess_wav 做静音裁剪 + 音量归一化，提升嵌入质量。
    """
    if not pcm_bytes or len(pcm_bytes) < 2:
        return np.array([], dtype=np.float32)
    try:
        from resemblyzer import preprocess_wav
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return preprocess_wav(audio, source_sr=sample_rate)
    except Exception as exc:
        logger.warning(f"preprocess_wav 失败，回退到原始音频: {exc}")
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
