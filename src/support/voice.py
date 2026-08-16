"""Phase 1: 音频采集 + VAD + 多 provider STT（语音转文字）。

VoiceRecorder — 基于 webrtcvad 的麦克风采集、语音边界检测、环形缓冲区。
recognize_speech() — 分发到 volcengine / aliyun_nls / openai_whisper。
"""

from __future__ import annotations

import io
import threading
import time
import wave
from collections import deque

import numpy as np

# ── pyaudio + webrtcvad 在此处急切导入，因为整个模块只有在语音模式激活时才被加载
#    （app.py 中的惰性导入守卫）。
try:
    import pyaudio
except ImportError:
    pyaudio = None  # type: ignore

try:
    import webrtcvad
except ImportError:
    webrtcvad = None  # type: ignore

# ── 常量 ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000       # 采样率（Hz）
FRAME_DURATION_MS = 30    # 帧时长（ms，webrtcvad 仅支持 10/20/30）
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 采样点
CHANNELS = 1              # 单声道
SAMPLE_WIDTH = 2          # 位宽（字节），16-bit PCM

# 语音边界检测阈值
SPEECH_START_FRAMES = 5    # 连续有声帧数，确认语音开始
SILENCE_END_FRAMES = 10    # 连续无声帧数，判定语音结束（~300ms）
PRE_SPEECH_BUFFER_MS = 200  # 保留语音开始前的音频时长
PRE_SPEECH_FRAMES = max(1, PRE_SPEECH_BUFFER_MS // FRAME_DURATION_MS)
MAX_UTTERANCE_SECONDS = 10.0  # 单条语句最大时长（VAD 一直判有声时的强制收尾兜底）


def _frame_rms(raw: bytes) -> float:
    """计算一帧 16-bit PCM 的 RMS 能量（用于底噪过滤）。"""
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(float(np.mean(samples * samples))))


# =============================================================================
# VoiceRecorder
# =============================================================================

class VoiceRecorder:
    """基于 VAD 的麦克风采集 + 语音边界检测。

    在后台线程中持续从麦克风读取 PCM 帧，逐帧送入 webrtcvad，
    根据语音活动切分出完整语句。

    用法::

        rec = VoiceRecorder(vad_sensitivity=2)
        rec.start()
        # … 等待用户说话 …
        rec.stop()          # 等待语句检测完成
        pcm = rec.get_audio()  # 获取原始 PCM，再交给 recognize_speech()
    """

    def __init__(self, vad_sensitivity: int = 2, energy_threshold: float = 2000.0) -> None:
        if pyaudio is None:
            raise ImportError(
                "语音模式需要 pyaudio。"
                "安装方式：pip install pyaudio"
            )
        if webrtcvad is None:
            raise ImportError(
                "语音模式需要 webrtcvad。"
                "安装方式：pip install webrtcvad"
            )

        self._vad = webrtcvad.Vad(vad_sensitivity)
        # 能量阈值（RMS）：低于该值的帧强制判为静音，过滤环境底噪。
        # 0 = 不启用能量滤波（纯 VAD）。默认 2000：底噪 RMS~1200 被滤除，语音 RMS 通常 >3000。
        self._energy_threshold = energy_threshold
        # TTS 播放时的能量阈值（回声防御）：播放期间调高，滤掉扬声器回声。
        # 初始与普通阈值相同，_set_tts_playing(True) 时 ×2.5。
        self._tts_playing = False
        self._tts_stopped_at: float | None = None  # 上次 TTS 停止时刻（回声保护期）
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        # 环形缓冲区：保存 (pcm_bytes, is_speech) 元组，用于边界检测
        self._ring: deque[tuple[bytes, bool]] = deque(maxlen=200)
        # 当前语句累积的原始 PCM 分块
        self._utterance_chunks: list[bytes] = []
        self._speaking = False
        self._utterance_done = threading.Event()
        self._utterance_pcm: bytes | None = None

        # 保护 _ring / _speaking 的线程安全锁
        self._lock = threading.Lock()

    # ── 公共 API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """启动后台音频采集。非阻塞。"""
        if self._running:
            return
        self._running = True
        self._utterance_done.clear()
        self._utterance_pcm = None
        self._utterance_chunks.clear()
        self._ring.clear()

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
            stream_callback=None,
        )
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止录音，等待当前语句检测完成。

        阻塞直到检测到语句结束（静默超时）或被外部打断。
        停止后通过 get_audio() 获取 PCM，再调用 recognize_speech() 转文字。
        """
        if not self._running:
            return None
        # 等待语句完成（上限 60 秒，防止挂死）
        self._utterance_done.wait(timeout=60.0)
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def is_speaking(self) -> bool:
        """用户当前是否在说话（VAD 生效中）？"""
        with self._lock:
            return self._speaking

    def set_tts_playing(self, playing: bool) -> None:
        """标记 TTS 是否正在播放（回声防御）。

        TTS 播放期间：能量阈值 ×2.5，扬声器回声大部分被滤；
        停止时记录时刻，供 is_echo_zone() 判断"回声保护期"。
        """
        with self._lock:
            self._tts_playing = playing
            if not playing:
                self._tts_stopped_at = time.time()

    def is_tts_playing(self) -> bool:
        """TTS 是否正在播放（供打断 watchdog 查询）。"""
        with self._lock:
            return self._tts_playing

    def is_echo_zone(self, protection_seconds: float = 0.5) -> bool:
        """是否处于"回声区"：TTS 播放中，或刚停止 protection_seconds 内。

        TTS 停止后扬声器仍有物理余音（~0.3s），watchdog 在此窗口内
        检测到高 RMS 不应直接打断（可能是余音），只标记待验证。
        """
        with self._lock:
            if self._tts_playing:
                return True
            if self._tts_stopped_at is not None:
                return time.time() - self._tts_stopped_at < protection_seconds
            return False

    def recent_rms(self, window_frames: int = 10) -> float:
        """最近 window_frames 帧（~300ms）的 RMS 峰值。

        用于暂停验证：TTS 暂停后回声消失 → RMS 降到环境水平（<5000）；
        用户说话 → RMS 峰值显著高于环境（>5000）。
        """
        with self._lock:
            if not self._ring:
                return 0.0
            frames = list(self._ring)[-window_frames:]
        peak = 0.0
        for raw, _ in frames:
            r = _frame_rms(raw)
            if r > peak:
                peak = r
        return peak

    def has_recent_speech(self, window_frames: int = 10) -> bool:
        """最近 window_frames 帧（~300ms）内是否有 VAD 判为有声的帧。

        比 is_speaking() 更灵敏：is_speaking 是"一段语音的持续状态"（需 5 帧
        才进入、10 帧无声才退出），在打断场景（用户短喊"停止"）可能采样不到。
        本方法直接查环形缓冲区，任何一帧有声即返回 True。
        """
        with self._lock:
            if not self._ring:
                return False
            frames = list(self._ring)[-window_frames:]
            return any(speech for _, speech in frames)

    def get_audio(self) -> bytes | None:
        """返回最近一条语句的原始 PCM 音频，无则返回 None。"""
        return self._utterance_pcm

    def cancel(self) -> None:
        """立即停止录音，不等待语句结束。"""
        self._running = False
        self._utterance_done.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    # ── 内部方法 ───────────────────────────────────────────────────────────

    def _record_loop(self) -> None:
        """后台线程：读帧 → VAD → 边界检测。"""
        speech_run: int = 0      # 连续有声帧计数
        silence_run: int = 0     # 连续无声帧计数
        local_speaking = False
        utterance_start_idx: int | None = None
        utterance_start_time: float = 0.0  # 语句开始时刻（防止 VAD 一直判有声时死等）

        while self._running:
            try:
                raw = self._stream.read(FRAME_SIZE, exception_on_overflow=False)
            except Exception:
                break

            if len(raw) < FRAME_SIZE * SAMPLE_WIDTH:
                continue

            is_speech = self._vad.is_speech(raw, SAMPLE_RATE)
            # 能量滤波：帧 RMS 低于阈值 → 强制判为静音（滤除环境底噪/电流声）
            # TTS 播放期间阈值 ×2.5（≈5000）：滤掉扬声器回声（P90~900），
            # 用户近距离喊话（P90~4500，峰值~11000）仍能通过
            if self._energy_threshold > 0 and is_speech:
                rms = _frame_rms(raw)
                effective = self._energy_threshold * (2.5 if self._tts_playing else 1)
                if rms < effective:
                    is_speech = False

            with self._lock:
                self._ring.append((raw, is_speech))
                self._speaking = local_speaking

            if is_speech:
                speech_run += 1
                silence_run = 0
                if not local_speaking and speech_run >= SPEECH_START_FRAMES:
                    # 语音开始：记录起始索引（含开始前保留的音频）
                    local_speaking = True
                    utterance_start_time = time.time()
                    utterance_start_idx = max(
                        0, len(self._ring) - speech_run - PRE_SPEECH_FRAMES
                    )
            else:
                silence_run += 1
                speech_run = 0
                if local_speaking and silence_run >= SILENCE_END_FRAMES:
                    # 语音结束：从环形缓冲区取出整段音频
                    local_speaking = False
                    self._finalize_utterance(utterance_start_idx)
                    utterance_start_idx = None
                    silence_run = 0
                    speech_run = 0

            # 最大语句时长兜底：连续说话超过 MAX_UTTERANCE_SECONDS 强制收尾，
            # 防止 VAD 因环境底噪一直判有声而永远等不到"语音结束"
            if local_speaking and time.time() - utterance_start_time >= MAX_UTTERANCE_SECONDS:
                local_speaking = False
                self._finalize_utterance(utterance_start_idx)
                utterance_start_idx = None
                silence_run = 0
                speech_run = 0

    def _finalize_utterance(self, start_idx: int | None) -> None:
        """从环形缓冲区取出语音段 PCM，并通知语句完成。"""
        with self._lock:
            ring_snapshot = list(self._ring)
        if start_idx is None:
            start_idx = 0
        # 收集从 start_idx 起的全部帧
        chunks: list[bytes] = []
        for i in range(start_idx, len(ring_snapshot)):
            chunks.append(ring_snapshot[i][0])
        self._utterance_pcm = b"".join(chunks)
        self._utterance_done.set()

    def take_recent_audio(self, max_seconds: float = 4.0) -> bytes:
        """从环形缓冲区取最近 max_seconds 秒音频（VAD 未判结束时兜底用）。

        用于 VAD 边界检测失灵（说话音量低于阈值）时，强制截取缓冲区音频。
        返回 PCM bytes；缓冲区为空返回 b""。
        """
        with self._lock:
            ring_snapshot = list(self._ring)
        if not ring_snapshot:
            return b""
        max_frames = int(max_seconds * 1000 / FRAME_DURATION_MS)
        frames = ring_snapshot[-max_frames:]
        return b"".join(f[0] for f in frames)

    def has_speech_frames(self) -> bool:
        """环形缓冲区中是否存在 VAD 判为有声的帧（判断用户是否说过话）。"""
        with self._lock:
            return any(speech for _, speech in self._ring)


# =============================================================================
# PCM → WAV 辅助函数（内存中转换）
# =============================================================================

def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """将原始 16-bit 单声道 PCM 转为 WAV 文件（bytes）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# =============================================================================
# STT: 多 provider 语音识别
# =============================================================================

def recognize_speech(
    pcm: bytes,
    *,
    provider: str = "volcengine",
    api_key: str = "",
    base_url: str = "",
    stt_params: dict | None = None,
) -> str:
    """将原始 16kHz PCM 音频转成文字。

    Providers:
    - ``volcengine``（默认）：火山引擎一句话识别（极速版），HTTP 一次性请求。
      凭证在 ``stt_params``：{app_id, token, cluster}（火山控制台"语音技术 → 应用管理"获取）。
    - ``aliyun_nls``：阿里云 NLS 一句话识别 RESTful API。
    - ``openai_whisper``：OpenAI Whisper API。回退到主 ``api_key`` / ``base_url``。

    ``stt_params`` 为厂商私有参数字典，各 provider 自己取用，互不干扰——
    换厂商只需改 ``provider`` 和对应参数，调用方无需关心其他厂商的字段。
    返回识别文本；失败时返回 ``"[未识别]"``。
    """
    if not pcm or len(pcm) < FRAME_SIZE * 2:
        return ""

    params: dict = stt_params or {}

    if provider == "volcengine":
        return _recognize_volcengine(
            pcm,
            app_id=params.get("app_id", ""),
            token=params.get("token", ""),
            cluster=params.get("cluster", ""),
        )
    elif provider == "aliyun_nls":
        return _recognize_aliyun_nls(pcm, api_key=api_key)
    elif provider == "openai_whisper":
        return _recognize_openai_whisper(pcm, api_key=api_key, base_url=base_url)
    else:
        # 未知 provider → 回退 volcengine 并告警
        import logging
        logging.warning(f"未知 STT provider '{provider}'，回退到 volcengine")
        return _recognize_volcengine(
            pcm,
            app_id=params.get("app_id", ""),
            token=params.get("token", ""),
            cluster=params.get("cluster", ""),
        )


# ── provider 实现 ────────────────────────────────────────────────────────────

def _recognize_volcengine(
    pcm: bytes,
    *,
    app_id: str,
    token: str,
    cluster: str,
) -> str:
    """火山引擎一句话识别（极速版）。

    端点：``https://openspeech.bytedance.com/api/v1/asr``（2026-08-11 实测确认）
    认证：``Authorization: Bearer; {token}``
    请求体：app / user / audio / request 四段 JSON（与流式识别同构），
    音频 base64 内嵌在 ``audio.data`` 一次性发送。
    成功码：``code == 1000``（非 0）。
    """
    import base64
    import json
    import uuid

    if not app_id or not token:
        return "[STT 错误: 缺少火山引擎 app_id/token]"

    wav_bytes = pcm_to_wav(pcm)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    body = {
        "app": {
            "appid": app_id,
            "token": token,
            "cluster": cluster or "volcano_icloud_common",
        },
        "user": {
            "uid": "super-code",
        },
        "audio": {
            "format": "wav",
            "rate": SAMPLE_RATE,
            "bits": 16,
            "channel": 1,
            "codec": "raw",
            "data": audio_b64,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "workflow": "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
            "nbest": 1,
            "result_type": "full",
            "sequence": 1,  # 必需：缺少会报 "req.sequence does not exist"
        },
    }

    try:
        import httpx
    except ImportError:
        return "[STT 错误: httpx 未安装]"

    url = "https://openspeech.bytedance.com/api/v1/asr"
    headers = {
        "Authorization": f"Bearer; {token}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(
            url,
            content=json.dumps(body),
            headers=headers,
            timeout=30.0,
        )
        if resp.status_code != 200:
            import logging
            logging.warning(f"火山引擎一句话识别 HTTP {resp.status_code}: {resp.text[:200]}")
            return "[未识别]"

        data = resp.json()
        code = data.get("code", -1)
        if code != 1000:
            import logging
            logging.warning(f"火山引擎一句话识别 code={code}: {data.get('message', '')}")
            return "[未识别]"

        # result 可能是字符串 / dict / list[{text, confidence}, ...]，兼容三种
        result = data.get("result", "")
        if isinstance(result, dict):
            result = result.get("text") or result.get("result") or ""
        elif isinstance(result, list):
            texts = [
                str(item["text"]) for item in result
                if isinstance(item, dict) and item.get("text")
            ]
            result = " ".join(texts)
        return str(result).strip()
    except Exception as exc:
        import logging
        logging.warning(f"火山引擎一句话识别异常: {exc}")
        return "[未识别]"


def _recognize_aliyun_nls(pcm: bytes, *, api_key: str) -> str:
    """阿里云 NLS 一句话识别（RESTful，非 WebSocket）。

    使用 ``/v1/recognize`` 端点 + token 认证。
    ``api_key`` 是 AccessKeySecret；AccessKeyId 需单独配置或从环境变量
    ``ALIBABA_CLOUD_ACCESS_KEY_ID`` 读取。

    注意：完整实现需要签名（HMAC-SHA1）——当前为占位实现。
    """
    try:
        import httpx
    except ImportError:
        return "[STT 错误: httpx 未安装]"
    import os

    access_key_id = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
    if not access_key_id or not api_key:
        return "[STT 错误: 缺少阿里云 AccessKey]"

    wav_bytes = pcm_to_wav(pcm)

    # 构建签名（HMAC-SHA1）
    # https://help.aliyun.com/document_detail/324260.html
    try:
        # 简化版：生产环境请使用官方 SDK 或完整签名
        import requests

        url = "https://nls-slp.cn-shanghai.aliyuncs.com/v1/recognize"
        # 占位——真实实现需要 STS token 或完整 AK 签名
        resp = requests.post(
            url,
            headers={"Content-Type": "audio/wav"},
            data=wav_bytes,
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.json().get("result", "").strip()
        else:
            import logging
            logging.warning(f"阿里云 NLS 错误 {resp.status_code}: {resp.text[:200]}")
            return "[未识别]"
    except Exception as exc:
        import logging
        logging.warning(f"阿里云 NLS 异常: {exc}")
        return "[未识别]"


def _recognize_openai_whisper(pcm: bytes, *, api_key: str, base_url: str) -> str:
    """OpenAI Whisper API（标准 audio.transcriptions.create）。"""
    try:
        from openai import OpenAI
    except ImportError:
        return "[STT 错误: openai 库未安装]"

    if not api_key:
        return "[STT 错误: 缺少 API key]"

    wav_bytes = pcm_to_wav(pcm)
    try:
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        import io as _io
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("recording.wav", _io.BytesIO(wav_bytes), "audio/wav"),
            language="zh",
        )
        return transcript.text.strip() if transcript.text else ""
    except Exception as exc:
        import logging
        logging.warning(f"OpenAI Whisper 异常: {exc}")
        return "[未识别]"


# =============================================================================
# 快速测试辅助（--voice-test CLI）
# =============================================================================

def test_record(
    api_key: str = "",
    provider: str = "volcengine",
    base_url: str = "",
    stt_params: dict | None = None,
) -> str:
    """从麦克风录音直到静默（~5s 上限），转文字，返回文本。

    供 ``--voice-test`` CLI 参数使用。未检测到语音返回 ``"[超时]"``。
    """
    rec = VoiceRecorder(vad_sensitivity=2)
    rec.start()
    print("🎤 请说话…（检测到语音会自动结束录音）")
    # 等待语句完成（检测到语音结束即返回；上限 10s）
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if rec._utterance_done.is_set():
            break
        time.sleep(0.1)
    rec.cancel()  # 立即停止，不依赖 stop() 的 60s 等待
    pcm = rec.get_audio()
    if not pcm or len(pcm) < FRAME_SIZE * 2:
        return "[超时] 未检测到语音"
    return recognize_speech(
        pcm,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        stt_params=stt_params,
    )


def voice_enroll(name: str, storage_dir: str | None = None) -> str:
    """声纹注册：录 3 秒语音 → 提取嵌入向量 → 保存到声纹库。

    供 ``--voice-enroll NAME`` CLI 使用。
    ``storage_dir`` 缺省为 ``~/.config/super-code``（与全局配置同目录）。
    返回结果消息（成功/失败），由调用方打印。
    """
    from pathlib import Path as _Path
    from support.voiceprint import VoiceprintManager

    if storage_dir is None:
        storage_dir = _Path.home() / ".config" / "super-code"
    mgr = VoiceprintManager(storage_dir)

    print("🔊 请说 'super-code 语音验证'（录音 3 秒）…")
    rec = VoiceRecorder(vad_sensitivity=2)
    rec.start()
    # 录 3 秒（或检测到语音结束提前返回）
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if rec._utterance_done.is_set():
            break
        time.sleep(0.1)
    rec.cancel()
    pcm = rec.get_audio()
    if not pcm or len(pcm) < FRAME_SIZE * 2:  # < 30ms 视为无效
        return "❌ 未检测到语音，注册取消"
    path = mgr.enroll(name, pcm)
    if path is None:
        # enroll 返回 None 有两种原因：模型不可用 或 音频无效
        from support.voiceprint import _get_encoder
        if _get_encoder() is None:
            return "❌ 声纹模型不可用，注册失败"
        return "❌ 音频无效（可能太短或太安静），请重新录音"
    return f"✅ 声纹 '{name}' 注册成功 → {path}"


# =============================================================================
# TTS：Edge-TTS 语音合成 + 播放（Phase 2）
# =============================================================================

# 播放采样率：pydub 转码后的 PCM 输出格式（Edge-TTS 默认 24kHz）
TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2  # 16bit

DEFAULT_TTS_VOICE = "zh-CN-YunxiNeural"  # 云希男声（默认，管家感）


def _ensure_ffmpeg_in_path() -> None:
    """确保 pydub 能调起 ffmpeg：找不到时把常见安装位置加入 PATH。

    winget 安装的 ffmpeg 在 ``%LOCALAPPDATA%\\Microsoft\\WinGet\\Links``，
    但当前进程的 PATH 可能不含该目录（旧终端 / 直接跑脚本），
    pydub 调 ffmpeg 会 FileNotFoundError。这里兜底补上。
    """
    import os
    import shutil
    from pathlib import Path as _Path

    if shutil.which("ffmpeg"):
        return  # 已在 PATH，无需处理

    candidates = [
        _Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
        _Path.home() / "scoop" / "shims",
        _Path("C:/ffmpeg/bin"),
        _Path("C:/Program Files/ffmpeg/bin"),
    ]
    for d in candidates:
        if (d / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return


def edge_tts_stream(
    text: str,
    voice: str = DEFAULT_TTS_VOICE,
) -> "async_iterator":
    """调用 Edge-TTS，流式返回 PCM 音频块（24kHz/16bit/mono）。

    边下载 MP3 边用 pydub 转 PCM，无需缓存完整文件。
    用法::

        async for pcm_chunk in edge_tts_stream("你好"):
            play_pcm(pcm_chunk)
    """
    import asyncio
    import io as _io

    async def _gen():
        try:
            import edge_tts
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "TTS 需要 edge-tts + pydub。安装: pip install edge-tts pydub"
            )
        _ensure_ffmpeg_in_path()

        communicate = edge_tts.Communicate(text, voice)
        # edge_tts 的 stream() 是异步生成器，产出 {"type": "audio", "data": mp3_bytes}
        # ⚠️ 不能逐 chunk 解码：MP3 是分块流，单个 chunk 可能不完整（帧被切两半），
        # pydub 逐块解码会报 "Failed to find two consecutive MPEG audio frames"。
        # 正确做法：累积全部 MP3 字节，一次性解码成 PCM。
        # try/finally 确保 aiohttp session 关闭，避免 asyncio 泄漏
        mp3_buffer = bytearray()
        try:
            async for chunk in communicate.stream():
                if chunk["type"] != "audio":
                    continue
                mp3_bytes = chunk["data"]
                if not mp3_bytes:
                    continue
                mp3_buffer.extend(mp3_bytes)
            if mp3_buffer:
                # 全部 MP3 收集完，一次性解码（pydub 内部调 ffmpeg）
                seg = AudioSegment.from_mp3(_io.BytesIO(bytes(mp3_buffer)))
                # 统一转 24kHz/16bit/mono PCM
                seg = seg.set_frame_rate(TTS_SAMPLE_RATE).set_channels(1).set_sample_width(2)
                yield seg.raw_data
        finally:
            # edge-tts Communicate 提供 close() 关闭底层 session
            try:
                communicate.close()
            except Exception:
                pass

    return _gen()


def play_audio_chunks(
    chunks,
    on_interrupt=None,
    sample_rate: int = TTS_SAMPLE_RATE,
) -> bool:
    """用 pyaudio 流式播放 PCM 音频块。返回是否被中断。

    每个 chunk 播放前调用 ``on_interrupt()``，若返回 True 则停止播放。

    - ``chunks``：PCM bytes 的迭代器（如 edge_tts_stream 的异步产出，需先收集或适配）
    - ``on_interrupt``：无参回调，返回 True 表示应停止
    """
    if pyaudio is None:
        raise ImportError("播放音频需要 pyaudio。安装: pip install pyaudio")

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=TTS_CHANNELS,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024,
        )
        interrupted = False
        # 每 100ms 切成一个小块播放：chunk 之间检查 on_interrupt。
        # ⚠️ 不能整块 write——edge_tts_stream 一次性产出整段 PCM，
        # 若整块写，打断检查只在播放前/后触发，播放中无法中断。
        chunk_bytes = sample_rate * TTS_SAMPLE_WIDTH * TTS_CHANNELS // 10  # 100ms
        for chunk in chunks:
            if not chunk:
                continue
            for offset in range(0, len(chunk), chunk_bytes):
                if on_interrupt and on_interrupt():
                    interrupted = True
                    break
                stream.write(chunk[offset:offset + chunk_bytes])
            if interrupted:
                break
        stream.stop_stream()
        stream.close()
        return interrupted
    finally:
        pa.terminate()


def speak_text(
    text: str,
    voice: str = DEFAULT_TTS_VOICE,
    on_interrupt=None,
) -> bool:
    """合成并播放一段文本（同步）。返回是否被中断。

    供 CLI 测试和 voice_repl 使用。内部将异步的 edge_tts_stream 收集后同步播放。

    :param text: 要朗读的文本
    :param voice: Edge-TTS 音色
    :param on_interrupt: 无参回调，返回 True 停止播放
    :return: True 表示播放被中断
    """
    import asyncio

    def _sync_play():
        # 同步方式：edge_tts 是异步 API，用 asyncio.run 收集 PCM
        chunks: list[bytes] = []

        async def _collect():
            gen = edge_tts_stream(text, voice)
            try:
                async for pcm in gen:
                    chunks.append(pcm)
            finally:
                # 显式关闭异步生成器，避免 "Task was destroyed but it is pending" 泄漏
                try:
                    await gen.aclose()
                except Exception:
                    pass

        try:
            asyncio.run(_collect())
        except Exception as exc:
            import logging
            logging.warning(f"TTS 合成失败: {exc}")
            return False

        if not chunks:
            return False
        return play_audio_chunks(chunks, on_interrupt=on_interrupt)

    return _sync_play()


def tts_test() -> str:
    """TTS 自测：朗读测试句子。供 ``--voice-tts-test`` CLI 使用。"""
    test_text = "你好，我是super-code，语音功能测试成功"
    print(f"🔊 正在朗读: {test_text}")
    speak_text(test_text)
    return "✅ TTS 测试完成"
