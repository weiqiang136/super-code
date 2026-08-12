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
            if self._energy_threshold > 0 and is_speech:
                rms = _frame_rms(raw)
                if rms < self._energy_threshold:
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
    volcengine_app_id: str = "",
    volcengine_token: str = "",
    volcengine_cluster: str = "",
) -> str:
    """将原始 16kHz PCM 音频转成文字。

    Providers:
    - ``volcengine``（默认）：火山引擎一句话识别（极速版），HTTP 一次性请求。
      凭证：app_id + token + cluster（火山控制台"语音技术 → 应用管理"获取）。
    - ``aliyun_nls``：阿里云 NLS 一句话识别 RESTful API。
    - ``openai_whisper``：OpenAI Whisper API。回退到主 ``api_key`` / ``base_url``。

    返回识别文本；失败时返回 ``"[未识别]"``。
    """
    if not pcm or len(pcm) < FRAME_SIZE * 2:
        return ""

    if provider == "volcengine":
        return _recognize_volcengine(
            pcm,
            app_id=volcengine_app_id,
            token=volcengine_token,
            cluster=volcengine_cluster,
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
            app_id=volcengine_app_id,
            token=volcengine_token,
            cluster=volcengine_cluster,
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
    volcengine_app_id: str = "",
    volcengine_token: str = "",
    volcengine_cluster: str = "",
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
        volcengine_app_id=volcengine_app_id,
        volcengine_token=volcengine_token,
        volcengine_cluster=volcengine_cluster,
    )
