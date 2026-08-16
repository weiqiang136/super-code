"""语音 REPL 主循环（Phase 3）。

把 P1（录音+STT）、P1.5（声纹）、P2（TTS）串成完整的语音对话循环：
录音 → STT → 声纹验证 → engine.submit() → 流式 TTS → 下一轮。

架构要点：
- 本模块用独立的 ``asyncio.run()`` 入口（``voice_repl``），绝不与 prompt_toolkit 共享事件循环
- ``engine.submit()`` 是同步生成器：放入线程池执行，事件通过 queue 逐条传给协程，
  实现真正的流式（文本边生成边送 TTS，不必等整轮结束）
- 所有同步阻塞调用（STT 网络请求、TTS 合成播放）都用 ``asyncio.to_thread`` 扔出事件循环
- 文本按句子边界（标点）缓冲后送 TTS
- 打断为简化版（P4 实现 engine.abort 深度打断）
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from pathlib import Path

from core.engine import Engine
from support.voice import VoiceRecorder

logger = logging.getLogger(__name__)

# 句子边界标点：攒够一个句子才送 TTS，避免逐 token 合成
_SENTENCE_DELIMITERS = {"。", "！", "？", "；", "\n", "：", ".", "!", "?", ";"}

# 聆听超时（秒）：未检测到任何语音时，等这么久后放弃
LISTEN_TIMEOUT = 4.0
# 语音结束后持续静默（秒）：达到即判定"话说完了"，提交 STT
SILENCE_SETTLE = 1.2
# 单次聆听总上限（秒）：防环境底噪导致无限等
MAX_LISTEN = 8.0
# 打断验证的 RMS 阈值：暂停播放后测麦克风 RMS 峰值，
# 超过该值判为人声（打断），低于判为回声/环境噪音（恢复）。
# 依据实测：环境峰值 ~3900 < 5000 < 人声峰值 ~11000
INTERRUPT_RMS_THRESHOLD = 5000.0


# =============================================================================
# 人格系统（P_persona）
# =============================================================================

VOICE_PERSONA_PROMPT = {
    "role": "system",
    "content": """当前为语音对话模式。回复要求：

1. 用口语化短句，不要用 markdown 格式（不用 #、*、`、```、- 列表等）
2. 优先一两句给出结论。细节可以展开，但第一句必须直接回答问题
3. 不要说"当然可以""好的，让我来帮你"这类填充短语——直接做事
4. 代码文件名直接用中文描述（"config.py 文件"而不是 `config.py`）
5. 回复保持简洁，通常不超过 5 句话""",
}

# 人格风格 → 追加提示
PERSONA_STYLE_SUFFIX = {
    "butler": "语气保持英式管家的克制礼貌，偶尔可以带一点干幽默。称呼用户为'先生'。",
    "concise": "回复极度简洁，能一个字说清不用两个字。",
    "casual": "回复像朋友聊天一样自然随意。",
}


def build_voice_persona_prompt(style: str = "butler") -> dict:
    """构建语音人格系统提示词（基础约束 + 风格追加）。"""
    base = VOICE_PERSONA_PROMPT["content"]
    suffix = PERSONA_STYLE_SUFFIX.get(style)
    if suffix:
        base = base + "\n\n" + suffix
    return {"role": "system", "content": base}


# =============================================================================
# 句子缓冲（流式 TTS 的句子边界策略）
# =============================================================================

class SentenceBuffer:
    """按标点分句的文本缓冲：攒够一个句子再送 TTS。"""

    def __init__(self, delimiters: set[str] | None = None):
        self._delims = delimiters or _SENTENCE_DELIMITERS
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        """喂入文本片段，返回完整句子列表（可为空）。"""
        self._buf += chunk
        sentences: list[str] = []
        while True:
            # 找最早出现的分隔符
            best_idx = -1
            for d in self._delims:
                idx = self._buf.find(d)
                if idx != -1 and (best_idx == -1 or idx < best_idx):
                    best_idx = idx
            if best_idx == -1:
                break
            sentences.append(self._buf[:best_idx + 1])
            self._buf = self._buf[best_idx + 1:]
        return sentences

    def flush(self) -> str:
        """返回剩余缓冲（清空）。"""
        rest = self._buf
        self._buf = ""
        return rest

    @property
    def has_content(self) -> bool:
        return bool(self._buf.strip())


# =============================================================================
# 语音 REPL 主循环
# =============================================================================

def voice_repl(
    engine: Engine,
    console,
    *,
    stt_provider: str = "volcengine",
    stt_api_key: str = "",
    base_url: str = "",
    stt_params: dict | None = None,
    tts_voice: str = "zh-CN-YunxiNeural",
    speaker_verification: bool = False,
    speaker_threshold: float = 0.75,
    persona_style: str = "butler",
    voice_model: str = "",
    voiceprint_dir: str | Path | None = None,
) -> None:
    """语音对话主循环（同步入口，内部用独立 asyncio.run）。

    供 F2 快捷键 / ``--voice`` 启动调用。Ctrl+C 退出。
    """
    try:
        asyncio.run(_voice_repl_async(
            engine, console,
            stt_provider=stt_provider,
            stt_api_key=stt_api_key,
            base_url=base_url,
            stt_params=stt_params,
            tts_voice=tts_voice,
            speaker_verification=speaker_verification,
            speaker_threshold=speaker_threshold,
            persona_style=persona_style,
            voice_model=voice_model,
            voiceprint_dir=voiceprint_dir,
        ))
    except KeyboardInterrupt:
        console.print("\n[dim yellow]⏹ 语音模式已退出[/dim yellow]")
    except Exception as exc:
        logger.warning(f"语音模式异常退出: {exc}")
        console.print(f"\n[red]语音模式异常退出:[/red] {exc}")


async def _voice_repl_async(
    engine: Engine,
    console,
    *,
    stt_provider: str,
    stt_api_key: str,
    base_url: str,
    stt_params: dict | None,
    tts_voice: str,
    speaker_verification: bool,
    speaker_threshold: float,
    persona_style: str,
    voice_model: str,
    voiceprint_dir: str | Path | None,
) -> None:
    from support.voice import FRAME_SIZE, VoiceRecorder, speak_text
    from support.voiceprint import VoiceprintManager

    # ── 进入语音模式 banner ──
    console.print("[bold bright_magenta]🎤 语音模式已启动[/bold bright_magenta]")
    console.print("  ⚠️  文件/命令操作将自动执行（auto_approve）")
    if speaker_verification:
        console.print("  🔒 声纹验证已启用 · 仅注册声音可操作")
    console.print("  💡 推荐使用耳机以获得最佳体验")
    console.print("  Ctrl+C 退出 · 直接说话即可\n")

    # ── 声纹管理器（仅启用时；VoiceprintManager 复用 50MB 模型）──
    vpm = None
    if speaker_verification:
        vpm = VoiceprintManager(voiceprint_dir or (Path.home() / ".config" / "super-code"))

    # ── 全局停止标志：退出时通知 _run_engine_turn 停止 engine/TTS ──
    stop_event = threading.Event()

    # ── 人格注入：在现有 system 提示之后插入语音人格 ──
    persona_injected = False
    persona = build_voice_persona_prompt(persona_style)
    try:
        if engine._messages and engine._messages[0].get("role") == "system":
            engine._messages.insert(1, persona)
        else:
            engine._messages.insert(0, persona)
        persona_injected = True
    except Exception:
        pass

    # ── 模型切换：voice.model 非空 → 临时替换，退出恢复 ──
    original_model = engine._model
    if voice_model:
        engine._model = voice_model

    recorder = VoiceRecorder(vad_sensitivity=2)

    async def _speak(text: str) -> None:
        """把 TTS 播放扔出事件循环（speak_text 内部有 asyncio.run，不能在 loop 内直调）。"""
        if text and text.strip():
            await asyncio.to_thread(speak_text, text, tts_voice)

    try:
        while True:
            # ── 1. 等用户说话 → STT ──
            text = await _listen_and_transcribe(
                recorder, console,
                provider=stt_provider, api_key=stt_api_key,
                base_url=base_url, stt_params=stt_params,
            )
            if text is None:  # 用户中断（Ctrl+C 已在外部处理，这里兜底）
                break
            if not text:
                continue  # 未识别到内容 → 重新听

            console.print(f"[bold]你:[/bold] {text}")

            # ── 2. 声纹验证 ──
            if vpm is not None:
                passed, score, name = vpm.verify(
                    recorder.get_audio() or b"", threshold=speaker_threshold)
                if name is None:
                    console.print("🔊 声纹库为空，请先使用 --voice-enroll 注册")
                    await _speak("请先注册声纹")
                    continue
                if not passed:
                    console.print(f"[red]🔒 声纹不匹配 (相似度: {score:.2f})[/red]")
                    await _speak("声纹不匹配")
                    continue
                console.print(f"[dim]🔒 声纹验证通过 ({name}, {score:.2f})[/dim]")

            # ── 3. engine.submit + 流式 TTS（带深度打断 watchdog）──
            await _run_engine_turn(
                engine, text, console,
                tts_voice=tts_voice,
                recorder=recorder,
                stop_event=stop_event,
            )

            # ── 4. 短暂停顿后进入下一轮（P4 实现完整静默超时状态机）──
            await asyncio.sleep(0.5)
    finally:
        # ── 退出清理：先停 engine（可能有正在跑的轮次）──
        stop_event.set()
        try:
            engine.abort()
        except Exception:
            pass
        if persona_injected:
            try:
                for i, m in enumerate(engine._messages):
                    if m.get("role") == "system" and "语音对话模式" in str(m.get("content", "")):
                        engine._messages.pop(i)
                        break
            except Exception:
                pass
        if voice_model:
            engine._model = original_model
        try:
            recorder.cancel()
        except Exception:
            pass
        console.print("[dim yellow]🎤 语音模式已退出[/dim yellow]")


async def _listen_and_transcribe(
    recorder: VoiceRecorder,
    console,
    *,
    provider: str,
    api_key: str,
    base_url: str,
    stt_params: dict | None,
) -> str | None:
    """等待用户说话 → STT → 返回文本。未识别到内容返回空串。

    截断策略（解决"讲个笑话"被切成"讲个笑"）：
    - 未检测到任何语音：等 LISTEN_TIMEOUT 后放弃
    - 检测到语音开始：持续静默 SILENCE_SETTLE 秒才提交（等用户把话说完）
    - 总上限 MAX_LISTEN 秒（防止环境底噪导致无限等）
    """
    from support.voice import FRAME_SIZE, recognize_speech

    console.print("[dim]🎤 聆听中…（说话开始录音，静默自动结束）[/dim]")
    recorder.start()
    try:
        speech_started = False      # 是否已检测到语音
        silence_after_speech = 0.0  # 语音后持续静默时长
        start_time = time.time()

        while True:
            # 检测到语音结束（VAD 边界判定成功）
            if recorder._utterance_done.is_set():
                break

            # 已有语音 → 检查持续静默
            if speech_started and not recorder.is_speaking():
                silence_after_speech += 0.1
                if silence_after_speech >= SILENCE_SETTLE:
                    break  # 说完话后静默足够久 → 提交
            elif recorder.is_speaking():
                speech_started = True
                silence_after_speech = 0.0

            # 未说话超时 → 放弃
            if not speech_started and time.time() - start_time >= LISTEN_TIMEOUT:
                break
            # 总上限（防底噪无限等）
            if time.time() - start_time >= MAX_LISTEN:
                break

            await asyncio.sleep(0.1)

        recorder.cancel()
        # 优先用 VAD 边界判定结果；没有则取最近音频
        pcm = recorder.get_audio()
        if (not pcm or len(pcm) < FRAME_SIZE * 2) and speech_started:
            pcm = recorder.take_recent_audio(max_seconds=4.0)
        if not pcm or len(pcm) < FRAME_SIZE * 2:
            return ""
        # STT 是网络调用 → 线程池执行（recognize_speech 用关键字参数，用 partial）
        import functools
        fn = functools.partial(
            recognize_speech, pcm,
            provider=provider, api_key=api_key,
            base_url=base_url, stt_params=stt_params,
        )
        return await asyncio.to_thread(fn)
    except asyncio.CancelledError:
        recorder.cancel()
        return None


async def _vad_watchdog(
    recorder,
    interrupt_flag: threading.Event,
    suspicious: threading.Event,
    check_interval: float = 0.1,
) -> None:
    """深度打断 watchdog：后台持续检测麦克风 RMS，用户说话 → 设置打断标志。

    用 ``recorder.recent_rms() > INTERRUPT_RMS_THRESHOLD``（绝对阈值）判断，
    不用 VAD 判定——环境噪音（峰值 ~3900）会被 VAD 误判为有声，但 RMS 低于阈值。

    回声防御（暂停验证）：
    - 回声区（TTS 播放中/刚停 0.5s 内，可能有扬声器余音）→ 只置 ``suspicious``，
      由 TTS 播放线程暂停验证（回声随暂停消失 → 恢复；人声持续 → 打断）
    - 非播放期 → 要求 0.6s 持续高 RMS 才打断（回声残余 <0.3s，人声通常 >0.6s；
      随机短声音 <0.6s 不误触发）
    """
    consecutive = 0
    while True:
        if recorder.recent_rms() > INTERRUPT_RMS_THRESHOLD:
            consecutive += 1
            if recorder.is_echo_zone():
                # 播放中或刚停：可能扬声器余音 → 交播放线程暂停验证
                if consecutive >= 3:  # 0.3s
                    suspicious.set()
                    consecutive = 0
            else:
                # 非播放期：要求 0.6s 持续（滤掉余音与随机短声音）
                if consecutive >= 6:
                    interrupt_flag.set()
                    return
        else:
            consecutive = 0
        await asyncio.sleep(check_interval)


async def _run_engine_turn(
    engine: Engine,
    user_input: str,
    console,
    *,
    tts_voice: str,
    recorder=None,
    interrupt_threshold: float = 0.3,
    stop_event: threading.Event | None = None,
) -> bool:
    """执行一轮 engine.submit + 流式 TTS。

    engine.submit 是同步生成器 → 线程池执行；事件经 queue 逐条传回协程，
    文本按句子边界缓冲后立即送 TTS（边生成边朗读）。

    recorder 非空时启用深度打断：VAD 检测到说话 → 停 TTS + engine.abort()。
    stop_event 非空时：语音模式退出 → 立即停止 engine 子线程（不再生成/TTS）。
    返回 True 表示被用户打断。
    """
    from support.voice import speak_text

    event_q: queue.Queue = queue.Queue()
    sbuf = SentenceBuffer()
    had_tools = False
    spinner = None
    interrupted = False
    interrupt_flag = threading.Event()
    suspicious = threading.Event()  # 播放中检测到可疑声音（待验证）
    _last_verify_time = 0.0  # 上次暂停验证时刻（节流用）

    # TTS 播放串行锁：填充语/正文句子不能同时占扬声器（pyaudio 单流冲突）
    tts_lock = asyncio.Lock()

    def _interrupt_check() -> bool:
        """TTS 播放线程的 on_interrupt 回调（每 100ms 调用一次）。

        暂停验证（外放鲁棒）：suspicious 被设置 → 暂停播放 0.5s，
        比较暂停前后 RMS：
        - 回声：暂停前（播放中）RMS 高 → 暂停后（扬声器停）显著下降 → 恢复播放
        - 人声：暂停前 RMS 高（播放+人声）→ 暂停后仍高（人声不依赖扬声器）→ 打断
        - 环境噪音：前后都不高 → 恢复

        节流：1s 内最多验证一次（外放回声持续触发时避免 TTS 卡顿）。
        """
        nonlocal _last_verify_time
        if interrupt_flag.is_set() or (stop_event and stop_event.is_set()):
            return True
        if suspicious.is_set():
            suspicious.clear()
            now = time.time()
            if now - _last_verify_time < 1.0:
                return False  # 节流：1s 内不重复验证
            _last_verify_time = now
            if recorder is not None:
                before = recorder.recent_rms()  # 暂停前（播放中，含回声）
                time.sleep(0.5)                 # 暂停播放，回声衰减
                after = recorder.recent_rms()   # 暂停后
                # 暂停后仍高且无明显下降 → 人声（不依赖扬声器）→ 打断
                if after > INTERRUPT_RMS_THRESHOLD and after > before * 0.7:
                    interrupt_flag.set()
                    return True
            # 否则：回声（暂停后显著下降）或环境噪音 → 恢复播放
        return False

    async def _speak(text: str) -> None:
        """带锁的 TTS 播放（放线程池执行，串行不重叠）。

        打断时（interrupt_flag 被设置）→ on_interrupt 回调返回 True → 立即停播。
        """
        if text and text.strip():
            async with tts_lock:
                # 已被打断或语音模式退出 → 不再播放新句子
                if interrupt_flag.is_set() or (stop_event and stop_event.is_set()):
                    return
                # TTS 播放期间标记回声防御（提高 VAD 能量阈值滤回声）
                if recorder is not None:
                    recorder.set_tts_playing(True)
                try:
                    await asyncio.to_thread(
                        speak_text, text, tts_voice,
                        on_interrupt=_interrupt_check,
                    )
                finally:
                    if recorder is not None:
                        recorder.set_tts_playing(False)

    def _run_engine():
        """子线程：消费 engine.submit 生成器，事件逐个入队。

        stop_event 被设置（语音模式退出）→ 立即 abort engine 并结束。
        """
        try:
            for event in engine.submit(user_input):
                # 语音模式退出 → 停止
                if stop_event and stop_event.is_set():
                    try:
                        engine.abort()
                    except Exception:
                        pass
                    break
                event_q.put(event)
        except Exception as exc:
            event_q.put(("error", str(exc)))
        finally:
            event_q.put(None)  # 哨兵：结束

    from tui.rendering import SpinnerManager
    spinner = SpinnerManager(console)

    # ── 深度打断 watchdog ──
    # recorder 非空时：先重启采集（_listen_and_transcribe 结束时已 cancel），
    # 后台持续检测 VAD，用户说话 → interrupt_flag.set()。
    # 该 flag 同时作用于：TTS on_interrupt 回调（停播）、事件循环（engine.abort()）。
    vad_task = None
    if recorder is not None:
        try:
            recorder.start()  # 重新启动麦克风采集（打断监听用）
        except Exception as exc:
            logger.warning(f"打断监听启动失败: {exc}")
            recorder = None  # 降级：无法打断，但对话继续
    if recorder is not None:
        vad_task = asyncio.create_task(_vad_watchdog(recorder, interrupt_flag, suspicious))

    # 启动子线程跑 engine（不阻塞事件循环）
    loop = asyncio.get_event_loop()
    engine_thread = loop.run_in_executor(None, _run_engine)

    spinner.start("Thinking…")
    # 首 token 填充语：LLM 思考期间立即播"好的"，掩盖 2-5s 空白（与正文 TTS 共用串行锁）
    filler_task = asyncio.create_task(_speak("好的"))
    try:
        while True:
            # 语音模式退出 → 立即停止
            if stop_event and stop_event.is_set():
                interrupted = True
                break
            # 阻塞取事件（放线程池，避免卡事件循环）
            event = await asyncio.to_thread(event_q.get)
            if event is None:
                break
            etype = event[0]
            # ── 深度打断：用户说话 → abort engine + 停 TTS ──
            if interrupt_flag.is_set():
                interrupted = True
                console.print("\n[yellow]⏹ 已打断[/yellow]")
                try:
                    engine.abort()
                except Exception:
                    pass
                break
            if etype == "text":
                text_chunk = event[1]
                for sentence in sbuf.feed(text_chunk):
                    spinner.stop()
                    await _speak(sentence)
                    spinner.start("Thinking…")
            elif etype == "tool_call":
                had_tools = True
                console.print(f"  [dim]↳ {event[1]}[/dim]")
            elif etype == "tool_result":
                result = event[3]
                if result.is_error:
                    console.print(f"  [dim]↳ {event[1]}[/dim] [red]✗[/red]")
                else:
                    console.print(f"  [dim]↳ {event[1]}[/dim] [green]✓[/green]")
            elif etype == "error":
                console.print(f"[red]⚠️ 错误: {event[1]}[/red]")
                await _speak("出错了，请查看终端信息")
            elif etype == "thinking":
                spinner.start("Thinking…")
            elif etype == "compact":
                spinner.start("Compacting context…")
            elif etype == "waiting":
                spinner.start("Preparing tool call…")
    finally:
        spinner.stop()
        # 打断后：engine.submit 生成器还在线程里跑（可能阻塞在流读取），
        # abort() 已关 stream → 生成器会抛 AbortedError 结束。等它退出。
        if vad_task:
            vad_task.cancel()
        # 本轮结束：停止打断监听采集（下一轮 _listen_and_transcribe 会重新 start）
        if recorder is not None:
            try:
                recorder.cancel()
            except Exception:
                pass
        try:
            await engine_thread
        except Exception:
            pass

    # 若被打断，跳过 flush 和完成通知
    if interrupted:
        return True

    # flush 剩余缓冲
    rest = sbuf.flush()
    if rest.strip():
        await _speak(rest)

    # 任务完成通知（仅真正调用过工具才播）
    if had_tools:
        console.print("[dim]✅ 任务完成[/dim]")
        await _speak("完成了")

    return False
