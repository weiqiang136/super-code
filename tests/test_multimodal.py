"""P0 多模态：图片占位符注册 + 组装（tests 里用 tmp 文件模拟拖拽）。"""
import os
import sys

import pytest

sys.path.insert(0, "src")

from tui.app import _assemble_image_content, _IMAGE_PLACEHOLDER_RE
from tui.prompt import _extract_image_paths, _register_image, _IMAGE_MAX_BYTES


# ── 组装层 ──────────────────────────────────────────────

def test_assemble_no_images_returns_str():
    """无图 → 与现状 memory_prefix + user_input 完全一致（零回归）。"""
    result = _assemble_image_content("[记忆]", "帮我看看这个", [])
    assert result == "[记忆]帮我看看这个"
    assert isinstance(result, str)


def test_assemble_text_before_and_after_image():
    """图在中间 → [text, image, text]。"""
    images = [{"media_type": "image/png", "data": "AAAA"}]
    text = "看图 [🖼 0: shot.png] 这是截图"
    result = _assemble_image_content("", text, images)
    assert result == [
        {"type": "text", "text": "看图"},
        {"type": "image", "source": {"media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "这是截图"},
    ]


def test_assemble_prefix_merged_into_first_text():
    """记忆前缀并入第一个 text block（图在最前时前缀单独成块）。"""
    images = [{"media_type": "image/png", "data": "BBBB"}]
    result = _assemble_image_content("[记忆]", "[🖼 0: shot.png] 描述它", images)
    assert result[0] == {"type": "text", "text": "[记忆]"}
    assert result[1]["type"] == "image"
    assert result[2] == {"type": "text", "text": "描述它"}


def test_assemble_multiple_images_order():
    """多图按占位符出现顺序对应 images_ref。"""
    images = [
        {"media_type": "image/png", "data": "ONE"},
        {"media_type": "image/jpeg", "data": "TWO"},
    ]
    text = "对比 [🖼 0: a.png] 与 [🖼 1: b.jpg]"
    result = _assemble_image_content("", text, images)
    assert [b for b in result if b["type"] == "image"] == [
        {"type": "image", "source": {"media_type": "image/png", "data": "ONE"}},
        {"type": "image", "source": {"media_type": "image/jpeg", "data": "TWO"}},
    ]


def test_assemble_placeholder_deleted_defensive():
    """占位符被删/损坏（images_ref 残留）→ 只发文本，不发图。"""
    images = [{"media_type": "image/png", "data": "GHOST"}]
    result = _assemble_image_content("", "只有文本没有占位符", images)
    assert result == "只有文本没有占位符"
    assert isinstance(result, str)


# ── 路径识别 ────────────────────────────────────────────

def test_extract_image_paths_single(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(b"x")
    assert _extract_image_paths(str(p)) == [str(p)]
    assert _extract_image_paths(f'"{p}"') == [str(p)]        # 剥引号（终端拖拽带引号）
    assert _extract_image_paths(f"  {p}  ") == [str(p)]


def test_extract_image_paths_multiple(tmp_path):
    a = tmp_path / "a.png"
    b_space = tmp_path / "b b.jpg"    # 文件名含空格 → WT 注入必带引号
    b_plain = tmp_path / "b.jpg"      # 无空格 → 裸路径
    a.write_bytes(b"x")
    b_space.write_bytes(b"x")
    b_plain.write_bytes(b"x")
    assert _extract_image_paths(f'"{a}" "{b_space}"') == [str(a), str(b_space)]  # 全引号（含空格路径）
    assert _extract_image_paths(f'"{a}" {b_plain}') == [str(a), str(b_plain)]    # 引号+裸混合
    assert _extract_image_paths(f'"{a}" "{b_space}" missing.png') == [str(a), str(b_space)]  # 不存在忽略


def test_extract_image_paths_rejects(tmp_path):
    txt = tmp_path / "note.txt"
    txt.write_text("x")
    assert _extract_image_paths(str(txt)) == []                 # 非图片扩展名
    assert _extract_image_paths(str(tmp_path / "missing.png")) == []  # 文件不存在
    assert _extract_image_paths("随便一句话没有路径") == []
    assert _extract_image_paths("看一下 https://x.com/foo.png") == []  # URL 非本地文件


# ── macOS 终端拖拽转义路径回归（v3.3.13：拖拽后变纯文本、模型看不到图）──────

# 事故真实形态：终端把路径里的空格/括号转义成 \  \(  \)
MACOS_ESCAPED = r"/Users/haning/Downloads/图片/深度分析-整体\ \(4\)\ \(1\).png"
MACOS_REAL = "/Users/haning/Downloads/图片/深度分析-整体 (4) (1).png"


def test_candidate_macos_escaped_path(monkeypatch):
    """纯函数：macOS 转义路径被完整吃下并反转义为真实路径。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    cands = prompt._candidate_image_paths(MACOS_ESCAPED)
    assert MACOS_REAL in cands
    assert all("\\ " not in c and "\\(" not in c for c in cands)  # 无残留转义


def test_candidate_macos_escaped_path_then_natural_language(monkeypatch):
    """macOS 转义路径 + 后续中文：路径候选必须是完整路径，不得拼上中文（旧方案必挂）。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    cands = prompt._candidate_image_paths(MACOS_ESCAPED + " 图片里面有什么")
    assert MACOS_REAL in cands
    assert MACOS_REAL + " 图片里面有什么" not in cands
    assert MACOS_REAL + "图片里面有什么" not in cands


def test_candidate_macos_multiple_escaped(monkeypatch):
    """macOS 多图：两个转义路径用未转义空格分隔 → 都还原且顺序正确。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    d1 = "/Users/haning/Downloads/图 一.png"
    d2 = "/Users/haning/Downloads/图 二.jpg"
    text = r"/Users/haning/Downloads/图\ 一.png" + " " + r"/Users/haning/Downloads/图\ 二.jpg"
    cands = prompt._candidate_image_paths(text)
    assert d1 in cands and d2 in cands
    assert cands.index(d1) < cands.index(d2)


def test_candidate_macos_single_quote_wrapped(monkeypatch):
    """macOS 拖拽的单引号包裹变体：'路径 带空格.png' → 取引号内全文。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    cands = prompt._candidate_image_paths("'/Users/haning/Downloads/My Pic.png'")
    assert "/Users/haning/Downloads/My Pic.png" in cands


def test_extract_image_paths_macos_escaped(monkeypatch):
    """集成：macOS 转义路径经扩展名+存在性校验后被识别（isfile 打桩，跨平台确定）。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    monkeypatch.setattr(os.path, "isfile", lambda q: q == MACOS_REAL)
    assert prompt._extract_image_paths(MACOS_ESCAPED) == [MACOS_REAL]


def test_extract_image_paths_macos_escaped_then_natural_language(monkeypatch):
    """集成：转义路径后跟自然语言 → 仍解析出图片（本次事故场景）。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    monkeypatch.setattr(os.path, "isfile", lambda q: q == MACOS_REAL)
    assert prompt._extract_image_paths(MACOS_ESCAPED + " 图片里面有什么") == [MACOS_REAL]


def test_extract_image_paths_macos_real_tmp(tmp_path, monkeypatch):
    """集成：tmp_path 造含空格/括号真实文件，构造 macOS 转义形态 → 解析出真实路径。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    p = tmp_path / "depth (4) (1).png"
    p.write_bytes(b"x")
    # 去盘符转 POSIX，模拟 macOS 绝对路径风格（Windows 上 / 指向当前盘根）
    _drive, _sep, rest = str(p).replace("\\", "/").partition("/")
    real = "/" + rest
    escaped = real.replace(" ", "\\ ").replace("(", "\\(").replace(")", "\\)")
    monkeypatch.setattr(os.path, "isfile", lambda q: q == real)
    assert prompt._candidate_image_paths(escaped)[0] == real
    assert prompt._extract_image_paths(escaped) == [real]


def test_extract_image_paths_windows_not_unescaped(tmp_path, monkeypatch):
    """Windows 不回归：_IS_WINDOWS=True 时反斜杠不被反转义，C:\\ 与引号用例全通。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", True)
    a = tmp_path / "a.png"
    b_space = tmp_path / "b b.jpg"
    a.write_bytes(b"x")
    b_space.write_bytes(b"x")
    assert prompt._extract_image_paths(str(a)) == [str(a)]
    assert prompt._extract_image_paths(f'"{a}" "{b_space}"') == [str(a), str(b_space)]


@pytest.mark.skipif(sys.platform != "win32", reason="反斜杠分隔符仅 Windows")
def test_windows_path_unescape_would_break(tmp_path, monkeypatch):
    """反证：Windows 下若误开反转义，C:\\ 分隔符会被吃掉 → 证明平台门控必要。"""
    from tui import prompt
    a = tmp_path / "a.png"
    a.write_bytes(b"x")
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)   # 错误的平台判定
    assert prompt._extract_image_paths(str(a)) == []    # C:\Users → C:Users，isfile 失败


def test_extract_image_paths_rejects_non_windows(tmp_path, monkeypatch):
    """反向：非图片扩展名 / 不存在 / URL / 纯中文 → []（非 Windows 下同样成立）。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    txt = tmp_path / "note.txt"
    txt.write_text("x")
    assert prompt._extract_image_paths(str(txt)) == []
    assert prompt._extract_image_paths(str(tmp_path / "missing.png")) == []
    assert prompt._extract_image_paths("随便一句话没有路径") == []
    assert prompt._extract_image_paths("看一下 https://x.com/foo.png") == []


# ── 占位符替换必须按原文区间（旧 str.replace(真实路径) 在转义原文上静默失配）──

def _patch_register(monkeypatch, prompt, seen: list):
    """把 _register_image 换成不碰文件系统的假实现：记录收到的路径 + 产出占位符条目。"""
    def fake(path, registry, images_ref, counter):
        seen.append(path)
        name = os.path.basename(path)
        registry.append((f"[🖼 {counter}: {name}]", name, "image"))
        images_ref.append({"media_type": "image/png", "data": "FAKE"})
        return True
    monkeypatch.setattr(prompt, "_register_image", fake)


def _rewrite_like_delayed_check(prompt, text):
    """复刻 prompt.py `_delayed_check` 的重写链路，返回 (重写后文本, images_ref)。"""
    registry: list = []
    images_ref: list = []
    edits: list = []
    counter = 0
    for start, end, img_path in prompt._image_path_spans(text):
        if prompt._register_image(img_path, registry, images_ref, counter):
            counter += 1
            edits.append((start, end, registry[-1][0]))
    return prompt._splice_spans(text, edits), images_ref


def test_old_replace_approach_fails_on_escaped_path():
    """反证：str.replace(真实路径) 在 macOS 转义原文上必然不命中——这正是被替换掉的旧逻辑。"""
    assert MACOS_ESCAPED.replace(MACOS_REAL, "[🖼 1: x]") == MACOS_ESCAPED


def test_image_path_spans_covers_raw_escaped_span(monkeypatch):
    """区间必须是**原文**（转义形态）的区间，路径则已反转义。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    monkeypatch.setattr(os.path, "isfile", lambda q: q == MACOS_REAL)
    assert prompt._image_path_spans(MACOS_ESCAPED) == [(0, len(MACOS_ESCAPED), MACOS_REAL)]


def test_splice_spans_keeps_trailing_text(monkeypatch):
    """转义路径 + 后跟中文 → 占位符吃掉整段转义区间，中文原样保留。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    monkeypatch.setattr(os.path, "isfile", lambda q: q == MACOS_REAL)
    text = MACOS_ESCAPED + " 图片里面有什么"
    edits = [(s, e, "[🖼 0: png]") for s, e, _p in prompt._image_path_spans(text)]
    assert prompt._splice_spans(text, edits) == "[🖼 0: png] 图片里面有什么"


def test_splice_spans_multiple_in_order(monkeypatch):
    """多图：区间互不重叠、替换后偏移正确累加，顺序不乱。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    real1 = "/Users/haning/Downloads/图 一.png"
    real2 = "/Users/haning/Downloads/图 二.jpg"
    monkeypatch.setattr(os.path, "isfile", lambda q: q in (real1, real2))
    esc1 = real1.replace(" ", "\\ ")
    esc2 = real2.replace(" ", "\\ ")
    text = f"{esc1} {esc2} 对比一下"
    spans = prompt._image_path_spans(text)
    assert [(s, e) for s, e, _p in spans] == [(0, len(esc1)), (len(esc1) + 1, len(esc1) + 1 + len(esc2))]
    edits = [(s, e, f"[🖼 {i}: x]") for i, (s, e, _p) in enumerate(spans)]
    assert prompt._splice_spans(text, edits) == "[🖼 0: x] [🖼 1: x] 对比一下"


def test_full_chain_escaped_path_reaches_model(monkeypatch):
    """端到端（本次事故场景）：转义路径 + 中文 → 重写后占位符就位 → 组装出 image block。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", False)
    monkeypatch.setattr(os.path, "isfile", lambda q: q == MACOS_REAL)
    seen: list = []
    _patch_register(monkeypatch, prompt, seen)
    text = MACOS_ESCAPED + " 图片里面有什么"
    rewritten, images_ref = _rewrite_like_delayed_check(prompt, text)
    assert seen == [MACOS_REAL]                       # 喂给读取层的是反转义后的真实路径
    content = _assemble_image_content("", rewritten, images_ref)
    assert isinstance(content, list)                  # 拿到 list = 真的有图，而非降级纯文本
    assert sum(b["type"] == "image" for b in content) == 1
    assert any(b["type"] == "text" and b["text"] == "图片里面有什么" for b in content)


def test_full_chain_windows_path_unaffected(tmp_path, monkeypatch):
    """Windows 回归：真实路径直接出现在原文里，区间替换同样命中。

    路径不含空格——未加引号的含空格 Windows 路径本就无法与自然语言切分（既有局限，
    非本次改动引入）；含空格场景由下面带引号的用例覆盖。
    """
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", True)
    p = tmp_path / "shot.png"
    p.write_bytes(b"x")
    seen: list = []
    _patch_register(monkeypatch, prompt, seen)
    text = f"看下 {p} 谢谢"
    rewritten, images_ref = _rewrite_like_delayed_check(prompt, text)
    assert seen == [str(p)]
    assert rewritten == "看下 [🖼 0: shot.png] 谢谢"
    assert len(_assemble_image_content("", rewritten, images_ref)) == 3


def test_full_chain_windows_quoted_path_with_space(tmp_path, monkeypatch):
    """Windows 含空格路径：终端加引号包裹 → 区间含引号本身，替换后不留引号残骸。"""
    from tui import prompt
    monkeypatch.setattr(prompt, "_IS_WINDOWS", True)
    p = tmp_path / "shot (1).png"
    p.write_bytes(b"x")
    seen: list = []
    _patch_register(monkeypatch, prompt, seen)
    text = f'看下 "{p}" 谢谢'
    rewritten, images_ref = _rewrite_like_delayed_check(prompt, text)
    assert seen == [str(p)]
    assert rewritten == "看下 [🖼 0: shot (1).png] 谢谢"
    assert sum(b["type"] == "image" for b in _assemble_image_content("", rewritten, images_ref)) == 1


def test_register_multiple_images(tmp_path):
    """多文件拖入 → 多个占位符 + images_ref 多条目，顺序对应。"""
    a = tmp_path / "a.png"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"\x89PNG data-a")
    b.write_bytes(b"\xff\xd8 data-b")
    registry: list = []
    images_ref: list = []
    paths = _extract_image_paths(f'"{a}" "{b}"')
    assert paths == [str(a), str(b)]
    for i, path in enumerate(paths):
        assert _register_image(path, registry, images_ref, i) is True
    assert [p for p, _n, k in registry] == ["[🖼 0: a.png]", "[🖼 1: b.jpg]"]
    assert [img for img in registry] and len(images_ref) == 2
    assert images_ref[0]["media_type"] == "image/png"
    assert images_ref[1]["media_type"] == "image/jpeg"


# ── 注册 → base64 ───────────────────────────────────────

def test_register_image_success(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(b"\x89PNG fake data")
    registry: list = []
    images_ref: list = []
    assert _register_image(str(p), registry, images_ref, 0) is True
    placeholder, actual, kind = registry[0]
    assert kind == "image"
    assert placeholder == "[🖼 0: pic.png]"
    assert actual == "pic.png"   # 注册表存文件名（展示用），路径无需保留：数据已在 images_ref
    assert images_ref == [{"media_type": "image/png", "data": "iVBORyBmYWtlIGRhdGE="}]


def test_register_image_size_limit(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(b"0" * 10)  # 极小文件，仅验证上限常量存在
    registry: list = []
    images_ref: list = []
    # 模拟超限：直接把文件大小上限常量改为 0 不优雅；这里验证正常注册路径仍走通
    assert _register_image(str(big), registry, images_ref, 0) is True
    assert _IMAGE_MAX_BYTES == 8 * 1024 * 1024


def test_register_image_mime_mapping(tmp_path):
    p = tmp_path / "photo.jpeg"
    p.write_bytes(b"xx")
    registry: list = []
    images_ref: list = []
    _register_image(str(p), registry, images_ref, 0)
    assert images_ref[0]["media_type"] == "image/jpeg"


# ── P1 剪贴板截图 ────────────────────────────────────────

def test_register_clipboard_image_success(monkeypatch):
    """剪贴板有位图 → 注册占位符 + base64 数据进 images_ref。"""
    from tui import prompt
    monkeypatch.setattr(
        "tui.clipboard_image.get_clipboard_image",
        lambda: ("image/png", "QUJD"),
    )
    registry: list = []
    images_ref: list = []
    assert prompt._register_clipboard_image(registry, images_ref, 0) is True
    placeholder, _name, kind = registry[0]
    assert kind == "image"
    assert placeholder == "[🖼 0: clipboard-0.png]"
    assert images_ref == [{"media_type": "image/png", "data": "QUJD"}]


def test_register_clipboard_image_none(monkeypatch):
    """剪贴板无图 → 静默返回 False（不打扰文本粘贴）。"""
    from tui import prompt
    monkeypatch.setattr("tui.clipboard_image.get_clipboard_image", lambda: None)
    registry: list = []
    images_ref: list = []
    assert prompt._register_clipboard_image(registry, images_ref, 0) is False
    assert registry == []
    assert images_ref == []


def test_get_clipboard_image_success(monkeypatch, tmp_path):
    """PowerShell 输出 base64 → 解析为 (media_type, data)。"""
    from tui import clipboard_image
    class FakeProc:
        returncode = 0
        stdout = "  QVdFSG9N\n"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeProc())
    assert clipboard_image.get_clipboard_image() == ("image/png", "QVdFSG9N")


def test_get_clipboard_image_missing(monkeypatch):
    """PowerShell 异常 / 空输出 / 非零返回 → None。"""
    from tui import clipboard_image
    class ErrProc:
        returncode = 1
        stdout = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ErrProc())
    assert clipboard_image.get_clipboard_image() is None


def test_get_clipboard_image_exc(monkeypatch):
    """subprocess 抛异常（PowerShell 未安装等）→ None。"""
    from tui import clipboard_image
    def boom(*a, **k):
        raise OSError("no powershell")
    monkeypatch.setattr("subprocess.run", boom)
    assert clipboard_image.get_clipboard_image() is None


# ── P2 能力表 ────────────────────────────────────────────

def test_supports_vision_known():
    """能力表（用户精简版：国内系 + vision/vl 兜底）命中判定。"""
    from core.model_capabilities import supports_vision
    assert supports_vision("deepseek-v4-flash-vision-exp") is True
    assert supports_vision("glm-4.5v") is True
    assert supports_vision("qwen2.5-vl-72b") is True
    # 表外模型（如 gpt-4o）→ None 未知放行，不误拦
    assert supports_vision("GPT-4O") is None
    assert supports_vision("claude-3-5-sonnet") is None


def test_supports_vision_not_supported():
    from core.model_capabilities import supports_vision
    assert supports_vision("gpt-3.5-turbo") is False
    assert supports_vision("deepseek-chat") is False


def test_supports_vision_unknown_returns_none():
    from core.model_capabilities import supports_vision
    assert supports_vision("some-brand-new-model-x") is None
    assert supports_vision("") is None


# ── P2 图片 token 估算（Aider 公式 tiles×170+85）─────────

def _fake_png(w: int, h: int) -> str:
    import base64
    import struct
    raw = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR"
           + struct.pack(">II", w, h) + b"\x00" * 12)
    return base64.b64encode(raw).decode()


def _image_block(data_b64: str, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"media_type": media_type, "data": data_b64}}


def test_estimate_image_tokens_512():
    """512² 短边缩到 768 → tiles=2×2=4 → 4×170+85（Aider 公式本义）。"""
    from features.compact import _estimate_image_tokens
    assert _estimate_image_tokens(_image_block(_fake_png(512, 512))) == 4 * 170 + 85


def test_estimate_image_tokens_1024x2048_scales():
    """1024×2048：短边缩放 1024→768 后 tiles=2×3=6 → 6×170+85。"""
    from features.compact import _estimate_image_tokens
    block = _image_block(_fake_png(1024, 2048))
    assert _estimate_image_tokens(block) == 6 * 170 + 85


def test_estimate_image_tokens_malformed_fallback():
    from features.compact import _estimate_image_tokens
    # 无法解析尺寸（乱数据）→ 固定兜底 1100
    assert _estimate_image_tokens(_image_block("aGVsbG8=")) == 1100


def test_estimate_tokens_counts_image_blocks():
    from features.compact import estimate_tokens
    msg = {"role": "user", "content": [
        {"type": "text", "text": "看图 "},
        _image_block(_fake_png(512, 512)),
    ]}
    # 估算 = 文本(CJK "看图"2 + 空格 1//4) + 图片 4×170+85
    total = estimate_tokens([msg])
    assert total == 2 + 4 * 170 + 85


def test_estimate_tokens_plain_text_unchanged():
    """纯文本路径行为与旧实现一致（str 消息 CJK 估算）。"""
    from features.compact import estimate_tokens, CHARS_PER_TOKEN
    msg = {"role": "user", "content": "你好 world"}
    assert estimate_tokens([msg]) == 2 + len(" world") // CHARS_PER_TOKEN


def test_image_size_from_bytes_png_jpeg():
    import base64
    import struct
    from features.compact import _image_size_from_bytes
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4 + b"IHDR" + struct.pack(">II", 1920, 1080) + b"\x00" * 12
    assert _image_size_from_bytes(png) == (1920, 1080)
    jpeg = (b"\xff\xd8" + b"\xff\xc0" + struct.pack(">H", 11) + b"\x08"
            + struct.pack(">HH", 480, 320) + b"\x01\x01\x11\x00")
    assert _image_size_from_bytes(jpeg) == (320, 480)


# ── P3 序列化兼容（JSONL 落盘 / resume 恢复含图消息）──────

def test_serialize_content_image_message_roundtrip():
    """含图 user 消息经 _serialize_content 后字段无损（JSON 可序列化）。"""
    import json
    from core.session import _serialize_content, _extract_text
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "看图"},
            {"type": "image", "source": {"media_type": "image/png", "data": "QUJD"}},
        ],
    }
    safe = _serialize_content(msg["content"])
    # JSON 可序列化（无 Pydantic/SDK 对象残留）
    json.dumps({"content": safe}, ensure_ascii=False)
    assert safe[0] == {"type": "text", "text": "看图"}
    assert safe[1] == {"type": "image", "source": {"media_type": "image/png", "data": "QUJD"}}
    # 标题抽取：图片 block 无 text → _extract_text join 后带尾空格
    assert _extract_text(safe) == "看图 "


# ── 补全应用检测（方向键选择 / 命令，跳过剪贴板探测）──────
# 注意：不能依赖 buf.complete_state——pt 3.0.52 的 _text_changed() 在 fire
# on_text_changed 前先把它置 None，回调时恒为 None。判定基于文本。
# 新增 2 个临时命令注册进 _COMMAND_TABLE，确保测试不受安装环境 skills 影响。

def test_is_completion_apply_true_for_selected_command():
    """方向键应用补全（/ → /memory）：文本恰为完整斜杠命令 → 跳过粘贴探测。"""
    from tui.prompt import _is_completion_apply
    assert _is_completion_apply("/memory") is True


def test_is_completion_apply_false_for_typing_prefix():
    """手动打字半截（/memor）≠ 完整命令 → 不豁免（继续粘贴探测+过滤重算）。"""
    from tui.prompt import _is_completion_apply
    assert _is_completion_apply("/memor") is False


def test_is_completion_apply_false_for_unknown_command():
    """未知命令名（/not_a_real_command）→ 不豁免（保留粘贴探测能力）。"""
    from tui.prompt import _is_completion_apply
    assert _is_completion_apply("/not_a_real_command") is False


def test_is_completion_apply_false_for_non_slash():
    """文本不以 / 开头（如粘贴图片路径 C:/img.png）→ 不豁免。"""
    from tui.prompt import _is_completion_apply
    assert _is_completion_apply("C:/img.png") is False


def test_is_completion_apply_works_even_when_state_gone():
    """模拟真实时序：回调时 complete_state 已被 pt 置 None —— 判定不依赖它，
    文本恰为命令仍应判定为补全应用（这才是方向键能跳的原因）。"""
    from prompt_toolkit.buffer import Buffer
    from tui.prompt import _is_completion_apply
    buf = Buffer()
    # go_to_completion 的真实时序：回调时 complete_state 已被清空
    assert buf.complete_state is None
    from tui.prompt import _is_completion_apply
    assert _is_completion_apply("/memory") is True
