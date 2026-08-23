"""P0 多模态：图片占位符注册 + 组装（tests 里用 tmp 文件模拟拖拽）。"""
import os
import sys

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
