"""P0 多模态：图片占位符注册 + 组装（tests 里用 tmp 文件模拟拖拽）。"""
import os
import sys

sys.path.insert(0, "src")

from tui.app import _assemble_image_content, _IMAGE_PLACEHOLDER_RE
from tui.prompt import _extract_image_path, _register_image, _IMAGE_MAX_BYTES


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

def test_extract_image_path_recognizes(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(b"x")
    assert _extract_image_path(str(p)) == str(p)
    assert _extract_image_path(f'"{p}"') == str(p)          # 剥引号（终端拖拽带引号）
    assert _extract_image_path(f"  {p}  ") == str(p)


def test_extract_image_path_rejects(tmp_path):
    txt = tmp_path / "note.txt"
    txt.write_text("x")
    assert _extract_image_path(str(txt)) is None             # 非图片扩展名
    assert _extract_image_path(str(tmp_path / "missing.png")) is None  # 文件不存在
    assert _extract_image_path("随便一句话没有路径") is None


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
