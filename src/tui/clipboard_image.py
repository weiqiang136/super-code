"""Windows 剪贴板位图读取（P1 多模态）：PowerShell 转 base64，零 Python 依赖。

PowerShell 链路：Clipboard.GetImage() → MemoryStream.Save(PNG) → base64。
非 Windows / 剪贴板无图 / 超时 / 失败一律返回 None（静默跳过，不打扰主流程）。
"""
from __future__ import annotations

import os
import subprocess

_PS_SCRIPT = r"""
Add-Type -AssemblyName System.Windows.Forms
$img = [System.Windows.Forms.Clipboard]::GetImage()
if ($null -ne $img) {
  $ms = New-Object System.IO.MemoryStream
  $img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
  [System.Convert]::ToBase64String($ms.ToArray())
}
"""


def get_clipboard_image() -> tuple[str, str] | None:
    """探测剪贴板位图 → (media_type, base64)。

    Returns:
        ("image/png", base64)。剪贴板无图 / PowerShell 不可用 / 超时 → None。
    """
    if os.name != "nt":
        return None
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _PS_SCRIPT],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    data = proc.stdout.strip()
    return ("image/png", data) if data else None
