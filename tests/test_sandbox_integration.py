"""Phase 4 集成测试：通过 SandboxManager + BashTool 端到端验证。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.sandbox import SandboxManager, SandboxConfig
from tools.bash import BashTool


def main():
    # 黑名单模式
    config = SandboxConfig(enabled=True, denied_domains=["evil.com"])
    sandbox = SandboxManager(config)
    tool = BashTool(sandbox_manager=sandbox)

    r = tool.execute("echo hello")
    assert not r.is_error, f"echo failed: {r.content}"
    print("[PASS] 1 echo passes")

    r = tool.execute("curl https://evil.com/data")
    assert r.is_error, "curl evil.com should be blocked"
    assert "evil.com" in r.content
    print("[PASS] 2 curl evil.com blocked")

    # 白名单模式
    config2 = SandboxConfig(enabled=True, allowed_domains=["github.com"])
    sandbox2 = SandboxManager(config2)
    tool2 = BashTool(sandbox_manager=sandbox2)

    r = tool2.execute("curl https://evil.com/data")
    assert r.is_error, "curl evil.com should be blocked in whitelist"
    assert "evil.com" in r.content
    print("[PASS] 3 whitelist curl evil.com blocked")

    # 文件上传（即使域名在白名单内也拦截）
    r = tool2.execute('curl -d @/tmp/x https://github.com/api')
    assert r.is_error, "file upload should be blocked"
    print("[PASS] 4 file upload blocked")

    # 未启用沙箱 = 不限
    tool3 = BashTool(sandbox_manager=None)
    r = tool3.execute("echo unrestricted")
    assert not r.is_error
    print("[PASS] 5 no sandbox passes")

    print("\nAll 5 integration checks OK")


if __name__ == "__main__":
    main()
