"""super-code 的 Harbor agent adapter（Terminal-Bench 评测用）。

Installed-agent 模式：容器内 pip 安装 super-code-assistant，headless 模式
（`-p --auto-approve`）执行任务，从 stderr 的 [SUPERCODE_USAGE] 行解析
token 用量填入 AgentContext。

用法：
    harbor run -d terminal-bench/terminal-bench -e docker \
        --agent eval.harbor_agent.supercode_agent:SuperCode \
        -m deepseek/deepseek-v4-flash

依赖 super-code 的配套改动（V3.3.13+）：headless 结束时往 stderr 输出
`[SUPERCODE_USAGE] {"input_tokens": N, "output_tokens": M, ...}`。
"""
from __future__ import annotations

import json
import re
import shlex
from typing import override

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    with_prompt_template,
)
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_USAGE_RE = re.compile(r"\[SUPERCODE_USAGE\]\s*(\{.*\})")

# 国内服务器直连 PyPI 慢，用清华镜像加速容器内安装
_PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"


class SuperCode(BaseInstalledAgent):
    """Run super-code headless inside the task container."""

    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)

    @staticmethod
    @override
    def name() -> str:
        return "supercode"

    @override
    def get_version_command(self) -> str | None:
        return "super-code --help"

    @override
    def parse_version(self, stdout: str) -> str:
        # super-code 无 --version，从 pip 元数据拿版本号；拿不到就给个占位
        return "super-code-assistant (pip)"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # 任务容器是精简 python3（无 pip），且 apt 直连官方源在国内会超时。
        # 镜像可能是 Ubuntu（sources.list）或 Debian 12（deb822 的 *.sources），
        # 兼容两种布局换阿里云源，再装 python3-pip、走清华 PyPI 源装包。
        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                "if [ -f /etc/apt/sources.list ]; then "
                "sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g; "
                "s|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list; fi; "
                "find /etc/apt/sources.list.d -type f -name '*.sources' -exec "
                "sed -i 's|deb.debian.org|mirrors.aliyun.com|g; "
                "s|security.debian.org|mirrors.aliyun.com|g; "
                "s|archive.ubuntu.com|mirrors.aliyun.com|g' {} +; "
                "apt-get update -qq && apt-get install -y -qq python3-pip && "
                "python3 -m pip install --break-system-packages "
                f"-i {_PIP_INDEX} super-code-assistant && "
                "super-code --help > /dev/null"
            ),
            timeout_sec=900,
        )

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        access = self.model_connection
        if not access.api_key:
            raise RuntimeError(
                "No API key resolved — set DEEPSEEK_API_KEY in the harbor host env"
            )

        model_name = (
            self.model_name.split("/", 1)[1]
            if self.model_name and "/" in self.model_name
            else (self.model_name or "")
        )
        base_url = getattr(access, "base_url", None) or "https://api.deepseek.com"
        config = {
            "provider": "openai",
            "model": model_name,
            "api_key": access.api_key,
            "base_url": base_url,
            "max_tokens": 16384,
        }
        await self._upload_config_text(
            environment,
            content=json.dumps(config),
            remote_path="/tmp/super-code-eval.json",
            filename="super-code-eval.json",
        )

        result = await self.exec_as_agent(
            environment,
            command=(
                f"super-code {shlex.quote(instruction)} -p --auto-approve "
                f"--config /tmp/super-code-eval.json "
                f"2>&1 | tee /logs/agent/supercode.txt"
            ),
        )

        output = result.stdout or ""
        match = _USAGE_RE.search(output)
        if match:
            usage = json.loads(match.group(1))
            context.n_input_tokens = usage.get("input_tokens")
            context.n_output_tokens = usage.get("output_tokens")
            context.n_cache_tokens = usage.get("cache_read_input_tokens")
        else:
            # 没拿到 usage 行不视为失败（任务结果仍有效），但要在日志里可见
            print("[supercode-adapter] WARNING: [SUPERCODE_USAGE] not found in output")

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        pass
