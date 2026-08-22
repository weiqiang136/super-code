# AGENTS.md

本文件给 super-code（以及其它 AI 编码助手）在本仓库中工作时提供必要上下文。

## Commit 格式

`feat：V3.X.XX：描述` 或 `fix：V3.X.XX：描述`（项目历史风格）。中文冒号（`：`），
版本三段式（主版本始终 3），描述用中文。

## 高层架构

**入口 & 主循环**：`src/tui/app.py:main` 组装 `CommandContext`，进入 `bordered_prompt`
REPL 循环。

**核心数据流**：用户输入 → `engine.submit()` → `LLMClient.stream()`（OpenAI 兼容
API）。`submit()` yield 事件（`text`/`tool_call`/`tool_result`/`stale_reclaim`），
`tui/query.py:run_query()` 渲染；工具执行读并发/写串行，结果追加 `_messages`。

**消息格式**：内部消息用 block 结构（`{type: "tool_use"|"tool_result"|"text"}`），
`_to_openai_messages()` 转 OpenAI 格式；思考模型保留 `reasoning_content` 原样带回，
否则上下文虚缩。

**配置优先级**（`core/config.py:load_app_config`）：CLI > 项目 `./.super-code.json` >
便携（exe 同级 `super-code.json`）> 全局 `~/.config/super-code/` > 默认值。
Provider 仅 `openai`（`openai.OpenAI`，兼容 API 通用）。

## 关键子系统（跨文件理解）

### Snippet 凭证系统
`core/file_state.py`：snippet_id 是 Edit 凭证（限行范围 + file_version 失效检测）。
/resume 重建注册表；/compact 全失效强制重读。stale 回收：
`compact.py:reclaim_stale_read_results`（engine 每轮发送前调用，V3.2.6+）。

### 协调者模式
`--coordinator`：主 agent 得 Agent/SendMessage/TaskStop，`features/worker_manager.py`
管 worker pool（独立 Engine、只读 + auto_approve）；通知 `drain_notifications()`
批量取、合并单轮 `run_query()`。提示词（`features/coordinator.py`）规定：信任 worker
报告、不重复验证、不 git 写操作。

### 记忆系统
按 git root 映射 `~/.config/super-code/projects/<sanitized-path>/memory/`，git 内
只读项目级、非 git 读全局（互斥无 fallback）。核心 `features/memory.py`；辅助
`extract_memories.py`（daemon 抽取）、`find_relevant_memories.py`（轻量模型检索）、
`memory_scan.py`/`memory_age.py`/`memory_types.py`。

### 上下文压缩（Compact）
`features/compact.py`：CJK 估算（中文 1 字 ≈ 1 token），触发 = 窗口 × 0.8；
`COMPACT_BOUNDARY_MARKER` 只压增量；熔断器（3 次失败禁本 session）；PTL 兜底 ≤3；
空摘要拒落盘（防吞历史事故）。

### 会话持久化
`core/session.py`：JSONL 每行一条；`mark_checkpoint()`/`rollback_to_checkpoint()`
truncate，Ctrl+C 不留孤立 tool_use；meta.json 临时文件 + `os.replace()` 原子写。

### Skills 系统
`features/skills.py`：扫两处 `skills/` 的 `<name>/SKILL.md`；context 分 inline
（注入当前对话）/ fork（独立 Engine）；压缩时重注入已调用 body（保头截尾）。

### git-ai 集成
`features/git_ai.py`：Edit/Write 前后 `git-ai checkpoint agent-v1`（前 human 后
ai_agent），先 `git-ai bg start` 保 daemon。未装静默跳过；`SUPER_CODE_DEBUG_GIT_AI=1`。

### MCP 工具加载
`mcp/loader.py`：`.mcp.json` = 项目 → exe 同级 `mcp.json` → 全局
`~/.config/super-code/mcp.json`；JSON-RPC 拿 `tools/list`；Windows 用 `shutil.which()`
解析 `.cmd`；退出 `shutdown_mcp()` 统一关。

### Plan 模式
`features/plan.py`：Shift+Tab（仅输入等待时生效）。进入注入只读提示词；退出经
`_post_tool_hooks` 延迟清历史（防工具执行中途动 engine._messages）。

### 沙箱
`core/sandbox/` 包（非 sandbox.py）：`--sandbox` 或 `sandbox.enabled=true` 开启；
`blacklist.py` 命令黑名单（BashTool 前检查）+ `path_protection.py` 路径保护 +
`network.py` 域名白/黑名单。非 OS 级隔离。

## 常用命令

- `pytest tests/ -v` — 全量（24 个 test 文件）；`pytest tests/test_xxx.py -v` 单个
- `pyinstaller super-code.spec` — 打包 exe（`*.spec` 在 .gitignore，不入库）
- 无 lint/formatter 配置（无 ruff.toml/.flake8/eslint）

## Windows 特有问题

- lone surrogate（`\ud800-\udfff`）在 `engine.submit()` 入口替换 U+FFFD（防 API 报错）
- StdoutProxy（`prompt_toolkit.patch_stdout`）：后台线程 print 不糊进输入框
