# AGENTS.md

本文件给 super-code（以及其它 AI 编码助手）在本仓库中工作时提供必要上下文。

## Commit 格式

`feat：V3.X.XX：描述` 或 `fix：V3.X.XX：描述`。注意使用中文冒号（`：`不是 `:`），
版本号三段式（主版本始终 3），描述用中文。

## 高层架构

**入口 & 主循环**：`src/tui/app.py:main`。启动时组装核心对象（Engine、SessionStore、
PermissionChecker、PlanManager、WorkerManager 等），打包为 `CommandContext`，
进入 `bordered_prompt` REPL 循环。

**核心数据流**：
1. 用户输入 → `engine.submit(user_input)` → `LLMClient.stream()` 调用 OpenAI 兼容 API
2. `submit()` yield 事件元组（`("text", ...)`, `("tool_call", ...)`, `("tool_result", ...)`）
3. `tui/query.py:run_query()` 消费事件流 → Rich 渲染输出
4. 工具执行由 Engine 内部完成（读操作并发线程池，写操作串行），结果追加到 `_messages`

**消息格式**：内部消息用 block 结构（`{type: "tool_use"|"tool_result"|"text", ...}`），
通过 `_to_openai_messages()` 转换为 OpenAI 的 tool_calls/tool 格式。DeepSeek 等思考模型
需在 assistant 消息中保留 `reasoning_content` 字段原样带回下一轮。

**配置优先级**（`core/config.py:load_app_config`）：CLI 参数 > 配置文件 > 默认值。
配置文件：全局 `~/.config/super-code/super-code.json`，项目级 `./.super-code.json`（JSON 格式）。
Provider 仅支持 `openai`（但 `LLMClient` 用的是 `openai.OpenAI`，所有 OpenAI 兼容 API 都可用）。

## 关键子系统（需要跨文件理解的部分）

### 协调者模式
通过 `--coordinator` 开启。主 agent 获得 Agent/SendMessage/TaskStop 三个工具，
通过 `WorkerManager` 管理 worker pool。每个 worker 有独立 Engine 实例（只读工具 + auto_approve）。
worker 完成通知通过 `worker_manager.drain_notifications()` 批量获取，合并为单轮 `run_query()`。
协调者提示词（`features/coordinator.py`）硬性规定：信任 worker 报告、不重复验证、不 git 写操作。

### 记忆系统
项目级隔离：根据 git root 映射到 `~/.config/super-code/projects/<sanitized-path>/memory/`。
非 git 仓库回退到全局目录。核心文件在 `features/memory.py`，辅助文件：
- `extract_memories.py`：后台 daemon 线程抽取偏好/事实
- `find_relevant_memories.py`：用轻量模型（`extract_model` 配置）做相关性检索
- `memory_scan.py` / `memory_age.py` / `memory_types.py`：扫描、陈旧度警告、类型定义

### 上下文压缩（Compact）
`features/compact.py`：CJK-aware token 估算（中文 1 字 ≈ 1 token），触发阈值 = context window × 0.8。
使用 `COMPACT_BOUNDARY_MARKER`（`<!-- COMPACT_BOUNDARY -->`）标记压缩点，
后续压缩只处理增量对话，避免反复套娃。自动压缩带熔断器（连续 3 次失败后本 session 禁用）。
PTL 兜底重试最多 3 次。

### 会话持久化
`core/session.py`：JSONL 格式，每行一条消息。Turn 级 checkpoint/rollback 机制：
`mark_checkpoint()` 记录字节偏移 → `rollback_to_checkpoint()` 用 `truncate()` 截断，
保证 Ctrl+C 取消不留下孤立 tool_use。meta.json 用临时文件 + `os.replace()` 原子写入。

### Skills 系统
`features/skills.py`：扫描 `~/.config/super-code/skills/` 和 `./.super-code/skills/` 下的
`SKILL.md`（YAML frontmatter + Markdown body）。支持 inline（注入当前对话）和 fork（独立 Engine）
两种模式。bundled skills 通过 `features/skills_bundled.py` 注册。压缩时会重注入已调用的 skill body。

### git-ai 集成
`features/git_ai.py`：在 Edit/Write 工具执行前后调用 `git-ai checkpoint agent-v1`，
用于 `git-ai status` 统计 AI 代码占比。git-ai 未安装时静默跳过。调试：`SUPER_CODE_DEBUG_GIT_AI=1`。

### MCP 工具加载
`mcp/loader.py`：读取 `.mcp.json`（项目 > 全局），启动 MCP server 子进程，
通过 `mcp/client.py` 的 JSON-RPC 协议获取 `tools/list`。Windows 下 `shutil.which()` 解析
`npx.cmd` 路径（V3.0.11 修复）。程序退出时 `shutdown_mcp()` 统一关闭子进程。

### Plan 模式
`features/plan.py`：Shift+Tab 切换。进入时注入 plan 专用系统提示词，只允许只读工具。
退出时 `PlanModeManager.exit()` 通过 `_post_tool_hooks` 延迟清理历史（避免在工具执行中途
操作 engine._messages 导致时序问题）。

### 沙箱
`core/sandbox.py`：`--sandbox` 开启，黑名单过滤危险命令（非 OS 级隔离）。
BashTool 执行前检查命令是否在黑名单中。

## 常用命令

- `pytest tests/ -v` — 运行所有测试（13 个测试文件）
- `pip install -e .` — 开发模式安装（入口 `super-code`）
- `pyinstaller super-code.spec` — 打包为独立 exe（输出在 `dist/`）
- 无 lint/formatter 配置（没有 ruff.toml、.flake8、eslint 等）

## REPL 斜杠命令

`/help` `/clear` `/history` `/resume` `/compact` `/skills` `/cost` `/remember` `/memory` `/dream` `/init`

`/init` 是本文件自身的生成命令：它组装 prompt 让模型扫描项目并写/更新 AGENTS.md。

## Windows 特有问题

- `shutil.which()` 解析 `.cmd/.bat` 后缀（V3.0.11）
- `\ud800-\udfff` lone surrogate 清洗：engine.submit() 入口用 `_LONE_SURROGATE_RE` 替换为 U+FFFD
- StdoutProxy（`prompt_toolkit.patch_stdout`）：防止后台线程 print 糊进输入框
