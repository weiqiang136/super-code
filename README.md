# super-code

> 终端里的 AI 编程助手 —— 支持 DeepSeek、OpenAI 及所有 OpenAI 兼容接口

## 简介

super-code 是一个运行在终端的 AI 编程助手。通过 OpenAI 兼容接口接入大模型，让你在命令行里直接完成代码阅读、编写、重构、调试等任务，无需离开终端环境。
<img width="2537" height="1502" alt="动画03" src="https://github.com/user-attachments/assets/2ab7187c-33c5-4b4c-8011-c14f4bfb19d3" />

## 特性

- **多模型支持**：兼容 DeepSeek、OpenAI、GLM、Qwen 等所有 OpenAI 兼容接口
- **记忆系统**：按 git 仓库隔离，自动记录用户偏好与项目事实，跨会话持久化
- **协调者模式**：主 agent 调度多个并发 worker，适合大型任务并行拆分
- **上下文压缩**：CJK-aware token 估算，接近上下文上限时自动压缩历史消息
- **Skills 系统**：从本地 `SKILL.md` 文件加载自定义技能，支持 inline 和 fork 两种模式
- **Plan 模式**：`Shift+Tab` 切换只读规划模式，不执行任何写操作
- **MCP 工具**：通过 `.mcp.json` 接入 Model Context Protocol 工具服务
- **会话持久化**：JSONL 格式记录对话历史，支持随时恢复
- **沙箱保护**：`--sandbox` 模式对危险命令进行黑名单过滤

## 安装（Windows）

1. 从 [Releases](https://github.com/weiqiang136/super-code/releases) 下载 `super-code.exe`，放到固定目录，例如：
   ```
   C:\tools\super-code\super-code.exe
   ```

2. 将该目录添加到系统环境变量 `PATH`：
   - 打开「系统属性」→「高级」→「环境变量」
   - 在「系统变量」中找到 `Path`，点击「编辑」→「新建」
   - 填入 exe 所在目录，例如 `C:\tools\super-code`
   - 确定保存，重新打开终端生效

3. 验证安装：
   ```bash
   super-code --version
   ```

配置完成后，可在任意目录直接执行 `super-code` 启动。

## 配置

在 exe **同级目录**创建 `super-code.json`。

配置文件加载优先级（低 → 高）：

| 位置 | 路径 |
|---|---|
| 全局 | `~/.config/super-code/super-code.json` |
| 便携 | exe 同级目录 `super-code.json` |
| 项目 | 当前目录 `.super-code.json` |

同名字段后加载者覆盖前者；三者可叠加使用（如全局放 api_key，项目级覆盖 model）。

### 快速super-code.json示例

**DeepSeek：**

```json
{
  // LLM 服务商标识，目前仅支持 "openai"
  "provider": "openai",

  // 模型单次最大输出 token 数，131072 即 128K
  "max_tokens": 131072,

  // HTTP 读取超时（秒）。思考模型首 token 延迟极长，DeepSeek/GLM/Qwen 建议 ≥ 300
  "timeout": 900.0,

  // 使用的模型名（主要对话模型）
  "model": "deepseek-v4-pro",

  "api_key": "",

  "base_url": "https://api.deepseek.com",

  // 记忆相关性检索用的轻量模型。节省主要模型 token，仅用于判断哪些记忆与当前问题相关
  "extract_model": "deepseek-v4-flash",

  // 协调者模式：开启后可派生子 worker 并行工作
  "coordinator": true,

  // ========== 命令沙箱配置 ==========
  "sandbox": {
    // 总开关：true 启用沙箱（内置 30+ 条危险命令正则 + 可追加自定义规则）
    "enabled": true,

    // 精确命令名黑名单（如 "mkfs"），匹配命令的第一个单词
    "blocked_commands": [],

    // 额外正则规则列表，每项 [正则, 描述]，追加到内置规则后。示例：[["sudo reboot", "reboot with sudo"]]
    "extra_patterns": [],

    // 白名单豁免：以这些字符串开头的命令整体跳过沙箱检查
    "excluded_commands": [],

    // 沙箱激活时是否自动批准 Bash 调用（危险命令已被黑名单拦截，安全命令无需用户逐个确认）
    "auto_approve_if_sandboxed": false,

    // 网络外发域名白名单：非空时 curl/wget 只能访问列表中的域名
    "allowed_domains": [],

    // 网络外发域名黑名单：命中直接拒绝，优先级高于白名单
    "denied_domains": []
  },

  // ========== 模型级微调参数（按模型名子串匹配） ==========
  // 用于控制各模型的推理/思考行为，匹配到的 extra_body 会注入到 API 请求体
  "model_profiles": {
    "deepseek-v4-pro": {
      "extra_body": {
        // DeepSeek 推理强度设为最大（深度思考模式）
        "reasoning_effort": "max",
        "thinking": {
          "type": "enabled"  // 显式开启思维链输出
        }
      }
    }
  }
}
```


### 完整字段说明

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `provider` | string | `"openai"` | 接口类型，目前统一填 `"openai"` |
| `api_key` | string | — | API 密钥 |
| `base_url` | string | — | 自定义接口地址；使用 OpenAI 官方可省略 |
| `model` | string | `"gpt-5.1-codex"` | 主对话模型名 |
| `max_tokens` | int | `131072` | 模型单次最大输出 token 数 |
| `timeout` | float | `300.0` | HTTP 读超时（秒）；GLM / Qwen 等思考模型建议 ≥ 300 |
| `coordinator` | bool | `false` | 启用协调者模式（等价于 `--coordinator` 参数） |
| `extract_model` | string | `""` | 记忆相关性筛选用的轻量模型（如 `gpt-4o-mini`）；空字符串沿用主模型 |
| `auto_dream` | bool | `true` | 是否在后台自动整合记忆日志 |
| `dream_interval_hours` | float | `24.0` | 两次自动整合之间的最小间隔（小时） |
| `dream_min_sessions` | int | `5` | 触发自动整合所需的最少新会话数 |
| `model_profiles` | object | `{}` | 按模型名子串匹配的额外请求参数（见下方示例） |
| `sandbox` | object | `null` | 沙箱保护配置（见下方示例） |

### model_profiles 示例

为 GLM 系列关闭思考模式：

```json
{
  "model_profiles": {
    "glm": {
      "extra_body": {
        "thinking": { "type": "disabled" }
      }
    }
  }
}
```

key 为模型名子串（大小写不敏感），最长 key 优先匹配。

### sandbox 沙箱配置

```json
{
  "sandbox": {
    "enabled": true,
    "blocked_commands": ["mkfs", "fdisk"],
    "excluded_commands": ["docker", "bazel"],
    "extra_patterns": [["rm.*-rf.*/", "dangerous recursive delete"]],
    "auto_approve_if_sandboxed": true,
    "allowed_domains": ["api.openai.com", "api.deepseek.com"],
    "denied_domains": ["malicious.example.com"]
  }
}
```

| 字段 | 说明 |
|---|---|
| `enabled` | 是否启用沙箱（默认 false） |
| `blocked_commands` | 精确命令黑名单（匹配命令名） |
| `excluded_commands` | 跳过沙箱检查的命令前缀 |
| `extra_patterns` | 额外正则黑名单，格式 `[pattern, reason]` |
| `auto_approve_if_sandboxed` | 沙箱已保护时自动批准 Bash 调用 |
| `allowed_domains` | 网络外发域名白名单（非空时仅允许列表内域名） |
| `denied_domains` | 网络外发域名黑名单（优先级高于白名单） |

## 使用

```bash
# 进入你的项目目录
cd /path/to/your/project

# 启动
super-code.exe

# 开启协调者模式（多 agent 并发）
super-code.exe --coordinator
```

## 斜杠命令

| 命令 | 说明 |
|---|---|
| `/help` | 查看所有可用命令 |
| `/clear` | 清空当前对话，开始新会话 |
| `/history` | 列出当前目录下的历史会话 |
| `/resume [编号\|会话ID]` | 恢复历史会话；无参数时进入交互选择器 |
| `/compact [指令]` | 手动压缩上下文，可附加压缩指令 |
| `/skills` | 列出所有已加载的 skills |
| `/cost` | 查看本次会话 token 用量与费用 |
| `/remember <内容>` | 立即保存一条记忆到当天日志 |
| `/memory` | 查看 MEMORY.md 索引 |
| `/memory list` | 列出所有 topic 记忆文件 |
| `/memory <编号\|关键词>` | 用编辑器打开指定记忆文件 |
| `/dream` | 手动触发记忆整合（将日志蒸馏为持久记忆） |
| `/rename [新标题]` | 重命名当前会话 |
| `/init [额外提示]` | 扫描项目，自动生成或更新 `AGENTS.md` |

`Shift+Tab` 切换 **Plan 模式**（只读规划，不执行任何写操作）

## 记忆系统

super-code 会在每轮对话结束后，后台自动提取用户偏好、项目事实、反馈信息，保存到：

```
~/.config/super-code/projects/<项目路径>/memory/
```

下次会话启动时，相关记忆自动注入系统提示词，无需重复说明背景。

使用 `/dream` 可将碎片日志整合为结构化的持久记忆文件。

## Skills

在以下目录放置 `SKILL.md` 文件即可定义自定义技能：

- 全局：`~/.config/super-code/skills/`
- 项目：`.super-code/skills/`

```markdown
---
name: my-skill
description: 做某件事的技能描述
---

技能的详细说明和指令...
```

通过 `/my-skill` 调用，`/skills` 查看所有已加载的技能。

## 协调者模式

```bash
super-code.exe --coordinator
```

开启后，主 agent 可以创建并调度多个并发 worker agent，适合需要并行处理多个子任务的场景（如同时分析多个文件、并发执行多条搜索）。

## 开发者

```bash
git clone https://github.com/weiqiang136/super-code.git
cd super-code
pip install -e .
super-code
```

## License

[MIT License](LICENSE)
