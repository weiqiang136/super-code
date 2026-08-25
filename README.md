# super-code

Minimal AI coding assistant —— 命令行交互式 AI 编码助手。

## 安装

```bash
pip install super-code-cli
```

安装后**首次运行**自动初始化配置到 `~/.config/super-code/super-code.json`，
编辑该文件填入 api_key 即可使用。

## 配置

配置按优先级加载：CLI 参数 > 项目 `./.super-code.json` > 便携目录
（exe 同级 `super-code.json`）> 全局 `~/.config/super-code/super-code.json` > 默认值。

首次运行从内置模板生成全局配置，已有配置永不覆盖。

## 使用

```bash
super-code                 # 交互式 REPL
super-code -p "..."        # 一次性提问
super-code --resume <id>   # 恢复会话
super-code --coordinator   # 协调者模式（多 worker 并行）
```

## 功能

- OpenAI 兼容 API；多模态图片输入（拖拽/粘贴截图）
- 工具调用：文件读写编辑、Bash、Web 搜索/抓取、多 worker 协调者
- 会话持久化（恢复/增量压缩/剪枝）、记忆系统、Skills、Plan 模式、沙箱
