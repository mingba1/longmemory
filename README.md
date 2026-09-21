# longmemory

教学向的**长期记忆 AI 助手**。多轮对话并不等于有记忆：把历史原样拼进 prompt，换个会话就忘了，也撑不住长期偏好。

真正的记忆是系统在长期聊天里**记住你的喜好、懂你的需求**，不用每次重新自我介绍。好体验往往不靠模型更聪明，而靠系统能不能一直「记得你」。

本仓库把记忆拆成三层，不套 LangChain，也不上向量数据库；嵌入用本地 BGE，余弦用 NumPy，方便看清每一步。

## 记忆分层

| 层 | 存什么 | 怎么用 |
|---|---|---|
| **画像 Profile** | 稳定偏好 / 身份 / 长期需求（`profile.json`） | 每轮固定注入 system |
| **情景 Episodic** | 可检索短条目（`memories.json` + `vectors.npy`） | 当前问题向量检索 Top-K |
| **工作上下文** | 本会话最近 N 轮（`sessions/*.jsonl`） | 字符预算内保留，超出裁剪 |

刻意不做：无限追加整段聊天记录。

单轮流程：检索情景 → 组装（画像 + 记忆 + 最近轮次）→ 生成 → 抽取事实写回画像 / 追加情景。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 低于 3.14 且能装上 `torch` 时，会用本地 `BAAI/bge-small-zh-v1.5`（首次约一百兆）。当前环境若没有 torch（例如 Python 3.14），自动改用字符 n-gram 哈希向量，检索链路仍可演示，效果弱于 BGE。

可选大模型（不配也能跑，规则兜底）：

```bash
cp .env.example .env
# 填入 OPENAI_API_KEY；兼容网关可改 OPENAI_BASE_URL / OPENAI_MODEL
```

## 命令

```bash
# 交互聊天（换 --session 模拟「隔天再聊」，画像与情景仍在）
python -m longmemory chat --user u1 --session day1
python -m longmemory chat --user u1 --session day2

# 查看画像与情景记忆
python -m longmemory show --user u1

# 清空该用户记忆
python -m longmemory reset --user u1

# 固定剧本：建立偏好 → 新会话回忆，并打印有/无记忆上下文差异
python -m longmemory demo
```

`chat` 加 `-v` 可看到本轮记忆命中数、写入的情景条目。`--no-memory` 关闭长期记忆注入与写入，便于对照。

## demo 在证明什么

`demo` 会：

1. 在 `day1` 告诉助手姓名、Python、讨厌长回复等；
2. 换到 `day2` 再问「还记得我吗」——此时本会话几乎没有历史，但画像与情景检索仍在；
3. 再跑一遍**无记忆**组装，打印两侧注入的上下文。

你会看到：无记忆模式下 system 里没有画像/情景；有记忆模式才能在换 session 后继续「认得你」。这就是「多轮拼接 ≠ 长期记忆」。

剧本原文见 [data/demo/script.txt](data/demo/script.txt)。

## 数据落在哪

```
data/users/<user_id>/
  profile.json
  memories.json
  vectors.npy
  sessions/<session_id>.jsonl
```

`demo` 默认写到 `data/demo_users/`，不污染日常 `chat` 数据。

## 模块对照

| 文件 | 职责 |
|---|---|
| `store.py` | 用户目录读写 |
| `embed.py` / `retrieve.py` | BGE（可装 torch 时）或哈希向量 + NumPy 余弦 Top-K |
| `context.py` | 字符预算组装与裁剪 |
| `generate.py` | LLM 或无 Key 兜底 |
| `extract.py` | 画像 / 情景抽取 |
| `pipeline.py` | 单轮闭环 |
| `demo.py` / `cli.py` | 对比演示与命令行 |

## 和「把历史拼进去」差在哪

- **会话窗口**只服务当前对话流畅度，有预算、会裁剪。
- **画像**跨会话稳定注入，解决「我是谁、我喜欢怎样被对待」。
- **情景检索**只拉与当前问题相关的条目，避免把无关旧聊天全部塞进上下文。

这三块合在一起，才是可以长期陪伴的记忆系统，而不是越聊越胖的日志回放。
