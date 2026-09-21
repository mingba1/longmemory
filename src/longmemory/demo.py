"""脚本化演示：跨会话建立记忆，再对比有/无记忆注入的上下文。"""

from __future__ import annotations

from pathlib import Path

from longmemory.config import ROOT
from longmemory.context import assemble
from longmemory.pipeline import MemoryAssistant
from longmemory.retrieve import retrieve_memories

DEMO_USER = "demo"
SESSION_BUILD = "day1"
SESSION_RECALL = "day2"

SCRIPT: list[tuple[str, str]] = [
    (SESSION_BUILD, "你好，我叫小明，我写 Python，讨厌长回复，请尽量简洁。"),
    (SESSION_BUILD, "我在准备一个带长期记忆的助手项目，技术栈是 Python CLI。"),
    (SESSION_RECALL, "还记得我是谁、用什么语言、回复偏好吗？顺便帮我想一句项目介绍。"),
]


def run_demo(
    data_dir: Path | None = None,
    *,
    verbose: bool = True,
) -> dict:
    base = data_dir or (ROOT / "data" / "demo_users")
    base.mkdir(parents=True, exist_ok=True)

    assistant = MemoryAssistant(DEMO_USER, data_dir=base, use_memory=True)
    assistant.reset()

    if verbose:
        backend = assistant.embedder.backend
        print(f"嵌入后端: {backend}（bge=本地句向量，hash=无 torch 时的字符哈希兜底）")
        print("=== 阶段 1：建立记忆（session=day1）===")

    replies: list[str] = []
    for session_id, message in SCRIPT[:-1]:
        result = assistant.chat(session_id, message)
        replies.append(result.reply)
        if verbose:
            print(f"\n[{session_id}] 用户: {message}")
            print(f"[{session_id}] 助手({result.generated.mode}): {result.reply}")
            if result.new_memory:
                print(f"  + 情景记忆: {result.new_memory.text}")
            facts = result.profile.as_prompt_lines()
            if facts:
                print("  画像:", "; ".join(facts))

    recall_msg = SCRIPT[-1][1]
    if verbose:
        print("\n=== 阶段 2：新会话提问（session=day2，画像与情景仍在）===")

    # 有记忆一轮
    with_mem = assistant.chat(SESSION_RECALL, recall_msg)
    replies.append(with_mem.reply)

    if verbose:
        print(f"\n[day2] 用户: {recall_msg}")
        print(f"[day2] 助手(有记忆/{with_mem.generated.mode}): {with_mem.reply}")
        print("\n--- 有记忆时注入的上下文摘要 ---")
        print("画像:\n", with_mem.context.profile_text)
        print("相关记忆:\n", with_mem.context.memory_text)
        print(
            f"最近消息条数={with_mem.context.recent_turns_used}, "
            f"记忆条数={with_mem.context.memories_used}, "
            f"字符约={with_mem.context.char_count}"
        )

    # 无记忆对照：同一用户数据上，关闭记忆组装（不写入）
    if verbose:
        print("\n=== 对照：无记忆模式（同一问题，不注入画像/情景）===")

    bare = MemoryAssistant(
        DEMO_USER,
        data_dir=base,
        use_memory=False,
        embedder=assistant.embedder,
    )
    # 用独立 session，避免污染 day2 轮次对比展示；对照只展示组装差异
    profile, memories = assistant.show()
    vectors = assistant.store.load_vectors()
    turns = []  # 新会话、无历史
    scored = []
    if memories:
        q = assistant.embedder.embed_query(recall_msg)
        scored = retrieve_memories(q, memories, vectors)

    ctx_with = assemble(profile, scored, turns, recall_msg, use_memory=True)
    ctx_without = assemble(profile, scored, turns, recall_msg, use_memory=False)

    # 实际生成一条无记忆回复（写入 day2-bare，便于复查）
    bare_result = bare.chat("day2-bare", recall_msg)

    if verbose:
        print(f"助手(无记忆/{bare_result.generated.mode}): {bare_result.reply}")
        print("\n--- 上下文差异 ---")
        print("[有记忆] 画像:\n", ctx_with.profile_text)
        print("[有记忆] 记忆:\n", ctx_with.memory_text)
        print("[无记忆] 画像:\n", ctx_without.profile_text)
        print("[无记忆] 记忆:\n", ctx_without.memory_text)
        print(
            "\n结论：多轮拼接只覆盖本会话窗口；"
            "画像与情景检索才能在换 session 后仍「记得你」。"
        )

    return {
        "profile": profile,
        "memories": memories,
        "with_reply": with_mem.reply,
        "without_reply": bare_result.reply,
        "ctx_with": ctx_with,
        "ctx_without": ctx_without,
        "data_dir": str(base / DEMO_USER),
    }


def ensure_demo_script_file() -> Path:
    """把剧本落到 data/demo，方便讲解时打开看。"""
    path = ROOT / "data" / "demo" / "script.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 演示剧本：day1 建立偏好，day2 新会话回忆",
        "",
    ]
    for session_id, message in SCRIPT:
        lines.append(f"[{session_id}] {message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    ensure_demo_script_file()
    run_demo()
