"""按字符预算组装上下文：画像 → 相关记忆 → 最近对话 → 当前输入。

超预算优先砍旧轮次，再砍低分记忆。刻意不把整段历史无限追加。
"""

from __future__ import annotations

from dataclasses import dataclass

from longmemory.config import (
    DEFAULT_CONTEXT_BUDGET,
    DEFAULT_RECENT_TURNS,
    MEMORY_BUDGET,
    PROFILE_BUDGET,
)
from longmemory.retrieve import ScoredMemory
from longmemory.store import Profile, Turn

SYSTEM_BASE = (
    "你是一个带长期记忆的助手。"
    "下面「用户画像」和「相关记忆」来自跨会话存储，不是本轮临时拼接的闲聊记录。"
    "回答时优先遵循画像中的偏好；需要引用细节时用相关记忆。"
    "不要假装记得未列出的信息；不确定就诚实说不知道。"
)


@dataclass
class AssembledContext:
    system: str
    messages: list[dict[str, str]]
    profile_text: str
    memory_text: str
    recent_turns_used: int
    memories_used: int
    char_count: int


def _trim_tail(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 1)] + "…"


def format_profile(profile: Profile, budget: int = PROFILE_BUDGET) -> str:
    lines = profile.as_prompt_lines()
    if not lines:
        return "（暂无画像）"
    body = "\n".join(lines)
    return _trim_tail(body, budget)


def format_memories(
    scored: list[ScoredMemory],
    budget: int = MEMORY_BUDGET,
) -> tuple[str, int]:
    if not scored:
        return "（本轮未检索到相关情景记忆）", 0
    blocks: list[str] = []
    used = 0
    size = 0
    for item in scored:
        block = f"- [{item.item.id}|{item.score:.2f}|{item.item.kind}] {item.item.text}"
        if size + len(block) + 1 > budget and used > 0:
            break
        blocks.append(block)
        size += len(block) + 1
        used += 1
    return "\n".join(blocks), used


def recent_pairs(turns: list[Turn], max_turns: int = DEFAULT_RECENT_TURNS) -> list[Turn]:
    """取最近 max_turns 对对话（按消息条数近似：最后 2*max_turns 条）。"""
    n = max_turns * 2
    return turns[-n:] if len(turns) > n else list(turns)


def assemble(
    profile: Profile,
    scored_memories: list[ScoredMemory],
    turns: list[Turn],
    user_message: str,
    *,
    use_memory: bool = True,
    budget: int = DEFAULT_CONTEXT_BUDGET,
    recent_turns: int = DEFAULT_RECENT_TURNS,
) -> AssembledContext:
    if use_memory:
        profile_text = format_profile(profile)
        memory_text, memories_used = format_memories(scored_memories)
    else:
        profile_text = "（本模式关闭长期记忆）"
        memory_text = "（本模式关闭长期记忆）"
        memories_used = 0

    system = (
        f"{SYSTEM_BASE}\n\n"
        f"## 用户画像\n{profile_text}\n\n"
        f"## 相关记忆\n{memory_text}"
    )

    recent = recent_pairs(turns, recent_turns)
    # 预留 system + 当前用户消息
    fixed = len(system) + len(user_message) + 64
    remain = max(200, budget - fixed)

    # 从旧到新砍：保留尽可能多的尾部轮次
    kept: list[Turn] = []
    used_chars = 0
    for turn in reversed(recent):
        cost = len(turn.content) + 16
        if used_chars + cost > remain and kept:
            break
        kept.append(turn)
        used_chars += cost
    kept.reverse()

    # 若仍然超预算，再砍记忆区（重建 system）
    char_count = len(system) + used_chars + len(user_message)
    if char_count > budget and use_memory and scored_memories:
        reduced = scored_memories[: max(1, len(scored_memories) // 2)]
        memory_text, memories_used = format_memories(reduced, budget=MEMORY_BUDGET // 2)
        system = (
            f"{SYSTEM_BASE}\n\n"
            f"## 用户画像\n{profile_text}\n\n"
            f"## 相关记忆\n{memory_text}"
        )
        char_count = len(system) + used_chars + len(user_message)

    messages: list[dict[str, str]] = [
        {"role": t.role, "content": t.content} for t in kept
    ]
    messages.append({"role": "user", "content": user_message})

    return AssembledContext(
        system=system,
        messages=messages,
        profile_text=profile_text,
        memory_text=memory_text,
        recent_turns_used=len(kept),
        memories_used=memories_used,
        char_count=char_count,
    )
