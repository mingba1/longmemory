"""聊天生成：有 Key 走 OpenAI 兼容接口；无 Key 用画像+记忆规则兜底。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Generated:
    text: str
    mode: str  # llm | fallback


def chat_messages(system: str, messages: list[dict[str, str]]) -> str:
    import httpx

    api_key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload_messages = [{"role": "system", "content": system}, *messages]
    response = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0.3,
            "messages": payload_messages,
        },
        timeout=60.0,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"大模型接口返回 {response.status_code}：{response.text[:500]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _facts_from_profile_text(profile_text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in profile_text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ":" in body:
            key, value = body.split(":", 1)
            facts[key.strip()] = value.strip()
    return facts


def _memory_bullets(memory_text: str) -> list[str]:
    bullets: list[str] = []
    for line in memory_text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        # 去掉 [id|score|kind] 前缀
        text = re.sub(r"^-\s*\[[^\]]+\]\s*", "", line).strip()
        if text and "未检索" not in text and "关闭长期记忆" not in text:
            bullets.append(text)
    return bullets


def fallback_answer(
    user_message: str,
    profile_text: str,
    memory_text: str,
) -> str:
    """无 API Key 时仍能演示「记得你」：回显画像命中与相关记忆。"""
    facts = _facts_from_profile_text(profile_text)
    memories = _memory_bullets(memory_text)
    parts: list[str] = []

    if facts:
        hits = [f"{k}={v}" for k, v in list(facts.items())[:6]]
        parts.append("根据已记住的画像：" + "；".join(hits) + "。")
    else:
        parts.append("目前还没有稳定画像。你可以先告诉我偏好或身份信息。")

    if memories:
        parts.append("相关记忆：" + "；".join(memories[:3]) + "。")

    if any(w in user_message for w in ("我是谁", "记得我", "还记得", "我的偏好", "用什么语言")):
        name = facts.get("姓名")
        lang = facts.get("编程语言") or facts.get("语言") or facts.get("tech_stack")
        style = facts.get("回复风格") or facts.get("偏好")
        bits = []
        if name:
            bits.append(f"你是{name}")
        if lang:
            bits.append(f"常用{lang}")
        if style:
            bits.append(f"希望{style}")
        if bits:
            parts = ["记得：" + "，".join(bits) + "。"]
            if memories:
                parts.append("相关记忆：" + "；".join(memories[:2]) + "。")

    if facts.get("回复风格") and (
        "短" in facts["回复风格"] or "简洁" in facts["回复风格"]
    ):
        return " ".join(parts[:2]) if len(parts) > 1 else parts[0]

    return "\n".join(parts)


def generate(
    system: str,
    messages: list[dict[str, str]],
    profile_text: str,
    memory_text: str,
) -> Generated:
    if os.environ.get("OPENAI_API_KEY"):
        return Generated(chat_messages(system, messages), "llm")
    user_message = messages[-1]["content"] if messages else ""
    return Generated(
        fallback_answer(user_message, profile_text, memory_text),
        "fallback",
    )
