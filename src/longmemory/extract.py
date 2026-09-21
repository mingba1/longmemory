"""从本轮对话抽取画像事实与情景记忆条目。

有 Key 时让模型返回 JSON；无 Key 时用启发式规则，保证 demo 可跑通。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from longmemory.generate import chat_messages
from longmemory.store import MemoryItem, Profile, new_memory_id

EXTRACT_SYSTEM = (
    "你是记忆抽取器。根据用户本轮发言（可参考助手回复），"
    "提取值得长期保存的信息。只输出 JSON，不要其它文字。\n"
    "格式：\n"
    "{\n"
    '  "facts": [{"key": "短键名", "value": "简洁值"}],\n'
    '  "episode": {"text": "一条情景记忆，可空", "kind": "preference|fact|decision|episode"},\n'
    '  "skip_episode": true/false\n'
    "}\n"
    "规则：寒暄、反问、回忆确认（如「还记得我吗」）不要当新事实；"
    "事实键名用中文短词，如 编程语言、回复风格、职业、姓名；"
    "不要编造用户没说的内容。"
)

_NAME_STOP = {"谁", "什么", "哪", "哪里", "怎么", "怎样", "如何", "你", "我"}


@dataclass
class Extraction:
    facts: list[tuple[str, str]] = field(default_factory=list)
    episode_text: str = ""
    episode_kind: str = "episode"
    skip_episode: bool = False


def _is_recall_or_question(text: str) -> bool:
    t = text.strip()
    if any(p in t for p in ("还记得", "记得我", "你还记得", "我是谁")):
        return True
    if t.endswith("？") or t.endswith("?"):
        # 纯提问且没有「我叫/我写」这类陈述
        if not re.search(r"我(?:叫|是|写|用|做)|叫我|讨厌|喜欢|希望", t):
            return True
    return False


def _heuristic_extract(user_message: str) -> Extraction:
    if _is_recall_or_question(user_message):
        return Extraction(skip_episode=True)

    facts: list[tuple[str, str]] = []

    m = re.search(r"(?:我叫|叫我)\s*([^\s，。！？、；;,]{1,12})", user_message)
    if m and m.group(1) not in _NAME_STOP:
        facts.append(("姓名", m.group(1)))

    m = re.search(
        r"(?:我(?:用|写|做)|擅长)\s*(Python|Java|Go|Rust|TypeScript|JavaScript|C\+\+|C#)",
        user_message,
        re.I,
    )
    if m:
        facts.append(("编程语言", m.group(1)))

    if re.search(r"(?:讨厌|不要|别给)\s*(?:太)?(?:长回复|长答案|废话|啰嗦)", user_message):
        facts.append(("回复风格", "简洁短回复"))
    else:
        m = re.search(r"(?:喜欢|希望|请)\s*(简洁|短一点|简短|详细)", user_message)
        if m:
            facts.append(("回复风格", m.group(1)))

    m = re.search(
        r"(?:我是|职业是|我做)\s*([^\s，。！？、]{2,16}(?:工程师|开发|产品|设计师|学生|老师))",
        user_message,
    )
    if m:
        facts.append(("职业", m.group(1)))

    m = re.search(r"(?:住在|在)\s*([\u4e00-\u9fff]{2,8})(?:市|工作|生活)", user_message)
    if m:
        facts.append(("城市", m.group(1)))

    skip = False
    greetings = ("你好", "在吗", "嗨", "hello", "hi", "早上好", "晚上好")
    stripped = user_message.strip()
    if len(stripped) <= 6 and any(g in stripped.lower() for g in greetings):
        skip = True

    episode_text = ""
    kind = "episode"
    if facts:
        episode_text = "；".join(f"{k}={v}" for k, v in facts)
        kind = (
            "preference"
            if any(k in ("回复风格", "编程语言") for k, _ in facts)
            else "fact"
        )
    elif not skip and len(stripped) >= 8:
        episode_text = stripped[:200]
        kind = "episode"
    else:
        skip = True

    return Extraction(
        facts=facts,
        episode_text=episode_text,
        episode_kind=kind,
        skip_episode=skip,
    )


def _llm_extract(user_message: str, assistant_reply: str) -> Extraction:
    user = f"用户说：{user_message}\n助手答：{assistant_reply}"
    raw = chat_messages(EXTRACT_SYSTEM, [{"role": "user", "content": user}])
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    facts = []
    for item in data.get("facts") or []:
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if key and value:
            facts.append((key, value))
    episode = data.get("episode") or {}
    episode_text = str(episode.get("text") or "").strip()
    kind = str(episode.get("kind") or "episode").strip() or "episode"
    skip = bool(data.get("skip_episode")) or not episode_text
    return Extraction(
        facts=facts,
        episode_text=episode_text,
        episode_kind=kind,
        skip_episode=skip,
    )


def extract(user_message: str, assistant_reply: str = "") -> Extraction:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _llm_extract(user_message, assistant_reply)
        except Exception:
            return _heuristic_extract(user_message)
    return _heuristic_extract(user_message)


def apply_extraction(
    profile: Profile,
    extraction: Extraction,
    session_id: str,
) -> tuple[Profile, MemoryItem | None]:
    for key, value in extraction.facts:
        profile.set_fact(key, value, source="extract")

    if extraction.skip_episode or not extraction.episode_text:
        return profile, None

    item = MemoryItem(
        id=new_memory_id(),
        text=extraction.episode_text,
        kind=extraction.episode_kind,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        session_id=session_id,
        meta={"source": "extract"},
    )
    return profile, item
