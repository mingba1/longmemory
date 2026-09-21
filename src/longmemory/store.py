"""用户目录持久化：画像 JSON、情景记忆 + 向量、会话 jsonl。"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from longmemory.config import DEFAULT_DATA_DIR


@dataclass
class Profile:
    """稳定偏好 / 身份 / 长期需求。facts 里 key 冲突时覆盖并记 updated_at。"""

    facts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def set_fact(self, key: str, value: str, source: str = "extract") -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        prev = self.facts.get(key)
        self.facts[key] = {
            "value": value,
            "updated_at": now,
            "source": source,
            "previous": prev["value"] if prev else None,
        }

    def as_prompt_lines(self) -> list[str]:
        if not self.facts:
            return []
        lines = []
        for key, item in sorted(self.facts.items()):
            lines.append(f"- {key}: {item['value']}")
        return lines


@dataclass
class MemoryItem:
    id: str
    text: str
    kind: str  # preference | fact | decision | episode
    created_at: str
    session_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    role: str  # user | assistant
    content: str
    ts: str


class UserStore:
    def __init__(self, user_id: str, data_dir: Path | None = None) -> None:
        self.user_id = user_id
        self.root = (data_dir or DEFAULT_DATA_DIR) / user_id
        self.profile_path = self.root / "profile.json"
        self.memories_path = self.root / "memories.json"
        self.vectors_path = self.root / "vectors.npy"
        self.sessions_dir = self.root / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def load_profile(self) -> Profile:
        if not self.profile_path.exists():
            return Profile()
        raw = json.loads(self.profile_path.read_text(encoding="utf-8"))
        return Profile(facts=raw.get("facts", {}))

    def save_profile(self, profile: Profile) -> None:
        payload = {"facts": profile.facts}
        self.profile_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_memories(self) -> list[MemoryItem]:
        if not self.memories_path.exists():
            return []
        raw = json.loads(self.memories_path.read_text(encoding="utf-8"))
        return [MemoryItem(**item) for item in raw]

    def load_vectors(self) -> np.ndarray:
        if not self.vectors_path.exists():
            return np.zeros((0, 0), dtype=np.float32)
        return np.load(self.vectors_path).astype(np.float32)

    def save_memories(self, memories: list[MemoryItem], vectors: np.ndarray) -> None:
        if len(memories) != len(vectors):
            raise ValueError(
                f"记忆条数 {len(memories)} 与向量行数 {len(vectors)} 不一致"
            )
        payload = [asdict(m) for m in memories]
        self.memories_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if len(vectors) == 0:
            if self.vectors_path.exists():
                self.vectors_path.unlink()
        else:
            np.save(self.vectors_path, vectors.astype(np.float32))

    def append_memory(self, item: MemoryItem, vector: np.ndarray) -> None:
        memories = self.load_memories()
        vectors = self.load_vectors()
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if len(vectors) == 0:
            vectors = vec
        else:
            if vectors.shape[1] != vec.shape[1]:
                raise ValueError("向量维度与已有记忆不一致")
            vectors = np.vstack([vectors, vec])
        memories.append(item)
        self.save_memories(memories, vectors)

    def session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.jsonl"

    def load_turns(self, session_id: str) -> list[Turn]:
        path = self.session_path(session_id)
        if not path.exists():
            return []
        turns: list[Turn] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            turns.append(Turn(**raw))
        return turns

    def append_turn(self, session_id: str, role: str, content: str) -> Turn:
        turn = Turn(
            role=role,
            content=content,
            ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        path = self.session_path(session_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        return turn

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)


def new_memory_id() -> str:
    return f"mem-{uuid.uuid4().hex[:10]}"
