"""单轮闭环：检索 → 组装 → 生成 → 抽取落盘。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from longmemory.config import DEFAULT_DATA_DIR, DEFAULT_MIN_COSINE, DEFAULT_TOP_K
from longmemory.context import AssembledContext, assemble
from longmemory.embed import Embedder
from longmemory.extract import apply_extraction, extract
from longmemory.generate import Generated, generate
from longmemory.retrieve import ScoredMemory, retrieve_memories
from longmemory.store import MemoryItem, Profile, UserStore


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@dataclass
class TurnResult:
    reply: str
    generated: Generated
    context: AssembledContext
    scored_memories: list[ScoredMemory]
    new_memory: MemoryItem | None
    profile: Profile


class MemoryAssistant:
    def __init__(
        self,
        user_id: str,
        data_dir: Path | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_COSINE,
        use_memory: bool = True,
        embedder: Embedder | None = None,
    ) -> None:
        self.store = UserStore(user_id, data_dir or DEFAULT_DATA_DIR)
        self.top_k = top_k
        self.min_score = min_score
        self.use_memory = use_memory
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def chat(self, session_id: str, user_message: str) -> TurnResult:
        profile = self.store.load_profile()
        memories = self.store.load_memories()
        vectors = self.store.load_vectors()
        turns = self.store.load_turns(session_id)

        scored: list[ScoredMemory] = []
        if self.use_memory and memories:
            q = self.embedder.embed_query(user_message)
            scored = retrieve_memories(
                q,
                memories,
                vectors,
                top_k=self.top_k,
                min_score=self.min_score,
            )

        ctx = assemble(
            profile,
            scored,
            turns,
            user_message,
            use_memory=self.use_memory,
        )
        generated = generate(
            ctx.system,
            ctx.messages,
            ctx.profile_text,
            ctx.memory_text,
        )

        self.store.append_turn(session_id, "user", user_message)
        self.store.append_turn(session_id, "assistant", generated.text)

        new_memory: MemoryItem | None = None
        if self.use_memory:
            extraction = extract(user_message, generated.text)
            profile, new_memory = apply_extraction(profile, extraction, session_id)
            self.store.save_profile(profile)
            if new_memory is not None:
                vec = self.embedder.embed_passages([new_memory.text])[0]
                self.store.append_memory(new_memory, vec)

        return TurnResult(
            reply=generated.text,
            generated=generated,
            context=ctx,
            scored_memories=scored,
            new_memory=new_memory,
            profile=profile,
        )

    def show(self) -> tuple[Profile, list[MemoryItem]]:
        return self.store.load_profile(), self.store.load_memories()

    def reset(self) -> None:
        self.store.reset()
