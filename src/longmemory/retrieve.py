"""情景记忆检索：归一化向量的点积即余弦。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from longmemory.config import DEFAULT_MIN_COSINE, DEFAULT_TOP_K
from longmemory.store import MemoryItem


@dataclass(frozen=True)
class ScoredMemory:
    item: MemoryItem
    score: float
    index: int


def retrieve_memories(
    query_vec: np.ndarray,
    memories: list[MemoryItem],
    vectors: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_COSINE,
) -> list[ScoredMemory]:
    if not memories or len(vectors) == 0:
        return []
    if len(memories) != len(vectors):
        raise ValueError("memories 与 vectors 长度不一致")
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    # 已归一化时点积 = 余弦
    scores = vectors @ q
    order = np.argsort(-scores)
    results: list[ScoredMemory] = []
    for idx in order[:top_k]:
        score = float(scores[idx])
        if score < min_score:
            continue
        results.append(ScoredMemory(item=memories[int(idx)], score=score, index=int(idx)))
    return results
