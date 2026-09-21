"""嵌入：优先本地 BGE；无 torch 时用字符 n-gram 哈希向量兜底。

Python 3.14 等环境可能还没有 torch 轮子，兜底仍可演示「检索情景记忆」这条链路。
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from longmemory.config import EMBED_MODEL, QUERY_INSTRUCTION

FALLBACK_DIM = 384


def _try_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception:
        return None


def _tokenize(text: str) -> list[str]:
    """中英混合：单字、二字、三字、以及英文词。"""
    text = text.lower().strip()
    grams: list[str] = []
    # 连续中文块
    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        grams.extend(list(block))
        grams.extend(block[i : i + 2] for i in range(len(block) - 1))
        grams.extend(block[i : i + 3] for i in range(max(0, len(block) - 2)))
    # 英文/数字词
    for token in re.findall(r"[a-z0-9_+#]+", text):
        grams.append(token)
        if len(token) >= 3:
            grams.extend(token[i : i + 3] for i in range(len(token) - 2))
    return grams or [text]


def _hash_embed(text: str, dim: int = FALLBACK_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    grams = _tokenize(text)
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        # 更长短语略加权
        weight = 1.0 + 0.25 * min(len(gram), 4)
        vec[idx] += sign * weight
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec /= norm
    return vec


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self.model_name = model_name
        self._model = _try_sentence_transformer(model_name)
        self.backend = "bge" if self._model is not None else "hash"

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return FALLBACK_DIM

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._model is not None:
            array = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(array, dtype=np.float32)
        return np.stack([_hash_embed(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if self._model is not None:
            array = self._model.encode(
                [QUERY_INSTRUCTION + text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(array[0], dtype=np.float32)
        return _hash_embed(text)
