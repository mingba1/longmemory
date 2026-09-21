"""默认选择。改预算和 Top-K 前，先跑 demo 看有/无记忆对比。"""

from pathlib import Path

# 包所在仓库根目录（…/longMemory）
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "users"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# bge 中文检索：只加在查询侧。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

DEFAULT_TOP_K = 5
# BGE 余弦通常偏高；哈希兜底分数更分散，阈值放宽以便 demo 仍能命中。
DEFAULT_MIN_COSINE = 0.15
# 本会话最多保留的最近轮数（一对 user/assistant 算一轮）
DEFAULT_RECENT_TURNS = 6
# 组装进 prompt 的字符预算（粗略代替 tokenizer）
DEFAULT_CONTEXT_BUDGET = 3500
# 画像区、记忆区各自预留上限
PROFILE_BUDGET = 800
MEMORY_BUDGET = 1200
