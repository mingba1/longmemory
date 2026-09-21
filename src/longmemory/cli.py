"""CLI：chat / show / reset / demo。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from longmemory.config import DEFAULT_DATA_DIR, DEFAULT_MIN_COSINE, DEFAULT_TOP_K
from longmemory.demo import ensure_demo_script_file, run_demo
from longmemory.pipeline import MemoryAssistant


def _load_dotenv() -> None:
    """轻量加载仓库根目录 .env，不引入 python-dotenv 依赖。"""
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def cmd_chat(args: argparse.Namespace) -> int:
    assistant = MemoryAssistant(
        args.user,
        data_dir=Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR,
        top_k=args.top_k,
        min_score=args.min_score,
        use_memory=not args.no_memory,
    )
    print(
        f"长期记忆助手 | user={args.user} session={args.session} "
        f"memory={'on' if not args.no_memory else 'off'}"
    )
    print("输入消息回车发送；空行或 Ctrl-D / Ctrl-C 结束。\n")
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        result = assistant.chat(args.session, line)
        print(f"助手[{result.generated.mode}]> {result.reply}")
        if args.verbose:
            print(
                f"  (记忆命中 {result.context.memories_used}, "
                f"最近消息 {result.context.recent_turns_used}, "
                f"字符约 {result.context.char_count})"
            )
            if result.new_memory:
                print(f"  + 写入情景: {result.new_memory.text}")
        print()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    assistant = MemoryAssistant(
        args.user,
        data_dir=Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR,
    )
    profile, memories = assistant.show()
    print(f"== 画像 user={args.user} ==")
    lines = profile.as_prompt_lines()
    if not lines:
        print("（空）")
    else:
        print("\n".join(lines))
    print(f"\n== 情景记忆 ({len(memories)}) ==")
    if not memories:
        print("（空）")
    else:
        for item in memories:
            print(f"- [{item.id}|{item.kind}|{item.session_id}] {item.text}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    assistant = MemoryAssistant(
        args.user,
        data_dir=Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR,
    )
    assistant.reset()
    print(f"已清空 user={args.user} 的画像、情景记忆与会话。")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    path = ensure_demo_script_file()
    print(f"剧本已写入: {path}")
    data_dir = Path(args.data_dir) if args.data_dir else None
    run_demo(data_dir=data_dir, verbose=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longmemory",
        description="教学向长期记忆 AI 助手：画像 + 情景检索 + 上下文裁剪",
    )
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--data-dir",
        default="",
        help=f"用户数据根目录（默认 {DEFAULT_DATA_DIR}）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", parents=[parent], help="交互多轮；换 --session 模拟隔天再聊")
    chat.add_argument("--user", default="default")
    chat.add_argument("--session", default="s1")
    chat.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    chat.add_argument("--min-score", type=float, default=DEFAULT_MIN_COSINE)
    chat.add_argument("--no-memory", action="store_true", help="关闭长期记忆注入与写入")
    chat.add_argument("-v", "--verbose", action="store_true")
    chat.set_defaults(func=cmd_chat)

    show = sub.add_parser("show", parents=[parent], help="打印画像与情景记忆")
    show.add_argument("--user", default="default")
    show.set_defaults(func=cmd_show)

    reset = sub.add_parser("reset", parents=[parent], help="清空指定用户记忆")
    reset.add_argument("--user", default="default")
    reset.set_defaults(func=cmd_reset)

    demo = sub.add_parser("demo", parents=[parent], help="跨会话剧本 + 有/无记忆对比")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "data_dir", ""):
        args.data_dir = ""
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
