from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import Settings, load_settings
from .inputs import InputError, parse_bilibili_input, parse_local_mp4
from .models import InputSpec
from .naming import build_archive_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bili-summary",
        description="哔哩哔哩与本地 MP4 学习资料整理工具（阶段 1：离线预览）",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, help="非秘密 INI 配置文件路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="预览哔哩哔哩链接任务")
    run_parser.add_argument("source", help="哔哩哔哩链接、b23.tv 短链接或 BV 号")
    _add_archive_arguments(run_parser)

    file_parser = subparsers.add_parser("run-file", help="预览单个本地 MP4 任务")
    file_parser.add_argument("source", help="Windows 盘符路径或 WSL 绝对路径")
    file_parser.add_argument(
        "--hash",
        action="store_true",
        help="顺序读取文件并计算 SHA-256；大文件默认延后",
    )
    _add_archive_arguments(file_parser)

    doctor_parser = subparsers.add_parser("doctor", help="只读显示当前运行能力")
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _add_archive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject", default="未分类", help="学科目录")
    parser.add_argument("--course", help="可选课程目录")
    parser.add_argument("--title", help="覆盖显示标题")
    parser.add_argument("--json", action="store_true", dest="as_json", help="输出 JSON")


def _preview(spec: InputSpec, args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    title = args.title or spec.display_title
    archive_path = build_archive_path(
        settings.data_root,
        subject=args.subject,
        course=args.course,
        title=title,
    )
    return {
        "status": "preview_only",
        "input": spec.to_dict(),
        "archive_path": str(archive_path),
        "processing": {
            "audit_level": settings.audit_level,
            "transcriber_mode": settings.transcriber_mode,
            "cost_submission_limit_cny": settings.cost_submission_limit_cny,
        },
        "notice": "阶段 1 未下载、未转写、未调用 Codex，也未写入成果目录",
    }


def _doctor() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    return {
        "python": sys.version.split()[0],
        "project_root": str(project_root),
        "venv_active": sys.prefix != sys.base_prefix,
        "codex": shutil.which("codex"),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "valid_git_worktree": (project_root / ".git" / "HEAD").exists(),
        "stage": 1,
    }


def _print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if result.get("stage") == 1 and "python" in result:
        print("阶段 1 只读环境检查")
        for key, value in result.items():
            print(f"{key}: {value}")
        return
    input_data = result["input"]
    print("阶段 1 离线任务预览")
    print(f"输入类型: {input_data['source_type']}")
    print(f"规范输入: {input_data['canonical']}")
    print(f"成果目录: {result['archive_path']}")
    print(result["notice"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command == "doctor":
            _print_result(_doctor(), args.as_json)
            return 0
        if args.command == "run":
            spec = parse_bilibili_input(args.source)
        else:
            spec = parse_local_mp4(args.source, compute_hash=args.hash)
        _print_result(_preview(spec, args, settings), args.as_json)
        return 0
    except (InputError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
