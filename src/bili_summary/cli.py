from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .adapters import TextBackendError
from .bilibili import BilibiliError
from .config import Settings, load_settings
from .evaluation import EVALUATION_TASKS, run_text_profile_comparison
from .inputs import InputError, parse_bilibili_input, parse_local_mp4
from .long_pipeline import run_bilibili_long_pipeline
from .models import InputSpec
from .naming import build_archive_path
from .text_routing import (
    choose_execution_plan,
    parse_route_overrides,
    resolve_text_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bili-summary",
        description="哔哩哔哩与本地 MP4 学习资料整理工具（阶段 3.1）",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, help="非秘密 INI 配置文件路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="预览或执行哔哩哔哩链接任务")
    run_parser.add_argument("source", help="哔哩哔哩链接、b23.tv 短链接或 BV 号")
    run_parser.add_argument(
        "--execute",
        action="store_true",
        help="联网获取平台字幕，确认方案后调用所选文本后端；省略时只预览",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖同名已完成文本成果，并按本次文本方案重新处理",
    )
    run_parser.add_argument(
        "--long",
        action="store_true",
        help="显式标记长字幕任务（阶段 3.1 的执行流程均支持分片恢复）",
    )
    run_parser.add_argument(
        "--compare-deep",
        action="store_true",
        help="长流程完成 basic 后，独立执行 deep 全文审校并生成对比文件",
    )
    run_parser.add_argument(
        "--audit-level",
        choices=("off", "basic", "deep"),
        help="覆盖本次审校档位；--compare-deep 等同于 deep",
    )
    run_parser.add_argument(
        "--profile",
        help="使用 config.ini 中的整体文本预设，例如 quality 或 speed",
    )
    run_parser.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="TASK=PROFILE",
        help="覆盖一个文本任务的命名配置，可重复使用",
    )
    run_parser.add_argument(
        "--yes",
        action="store_true",
        help="确认已显示/指定的文本方案，供非交互运行使用",
    )
    _add_archive_arguments(run_parser)

    file_parser = subparsers.add_parser("run-file", help="预览单个本地 MP4 任务")
    file_parser.add_argument("source", help="Windows 盘符路径或 WSL 绝对路径")
    file_parser.add_argument(
        "--hash",
        action="store_true",
        help="顺序读取文件并计算 SHA-256；大文件默认延后",
    )
    _add_archive_arguments(file_parser)

    compare_parser = subparsers.add_parser(
        "compare-text",
        help="用同一份 SRT 比较多个文本配置，不改写学习成果",
    )
    compare_parser.add_argument("source", type=Path, help="用于对比的 SRT 文件")
    compare_parser.add_argument(
        "--profiles",
        nargs="+",
        default=["deepseek_flash_low", "deepseek_pro_high"],
        help="要比较的命名文本配置",
    )
    compare_parser.add_argument(
        "--tasks",
        nargs="+",
        choices=EVALUATION_TASKS,
        default=list(EVALUATION_TASKS),
        help="要比较的文本任务",
    )
    compare_parser.add_argument("--output", type=Path, help="覆盖对比成果目录")
    compare_parser.add_argument("--force", action="store_true", help="忽略已完成的对比缓存")
    compare_parser.add_argument("--yes", action="store_true", help="确认真实文本调用")
    compare_parser.add_argument("--json", action="store_true", dest="as_json")

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
            "text_plan": resolve_text_plan(
                settings,
                audit_level=(
                    "deep"
                    if getattr(args, "compare_deep", False)
                    else (getattr(args, "audit_level", None) or settings.audit_level)
                ),
                preset=getattr(args, "profile", None),
                route_overrides=parse_route_overrides(getattr(args, "route", [])),
            ).to_dict(),
        },
        "notice": "预览模式未联网、未转写、未调用文本模型，也未写入成果目录",
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
        "bilibili_cookie_configured": False,
        "stage": "3.1",
    }


def _print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if result.get("stage") == "3.1" and "python" in result:
        print("阶段 3.1 只读环境检查")
        for key, value in result.items():
            print(f"{key}: {value}")
        return
    if "input" in result:
        input_data = result["input"]
        print("离线任务预览")
        print(f"输入类型: {input_data['source_type']}")
        print(f"规范输入: {input_data['canonical']}")
        print(f"成果目录: {result['archive_path']}")
        print(result["notice"])
        return
    print(f"任务状态: {result['status']}")
    print(f"成果目录: {result['output_dir']}")
    if result.get("notice"):
        print(result["notice"])
    for file_name in result.get("files", []):
        print(f"- {file_name}")


def _stderr_input(prompt: str) -> str:
    print(prompt, end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().rstrip("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command == "doctor":
            doctor = _doctor()
            doctor["bilibili_cookie_configured"] = bool(settings.bilibili_cookie_file)
            if settings.bilibili_cookie_file:
                doctor["bilibili_cookie_file_exists"] = settings.bilibili_cookie_file.is_file()
            doctor["codex_model"] = settings.codex_model or "codex-cli-default"
            doctor["text_profiles"] = {
                name: {
                    "driver": profile.driver,
                    "model": profile.model or "后端默认模型",
                    "reasoning": profile.reasoning,
                }
                for name, profile in settings.text_profiles.items()
            }
            doctor["text_routes"] = settings.text_routes
            doctor["text_presets"] = sorted(settings.text_presets)
            doctor["deepseek_credentials_configured"] = bool(
                os.environ.get(settings.deepseek_api_key_env)
                or (
                    settings.deepseek_api_key_file
                    and settings.deepseek_api_key_file.is_file()
                )
            )
            doctor["long_chunk_minutes"] = {
                "target": settings.long_chunk_target_minutes,
                "max": settings.long_chunk_max_minutes,
                "deep_target": settings.deep_chunk_target_minutes,
                "deep_max": settings.deep_chunk_max_minutes,
            }
            _print_result(doctor, args.as_json)
            return 0
        if args.command == "compare-text":
            if not args.yes:
                if not sys.stdin.isatty():
                    raise ValueError("非交互对比必须增加 --yes")
                print(
                    f"将对 {args.source} 执行 {len(args.profiles) * len(args.tasks)} 次文本调用。",
                    file=sys.stderr,
                )
                if _stderr_input("输入 yes 继续，其他内容取消: ").strip().lower() != "yes":
                    raise ValueError("用户取消；未调用文本模型")
            result = run_text_profile_comparison(
                args.source,
                settings,
                profile_names=tuple(args.profiles),
                tasks=tuple(args.tasks),
                schemas_dir=Path(__file__).resolve().parents[2] / "schemas",
                output_dir=args.output,
                force=args.force,
                progress=lambda message: print(message, file=sys.stderr),
            )
            _print_result(result, args.as_json)
            return 0
        if args.command == "run":
            spec = parse_bilibili_input(args.source)
            if args.compare_deep and not args.long:
                raise ValueError("--compare-deep 只能与 --long 一起使用")
            if args.execute:
                project_root = Path(__file__).resolve().parents[2]
                requested_audit = (
                    "deep" if args.compare_deep else (args.audit_level or settings.audit_level)
                )
                route_overrides = parse_route_overrides(args.route)
                result = run_bilibili_long_pipeline(
                    spec,
                    settings,
                    subject=args.subject,
                    course=args.course,
                    title_override=args.title,
                    schemas_dir=project_root / "schemas",
                    compare_deep=args.compare_deep,
                    audit_level=requested_audit,
                    force=args.force,
                    plan_selector=lambda preview: choose_execution_plan(
                        settings,
                        preview=preview,
                        audit_level=requested_audit,
                        preset=args.profile,
                        route_overrides=route_overrides,
                        assume_yes=args.yes,
                        interactive=sys.stdin.isatty(),
                        input_fn=_stderr_input,
                        output_fn=lambda message: print(message, file=sys.stderr),
                    ),
                    progress=lambda message: print(message, file=sys.stderr),
                )
                _print_result(result, args.as_json)
                return 0
        else:
            spec = parse_local_mp4(args.source, compute_hash=args.hash)
        _print_result(_preview(spec, args, settings), args.as_json)
        return 0
    except (BilibiliError, TextBackendError, InputError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
