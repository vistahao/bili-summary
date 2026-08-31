from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .adapters import TextBackendError
from .aliyun_asr import (
    AliyunAsrError,
    AliyunTemporaryUploadClient,
    COMPARISON_MODELS,
    estimate_comparison_cost_cny,
    load_aliyun_asr_api_key,
)
from .asr_evaluation import run_aliyun_asr_comparison
from .bilibili import BilibiliError
from .cache_management import clean_cache, inspect_cache
from .config import Settings, load_settings
from .evaluation import EVALUATION_TASKS, run_text_profile_comparison
from .inputs import InputError, parse_bilibili_input, parse_local_mp4
from .local_asr import LocalAsrError
from .local_pipeline import (
    completed_local_result,
    estimate_local_online_cost,
    run_local_file_pipeline,
)
from .long_pipeline import run_bilibili_long_pipeline
from .media import MediaError, inspect_local_media, prepare_transcription_sample
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
        description="哔哩哔哩与本地 MP4 学习资料整理工具（阶段 5）",
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
        "--content-mode",
        choices=("lecture", "practice"),
        help="内容整理模式：知识讲座 lecture 或刷题课 practice",
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
        "--execute",
        action="store_true",
        help="取得字幕或转写后执行完整文本流水线；省略时只预览",
    )
    file_parser.add_argument("--force", action="store_true", help="按本次方案重做文本成果")
    file_parser.add_argument("--long", action="store_true", help="标记长本地课程（当前流程自动分片）")
    file_parser.add_argument(
        "--compare-deep",
        action="store_true",
        help="完成 Basic 后独立执行 Deep 审校并生成对比",
    )
    file_parser.add_argument(
        "--audit-level",
        choices=("off", "basic", "deep"),
        help="覆盖本次审校档位",
    )
    file_parser.add_argument(
        "--content-mode",
        choices=("lecture", "practice"),
        help="内容整理模式：知识讲座 lecture 或刷题课 practice",
    )
    file_parser.add_argument("--profile", help="使用 config.ini 中的整体文本预设")
    file_parser.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="TASK=PROFILE",
        help="覆盖一个文本任务的命名配置，可重复使用",
    )
    file_parser.add_argument(
        "--transcriber",
        choices=("auto", "online", "local"),
        help="覆盖本次无字幕转写模式",
    )
    file_parser.add_argument(
        "--yes",
        action="store_true",
        help="确认音频上传、费用门槛和文本方案，供非交互运行使用",
    )
    file_parser.add_argument(
        "--hash",
        action="store_true",
        help="顺序读取文件并计算 SHA-256；大文件默认延后",
    )
    file_parser.add_argument(
        "--probe",
        action="store_true",
        help="用 ffprobe 只读检查媒体轨道和字幕来源",
    )
    file_parser.add_argument(
        "--prepare-audio-sample",
        action="store_true",
        help="无字幕时生成供转写比较使用的临时 WAV，不修改原 MP4",
    )
    file_parser.add_argument(
        "--sample-start",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="音频样本起点秒数（默认 0）",
    )
    file_parser.add_argument(
        "--sample-minutes",
        type=int,
        choices=range(5, 11),
        default=10,
        metavar="5-10",
        help="音频样本分钟数（默认 10）",
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

    asr_check_parser = subparsers.add_parser(
        "check-aliyun-asr",
        help="验证两个阿里云模型的鉴权和临时上传权限，不上传音频、不提交转写",
    )
    asr_check_parser.add_argument("--json", action="store_true", dest="as_json")

    asr_compare_parser = subparsers.add_parser(
        "compare-aliyun-asr",
        help="上传同一 WAV 并执行 Qwen3 Filetrans 与 Paraformer v2 真实比较",
    )
    asr_compare_parser.add_argument("source", type=Path, help="5～10 分钟的 16 kHz 单声道 WAV")
    asr_compare_parser.add_argument("--output", type=Path, help="覆盖对比成果目录")
    asr_compare_parser.add_argument("--yes", action="store_true", help="确认上传与最多 1 元提交门槛")
    asr_compare_parser.add_argument("--json", action="store_true", dest="as_json")

    cache_status_parser = subparsers.add_parser(
        "cache-status",
        help="只读显示缓存占用、受管临时音频和5天到期状态",
    )
    cache_status_parser.add_argument("--json", action="store_true", dest="as_json")

    cache_clean_parser = subparsers.add_parser(
        "cache-clean",
        help="预览或清理已满5天的受管临时音频；默认不删除",
    )
    cache_clean_parser.add_argument(
        "--execute",
        action="store_true",
        help="实际删除通过校验且已到期的临时音频及其元数据",
    )
    cache_clean_parser.add_argument(
        "--yes",
        action="store_true",
        help="确认执行删除，供非交互运行使用",
    )
    cache_clean_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _add_archive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject", default="未分类", help="学科目录")
    parser.add_argument("--course", help="可选课程目录")
    parser.add_argument("--title", help="覆盖显示标题")
    parser.add_argument("--json", action="store_true", dest="as_json", help="输出 JSON")


def _preview(spec: InputSpec, args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    title = args.title or spec.display_title
    audit_level = (
        "deep"
        if getattr(args, "compare_deep", False)
        else (getattr(args, "audit_level", None) or settings.audit_level)
    )
    content_mode = getattr(args, "content_mode", None) or settings.content_mode
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
            "audit_level": audit_level,
            "content_mode": content_mode,
            "transcriber_mode": settings.transcriber_mode,
            "cost_submission_limit_cny": settings.cost_submission_limit_cny,
            "text_plan": resolve_text_plan(
                settings,
                audit_level=audit_level,
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
        "stage": "5",
    }


def _print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if result.get("stage") == "5" and "python" in result:
        print("阶段 5 只读环境检查")
        for key, value in result.items():
            print(f"{key}: {value}")
        return
    if "input" in result:
        input_data = result["input"]
        print("临时音频样本已就绪" if result.get("status") == "audio_sample_ready" else "离线任务预览")
        print(f"输入类型: {input_data['source_type']}")
        print(f"规范输入: {input_data['canonical']}")
        print(f"成果目录: {result['archive_path']}")
        temporary_audio = result.get("temporary_audio")
        if isinstance(temporary_audio, dict):
            audio = temporary_audio.get("audio", {})
            print(f"临时音频: {audio.get('path')}")
            print(f"复用缓存: {temporary_audio.get('reused')}")
            print(f"可清理时间: {temporary_audio.get('eligible_for_cleanup_at')}")
        print(result["notice"])
        return
    if result.get("status") == "aliyun_asr_preflight_complete":
        print("阿里云 ASR 调用前检查完成")
        for model in result["models"]:
            print(
                f"{model['model']}: 临时上传权限可用；"
                f"单文件上限 {model.get('max_file_size_mb') or '未知'} MiB"
            )
        print(result["notice"])
        return
    if result.get("status") in {
        "cache_inventory",
        "cache_cleanup_preview",
        "cache_cleanup_complete",
    }:
        print(f"缓存目录: {result['cache_root']}")
        if result["status"] == "cache_inventory":
            print(f"总占用: {_format_bytes(int(result['total_bytes']))}")
            print(f"受管临时音频: {_format_bytes(int(result['managed_audio_bytes']))}")
            print(f"当前可清理: {_format_bytes(int(result['eligible_bytes']))}")
            for item in result["managed_audio"]:
                state = "可清理" if item["eligible_for_cleanup"] else "保留"
                print(
                    f"- {state}: {item['audio_path']} "
                    f"({_format_bytes(int(item['size_bytes']))})；"
                    f"到期 {item['eligible_for_cleanup_at']}"
                )
        else:
            print(f"符合条件: {result['eligible_items']} 项，{_format_bytes(int(result['eligible_bytes']))}")
            print(f"实际删除: {result['deleted_items']} 项，{_format_bytes(int(result['deleted_bytes']))}")
        for warning in result.get("warnings", []):
            print(f"警告: {warning}")
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


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command == "cache-status":
            result = inspect_cache(settings.data_root / ".bili-summary-cache")
            _print_result(result, args.as_json)
            return 0
        if args.command == "cache-clean":
            cache_root = settings.data_root / ".bili-summary-cache"
            if args.execute and not args.yes:
                preview = clean_cache(cache_root, execute=False)
                if not sys.stdin.isatty():
                    raise ValueError("非交互清理必须同时使用 --execute 和 --yes")
                print(
                    f"将删除 {preview['eligible_items']} 项已到期临时音频，"
                    f"共 {_format_bytes(int(preview['eligible_bytes']))}。",
                    file=sys.stderr,
                )
                if _stderr_input("输入 clean 继续，其他内容取消: ").strip().lower() != "clean":
                    raise ValueError("用户取消；未删除缓存")
            result = clean_cache(cache_root, execute=args.execute)
            _print_result(result, args.as_json)
            return 0
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
            doctor["content_mode"] = settings.content_mode
            doctor["text_presets"] = sorted(settings.text_presets)
            doctor["deepseek_credentials_configured"] = bool(
                os.environ.get(settings.deepseek_api_key_env)
                or (
                    settings.deepseek_api_key_file
                    and settings.deepseek_api_key_file.is_file()
                )
            )
            doctor["aliyun_asr"] = {
                "region": "cn-beijing",
                "models": list(COMPARISON_MODELS),
                "workspace_configured": bool(settings.aliyun_asr_workspace_id),
                "credentials_configured": bool(
                    os.environ.get(settings.aliyun_asr_api_key_env)
                    or (
                        settings.aliyun_asr_api_key_file
                        and settings.aliyun_asr_api_key_file.is_file()
                    )
                ),
                "ten_minute_max_cost_cny": estimate_comparison_cost_cny(600),
            }
            doctor["local_asr"] = {
                "configured": bool(settings.local_asr_binary and settings.local_asr_model),
                "binary": str(settings.local_asr_binary) if settings.local_asr_binary else None,
                "binary_exists": bool(
                    settings.local_asr_binary and settings.local_asr_binary.is_file()
                ),
                "model": str(settings.local_asr_model) if settings.local_asr_model else None,
                "model_exists": bool(settings.local_asr_model and settings.local_asr_model.is_file()),
                "threads": settings.local_asr_threads,
            }
            doctor["long_chunk_minutes"] = {
                "target": settings.long_chunk_target_minutes,
                "max": settings.long_chunk_max_minutes,
                "deep_target": settings.deep_chunk_target_minutes,
                "deep_max": settings.deep_chunk_max_minutes,
            }
            _print_result(doctor, args.as_json)
            return 0
        if args.command == "check-aliyun-asr":
            client = AliyunTemporaryUploadClient(
                api_key=load_aliyun_asr_api_key(settings)
            )
            checks = [client.check_model_upload_access(model) for model in COMPARISON_MODELS]
            _print_result(
                {
                    "status": "aliyun_asr_preflight_complete",
                    "region": "cn-beijing",
                    "models": checks,
                    "notice": (
                        "本次只申请了短时上传策略，用于验证鉴权和模型绑定；"
                        "没有上传音频、提交转写或产生识别费用。实际调用权仍由首次转写确认。"
                    ),
                },
                args.as_json,
            )
            return 0
        if args.command == "compare-aliyun-asr":
            if not args.yes:
                if not sys.stdin.isatty():
                    raise ValueError("非交互转写比较必须增加 --yes")
                print(
                    "将把同一 WAV 分别上传到两个模型绑定的私有临时存储，"
                    "并提交两次转写；最坏费用受配置的 1 元门槛限制。",
                    file=sys.stderr,
                )
                if _stderr_input("输入 yes 继续，其他内容取消: ").strip().lower() != "yes":
                    raise ValueError("用户取消；未上传音频或提交转写")
            result = run_aliyun_asr_comparison(
                args.source,
                settings,
                output_dir=args.output,
                progress=lambda message: print(message, file=sys.stderr),
            )
            _print_result(result, args.as_json)
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
                content_mode = args.content_mode or settings.content_mode
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
                    content_mode=content_mode,
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
            if args.execute and args.prepare_audio_sample:
                raise ValueError("--execute 与 --prepare-audio-sample 不能同时使用")
            spec = parse_local_mp4(
                args.source,
                compute_hash=args.hash or args.prepare_audio_sample or args.execute,
            )
            if args.compare_deep and not args.long:
                raise ValueError("--compare-deep 只能与 --long 一起使用")
            requested_audit = (
                "deep" if args.compare_deep else (args.audit_level or settings.audit_level)
            )
            content_mode = args.content_mode or settings.content_mode
            if args.execute and not args.force:
                completed = completed_local_result(
                    spec,
                    settings,
                    subject=args.subject,
                    course=args.course,
                    title_override=args.title,
                    audit_level=requested_audit,
                    content_mode=content_mode,
                )
                if completed is not None:
                    _print_result(completed, args.as_json)
                    return 0
            media = None
            if args.probe or args.prepare_audio_sample:
                media = inspect_local_media(Path(spec.canonical))
                metadata = dict(spec.metadata)
                metadata["media_probe_status"] = "complete"
                metadata["media"] = media
                spec = replace(spec, metadata=metadata)
            result = _preview(spec, args, settings)
            if args.prepare_audio_sample:
                sample = prepare_transcription_sample(
                    Path(spec.canonical),
                    media=media,
                    source_sha256=spec.metadata["sha256"],
                    cache_root=settings.data_root / ".bili-summary-cache",
                    start_seconds=args.sample_start,
                    duration_seconds=args.sample_minutes * 60,
                )
                result["status"] = "audio_sample_ready"
                result["temporary_audio"] = sample
                result["notice"] = (
                    "只生成了可清理的转写比较样本；未调用语音或文本模型，原 MP4 未改动"
                )
            if args.execute:
                if media is None:
                    media = inspect_local_media(Path(spec.canonical))
                source_kind = media["text_source"]["kind"]
                if source_kind in {
                    "audio_transcription_required",
                    "unsupported_embedded_subtitle",
                }:
                    duration = float(media["probe"]["format"]["duration_seconds"])
                    costs = estimate_local_online_cost(duration)
                    mode = args.transcriber or settings.transcriber_mode
                    if mode in {"auto", "online"}:
                        print(
                            f"本地 MP4 无可用字幕；默认提交 qwen3-asr-flash-filetrans，"
                            f"按完整时长估算最多 {costs['qwen3-asr-flash-filetrans']:.6f} 元，"
                            f"本机提交门槛 {settings.cost_submission_limit_cny:.6f} 元。",
                            file=sys.stderr,
                        )
                    else:
                        print("本地 MP4 无可用字幕；本次只使用本地 CPU 转写。", file=sys.stderr)
                    if not args.yes:
                        if not sys.stdin.isatty():
                            raise ValueError("非交互本地执行必须增加 --yes")
                        if _stderr_input("输入 yes 继续，其他内容取消: ").strip().lower() != "yes":
                            raise ValueError("用户取消；未提取或上传完整音频")
                project_root = Path(__file__).resolve().parents[2]
                route_overrides = parse_route_overrides(args.route)
                result = run_local_file_pipeline(
                    spec,
                    settings,
                    subject=args.subject,
                    course=args.course,
                    title_override=args.title,
                    schemas_dir=project_root / "schemas",
                    compare_deep=args.compare_deep,
                    audit_level=requested_audit,
                    content_mode=content_mode,
                    force=args.force,
                    transcriber_mode=args.transcriber,
                    media=media,
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
            _print_result(result, args.as_json)
            return 0
        _print_result(_preview(spec, args, settings), args.as_json)
        return 0
    except (
        AliyunAsrError,
        BilibiliError,
        MediaError,
        TextBackendError,
        InputError,
        LocalAsrError,
        OSError,
        ValueError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
