from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .aliyun_asr import (
    AliyunAsrError,
    PARAFORMER_V2_MODEL,
    PRICE_CNY_PER_SECOND,
    QWEN_FILETRANS_MODEL,
)
from .asr_evaluation import run_aliyun_file_transcription
from .config import Settings
from .evaluation import parse_srt
from .local_asr import LocalAsrError, run_whisper_cpp_transcription
from .long_pipeline import StructuredBackend, run_bilibili_long_pipeline
from .media import (
    MediaError,
    extract_embedded_subtitle,
    inspect_local_media,
    prepare_full_transcription_audio,
)
from .models import InputSpec, PlatformTranscript, TextExecutionPlan, TranscriptSegment
from .naming import build_archive_path


OnlineRunner = Callable[..., dict[str, Any]]
LocalRunner = Callable[..., dict[str, Any]]
AudioPreparer = Callable[..., dict[str, Any]]


class _PreparedTranscriptClient:
    authenticated = False

    def __init__(self, transcript: PlatformTranscript) -> None:
        self.transcript = transcript

    def fetch_transcript(self, _spec: InputSpec) -> PlatformTranscript:
        return self.transcript


def estimate_local_online_cost(duration_seconds: float) -> dict[str, float]:
    return {
        QWEN_FILETRANS_MODEL: round(
            duration_seconds * PRICE_CNY_PER_SECOND[QWEN_FILETRANS_MODEL], 6
        ),
        PARAFORMER_V2_MODEL: round(
            duration_seconds * PRICE_CNY_PER_SECOND[PARAFORMER_V2_MODEL], 6
        ),
    }


def run_local_file_pipeline(
    spec: InputSpec,
    settings: Settings,
    *,
    subject: str,
    course: str | None,
    title_override: str | None,
    schemas_dir: Path,
    compare_deep: bool,
    audit_level: str | None = None,
    content_mode: str | None = None,
    force: bool = False,
    transcriber_mode: str | None = None,
    media: dict[str, Any] | None = None,
    backend: StructuredBackend | None = None,
    backends: Mapping[str, StructuredBackend] | None = None,
    plan_selector: Callable[[dict[str, Any]], TextExecutionPlan] | None = None,
    progress: Callable[[str], None] | None = None,
    online_runner: OnlineRunner = run_aliyun_file_transcription,
    local_runner: LocalRunner = run_whisper_cpp_transcription,
    audio_preparer: AudioPreparer = prepare_full_transcription_audio,
) -> dict[str, Any]:
    if spec.source_type != "local_mp4":
        raise ValueError("本地流水线只接受 local_mp4 输入")
    source_sha256 = str(spec.metadata.get("sha256") or "")
    if len(source_sha256) != 64:
        raise MediaError("执行本地任务前必须完成原 MP4 的 SHA-256")
    mode = transcriber_mode or settings.transcriber_mode
    if mode not in {"auto", "online", "local"}:
        raise ValueError(f"无效的转写模式：{mode}")
    notify = progress or (lambda _message: None)
    media_path = Path(spec.canonical)
    inspected = media or inspect_local_media(media_path)
    cache_root = settings.data_root / ".bili-summary-cache"
    local_cache = cache_root / f"local-{source_sha256[:16]}"
    transcript, source_label, acquisition = _acquire_local_transcript(
        spec,
        settings,
        media=inspected,
        local_cache=local_cache,
        mode=mode,
        progress=notify,
        online_runner=online_runner,
        local_runner=local_runner,
        audio_preparer=audio_preparer,
    )
    title = title_override or spec.display_title
    local_record = {
        "name": "local_mp4",
        "path": str(media_path),
        "title": title,
        "source_sha256": source_sha256,
        "size_bytes": spec.metadata.get("size_bytes"),
        "modified_ns": spec.metadata.get("modified_ns"),
        "media": inspected,
        "subtitle_acquisition": acquisition,
    }
    return run_bilibili_long_pipeline(
        spec,
        settings,
        subject=subject,
        course=course,
        title_override=title,
        schemas_dir=schemas_dir,
        compare_deep=compare_deep,
        audit_level=audit_level,
        content_mode=content_mode or settings.content_mode,
        force=force,
        client=_PreparedTranscriptClient(transcript),  # type: ignore[arg-type]
        backend=backend,
        backends=backends,
        plan_selector=plan_selector,
        progress=progress,
        task_id_override=f"local-{source_sha256[:16]}",
        source_label=source_label,
        record_section=("local", local_record),
    )


def completed_local_result(
    spec: InputSpec,
    settings: Settings,
    *,
    subject: str,
    course: str | None,
    title_override: str | None,
    audit_level: str,
    content_mode: str = "lecture",
) -> dict[str, Any] | None:
    """Return a finished local result before media extraction or ASR work."""

    output_dir = build_archive_path(
        settings.data_root,
        subject=subject,
        course=course,
        title=title_override or spec.display_title,
    )
    required = ["字幕.srt", "完整整理稿.md", "审校报告.md", "学习总结.md", "source.json"]
    if audit_level == "deep":
        required.extend(["审校报告-deep.md", "审校对比.md"])
    source_path = output_dir / "source.json"
    try:
        record = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    expected = f"local-{str(spec.metadata.get('sha256') or '')[:16]}"
    recorded_mode = record.get("processing", {}).get("content_mode", "lecture")
    if (
        record.get("status") != "complete"
        or record.get("task_id") != expected
        or recorded_mode != content_mode
        or not all((output_dir / name).is_file() for name in required)
    ):
        return None
    return {
        "status": "already_complete",
        "output_dir": str(output_dir),
        "files": [str(output_dir / name) for name in required],
        "task_id": expected,
        "content_mode": content_mode,
        "notice": "本地任务已经按相同内容模式完成；未提取音频、调用转写或文本模型",
    }


def _acquire_local_transcript(
    spec: InputSpec,
    settings: Settings,
    *,
    media: dict[str, Any],
    local_cache: Path,
    mode: str,
    progress: Callable[[str], None],
    online_runner: OnlineRunner,
    local_runner: LocalRunner,
    audio_preparer: AudioPreparer,
) -> tuple[PlatformTranscript, str, dict[str, Any]]:
    source_sha256 = str(spec.metadata["sha256"])
    source = media.get("text_source") if isinstance(media.get("text_source"), dict) else {}
    kind = source.get("kind")
    if kind == "external_srt":
        subtitle_path = Path(str(source["path"]))
        segments = _read_srt(subtitle_path)
        acquisition = {
            "kind": "external_srt",
            "path": str(subtitle_path),
            "sha256": _sha256(subtitle_path),
            "backup_policy": "source_file_managed_with_video",
        }
        return (
            _platform_transcript(spec, media, segments, acquisition, acquisition),
            "本地 MP4 同名外置字幕",
            acquisition,
        )
    if kind == "embedded_subtitle":
        extracted = extract_embedded_subtitle(
            Path(spec.canonical),
            stream_index=int(source["stream_index"]),
            source_sha256=source_sha256,
            cache_root=local_cache.parent,
        )
        subtitle_path = Path(str(extracted["path"]))
        segments = _read_srt(subtitle_path)
        acquisition = {
            "kind": "embedded_subtitle",
            "stream_index": source["stream_index"],
            "codec": source.get("codec"),
            "cache": extracted,
        }
        return (
            _platform_transcript(spec, media, segments, acquisition, acquisition),
            "本地 MP4 内嵌字幕",
            acquisition,
        )
    if kind == "unavailable_no_audio":
        raise MediaError("本地 MP4 没有可用字幕或音轨")
    if kind not in {"audio_transcription_required", "unsupported_embedded_subtitle"}:
        raise MediaError("本地媒体检查没有给出可处理的文本来源")

    duration = _media_duration(media)
    costs = estimate_local_online_cost(duration)
    if mode in {"auto", "online"} and costs[QWEN_FILETRANS_MODEL] > settings.cost_submission_limit_cny:
        if mode == "online" or settings.local_asr_binary is None or settings.local_asr_model is None:
            raise AliyunAsrError(
                f"{QWEN_FILETRANS_MODEL} 预计费用 {costs[QWEN_FILETRANS_MODEL]:.6f} 元超过提交门槛 "
                f"{settings.cost_submission_limit_cny:.6f} 元；尚未提取或上传完整音频",
                code="cost_limit_exceeded",
            )
        progress("Qwen3 超过在线费用门槛，改用已配置的本地 CPU 回退")
        mode = "local"
    audio = audio_preparer(
        Path(spec.canonical),
        media=media,
        source_sha256=source_sha256,
        cache_root=local_cache.parent,
    )
    audio_path = Path(str(audio["audio"]["path"]))
    errors: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    if mode in {"auto", "online"}:
        for model in (QWEN_FILETRANS_MODEL, PARAFORMER_V2_MODEL):
            try:
                result = online_runner(
                    audio_path,
                    settings,
                    model=model,
                    duration_seconds=duration,
                    cache_root=local_cache / "asr",
                    progress=progress,
                )
                break
            except AliyunAsrError as exc:
                errors.append(_safe_error(exc))
                if exc.code == "cost_limit_exceeded":
                    raise
                if model == QWEN_FILETRANS_MODEL:
                    progress(f"Qwen3 转写失败（{exc.code}），尝试 Paraformer v2")
        if result is None and mode == "online":
            raise AliyunAsrError(
                "Qwen3 与 Paraformer v2 均未完成转写",
                code="online_fallback_exhausted",
                retryable=any(error["retryable"] for error in errors),
            )
    if result is None:
        try:
            result = local_runner(
                audio_path,
                settings,
                cache_root=local_cache / "asr" / "local",
                source_identity=source_sha256,
                progress=progress,
            )
        except LocalAsrError as exc:
            errors.append(_safe_error(exc))
            raise LocalAsrError(
                "在线转写未完成，且本地 CPU 回退不可用；详情已标准化记录",
                code="all_transcribers_failed",
                retryable=any(error["retryable"] for error in errors),
            ) from exc
    acquisition = {
        "kind": "audio_transcription",
        "selected_mode": mode,
        "provider": result["provider"],
        "model": result["model"],
        "task_id": result.get("task_id"),
        "reused": result.get("reused", False),
        "usage": result.get("usage") or {},
        "provider_timing": result.get("provider_timing") or {},
        "estimated_max_cost_cny": result.get("estimated_max_cost_cny", 0.0),
        "attempt_errors": errors,
        "audio_cache": audio,
        "cache_backup_policy": "do_not_backup",
    }
    return (
        _platform_transcript(
            spec,
            media,
            tuple(result["segments"]),
            result["raw_transcript"],
            acquisition,
        ),
        f"本地 MP4 音频转写（{result['model']}）",
        acquisition,
    )


def _platform_transcript(
    spec: InputSpec,
    media: dict[str, Any],
    segments: tuple[TranscriptSegment, ...],
    raw: dict[str, Any],
    acquisition: dict[str, Any],
) -> PlatformTranscript:
    duration = _media_duration(media)
    return PlatformTranscript(
        input_spec=spec,
        video={
            "title": spec.display_title,
            "duration_seconds": duration,
            "page_count": 1,
            "source_sha256": spec.metadata.get("sha256"),
        },
        page={"part_number": 1, "title": spec.display_title, "duration_seconds": duration},
        subtitle={
            "source": acquisition["kind"],
            "lan": "zh" if acquisition["kind"] == "audio_transcription" else None,
            "provider": acquisition.get("provider"),
            "model": acquisition.get("model"),
        },
        raw_subtitle=raw,
        segments=segments,
    )


def _read_srt(path: Path) -> tuple[TranscriptSegment, ...]:
    try:
        value = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MediaError(f"字幕不是 UTF-8 编码：{path}") from exc
    try:
        return parse_srt(value)
    except ValueError as exc:
        raise MediaError(f"字幕没有有效句段：{path}") from exc


def _media_duration(media: dict[str, Any]) -> float:
    probe = media.get("probe") if isinstance(media.get("probe"), dict) else {}
    media_format = probe.get("format") if isinstance(probe.get("format"), dict) else {}
    try:
        duration = float(media_format["duration_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError("本地媒体缺少有效时长") from exc
    if duration <= 0:
        raise MediaError("本地媒体时长必须大于 0")
    return duration


def _safe_error(exc: Any) -> dict[str, Any]:
    return {
        "provider": str(getattr(exc, "provider", "unknown")),
        "code": str(getattr(exc, "code", "unknown")),
        "retryable": bool(getattr(exc, "retryable", False)),
        "message": str(exc)[-1000:],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
