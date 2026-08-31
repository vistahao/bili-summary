from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .storage import atomic_write_json


TEXT_SUBTITLE_CODECS = {
    "ass",
    "mov_text",
    "ssa",
    "subrip",
    "text",
    "webvtt",
}

SAMPLE_RETENTION_DAYS = 5
SAMPLE_SCHEMA_VERSION = 1
FULL_AUDIO_SCHEMA_VERSION = 1
EMBEDDED_SUBTITLE_SCHEMA_VERSION = 1


class MediaError(RuntimeError):
    """Raised when local media cannot be inspected safely."""


def find_external_srt(media_path: Path) -> Path | None:
    matches = sorted(
        (
            candidate
            for candidate in media_path.parent.iterdir()
            if candidate.is_file()
            and candidate.stem.casefold() == media_path.stem.casefold()
            and candidate.suffix.casefold() == ".srt"
        ),
        key=lambda path: path.name,
    )
    if len(matches) > 1:
        names = "、".join(path.name for path in matches)
        raise MediaError(f"发现多个同名 SRT，无法安全自动选择：{names}")
    return matches[0] if matches else None


def probe_media(
    media_path: Path,
    *,
    ffprobe_command: str = "ffprobe",
    timeout_seconds: int = 60,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    command = [
        ffprobe_command,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        completed = runner(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise MediaError("没有找到 ffprobe；请先安装 Ubuntu ffmpeg 软件包") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffprobe 检查超过 {timeout_seconds} 秒") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:] or "没有错误详情"
        raise MediaError(f"ffprobe 检查失败（退出码 {completed.returncode}）：{detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe 返回内容不是有效 JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("streams"), list):
        raise MediaError("ffprobe 返回内容缺少 streams 数组")

    streams = [_normalize_stream(stream) for stream in payload["streams"]]
    media_format = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    return {
        "tool": "ffprobe",
        "read_only": True,
        "format": {
            "name": _optional_text(media_format.get("format_name")),
            "duration_seconds": _optional_float(media_format.get("duration")),
            "size_bytes": _optional_int(media_format.get("size")),
            "bit_rate": _optional_int(media_format.get("bit_rate")),
        },
        "streams": streams,
        "stream_counts": {
            kind: sum(stream["type"] == kind for stream in streams)
            for kind in ("video", "audio", "subtitle")
        },
    }


def inspect_local_media(
    media_path: Path,
    *,
    ffprobe_command: str = "ffprobe",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    external_srt = find_external_srt(media_path)
    probe = probe_media(media_path, ffprobe_command=ffprobe_command, runner=runner)
    subtitles = [stream for stream in probe["streams"] if stream["type"] == "subtitle"]
    audio_streams = [stream for stream in probe["streams"] if stream["type"] == "audio"]
    extractable = [stream for stream in subtitles if stream["can_extract_srt"]]
    if external_srt is not None:
        text_source = {"kind": "external_srt", "path": str(external_srt)}
    elif extractable:
        text_source = {
            "kind": "embedded_subtitle",
            "stream_index": extractable[0]["index"],
            "codec": extractable[0]["codec"],
        }
    elif subtitles:
        text_source = {
            "kind": "unsupported_embedded_subtitle",
            "stream_indexes": [stream["index"] for stream in subtitles],
        }
    elif not audio_streams:
        text_source = {"kind": "unavailable_no_audio"}
    else:
        text_source = {"kind": "audio_transcription_required"}
    return {
        "external_srt": str(external_srt) if external_srt else None,
        "probe": probe,
        "text_source": text_source,
    }


def prepare_transcription_sample(
    media_path: Path,
    *,
    media: dict[str, Any],
    source_sha256: str,
    cache_root: Path,
    start_seconds: float = 0.0,
    duration_seconds: int = 600,
    retention_days: int = SAMPLE_RETENTION_DAYS,
    ffmpeg_command: str = "ffmpeg",
    timeout_seconds: int = 900,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create or reuse a small, lossless speech sample without changing the source MP4."""

    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256.casefold()
    ):
        raise MediaError("生成音频样本前必须完成有效的 SHA-256")
    if start_seconds < 0:
        raise MediaError("音频样本起点不能小于 0 秒")
    if not 300 <= duration_seconds <= 600:
        raise MediaError("音频样本时长必须在 5 到 10 分钟之间")
    if retention_days <= 0:
        raise MediaError("临时音频保留天数必须大于 0")

    text_source = media.get("text_source") if isinstance(media, dict) else None
    source_kind = text_source.get("kind") if isinstance(text_source, dict) else None
    if source_kind in {"external_srt", "embedded_subtitle"}:
        raise MediaError("已有可用字幕，不应提取转写音频")
    if source_kind == "unavailable_no_audio":
        raise MediaError("媒体没有可用字幕或音轨，无法准备转写样本")
    if source_kind not in {
        "audio_transcription_required",
        "unsupported_embedded_subtitle",
    }:
        raise MediaError("媒体检查结果不完整，无法安全准备转写样本")

    probe = media.get("probe") if isinstance(media.get("probe"), dict) else {}
    media_format = probe.get("format") if isinstance(probe.get("format"), dict) else {}
    media_duration = _optional_float(media_format.get("duration_seconds"))
    if media_duration is not None and start_seconds >= media_duration:
        raise MediaError("音频样本起点超出媒体时长")
    effective_duration = (
        min(float(duration_seconds), media_duration - start_seconds)
        if media_duration is not None
        else float(duration_seconds)
    )
    if effective_duration <= 0:
        raise MediaError("没有可提取的音频时长")

    profile = {
        "start_seconds": float(start_seconds),
        "requested_duration_seconds": duration_seconds,
        "sample_rate_hz": 16_000,
        "channels": 1,
        "sample_format": "signed_16_bit_pcm",
        "container": "wav",
    }
    fingerprint_source = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "source_sha256": source_sha256.casefold(),
        "profile": profile,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sample_dir = cache_root / f"local-{source_sha256.casefold()[:16]}" / "media"
    sample_path = sample_dir / "transcription-sample.wav"
    metadata_path = sample_dir / "transcription-sample.json"
    current_time = now or datetime.now(timezone.utc)

    cached = _read_sample_metadata(metadata_path)
    if (
        cached
        and cached.get("fingerprint") == fingerprint
        and sample_path.is_file()
        and sample_path.stat().st_size > 0
    ):
        cached["last_successful_use_at"] = _format_utc(current_time)
        cached["eligible_for_cleanup_at"] = _format_utc(
            current_time + timedelta(days=retention_days)
        )
        cached["reused"] = True
        atomic_write_json(metadata_path, cached)
        return cached

    sample_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".transcription-sample.", suffix=".wav", dir=sample_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    command = [
        ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(media_path),
        "-t",
        f"{effective_duration:.3f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(temporary_path),
    ]
    completed: subprocess.CompletedProcess[str] | None = None
    started = time.monotonic()
    try:
        completed = runner(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise MediaError("没有找到 ffmpeg；无法准备转写音频样本") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffmpeg 提取音频超过 {timeout_seconds} 秒") from exc
    finally:
        if completed is None:
            temporary_path.unlink(missing_ok=True)
    elapsed_seconds = time.monotonic() - started
    if completed is None:  # Defensive guard for custom runners with invalid behavior.
        raise MediaError("ffmpeg 执行器没有返回结果")
    if completed.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        detail = completed.stderr.strip()[-1000:] or "没有错误详情"
        raise MediaError(f"ffmpeg 提取音频失败（退出码 {completed.returncode}）：{detail}")
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise MediaError("ffmpeg 没有生成有效的音频样本")
    os.replace(temporary_path, sample_path)

    metadata = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "kind": "transcription_sample",
        "fingerprint": fingerprint,
        "backup_policy": "do_not_backup",
        "source": {
            "path": str(media_path),
            "sha256": source_sha256.casefold(),
        },
        "audio": {
            "path": str(sample_path),
            "size_bytes": sample_path.stat().st_size,
            "effective_duration_seconds": effective_duration,
            **profile,
        },
        "created_at": _format_utc(current_time),
        "last_successful_use_at": _format_utc(current_time),
        "eligible_for_cleanup_at": _format_utc(
            current_time + timedelta(days=retention_days)
        ),
        "retention_days": retention_days,
        "cleanup_policy": "eligible_after_last_successful_use; explicit cleanup only",
        "elapsed_seconds": elapsed_seconds,
        "reused": False,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def prepare_full_transcription_audio(
    media_path: Path,
    *,
    media: dict[str, Any],
    source_sha256: str,
    cache_root: Path,
    retention_days: int = SAMPLE_RETENTION_DAYS,
    ffmpeg_command: str = "ffmpeg",
    timeout_seconds: int = 3600,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Extract or reuse the complete first audio track for ASR without copying video."""

    _validate_source_hash(source_sha256)
    if retention_days <= 0:
        raise MediaError("临时音频保留天数必须大于 0")
    _require_transcription_source(media)
    probe = media.get("probe") if isinstance(media.get("probe"), dict) else {}
    media_format = probe.get("format") if isinstance(probe.get("format"), dict) else {}
    duration = _optional_float(media_format.get("duration_seconds"))
    if duration is None or duration <= 0:
        raise MediaError("完整转写前必须取得有效媒体时长")
    if duration > 12 * 60 * 60:
        raise MediaError("完整音频超过 Qwen3 Filetrans 的 12 小时上限")

    profile = {
        "scope": "complete_first_audio_track",
        "duration_seconds": duration,
        "sample_rate_hz": 16_000,
        "channels": 1,
        "sample_format": "signed_16_bit_pcm",
        "container": "wav",
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema_version": FULL_AUDIO_SCHEMA_VERSION,
                "source_sha256": source_sha256.casefold(),
                "profile": profile,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    audio_dir = cache_root / f"local-{source_sha256.casefold()[:16]}" / "media"
    audio_path = audio_dir / "transcription-full.wav"
    metadata_path = audio_dir / "transcription-full.json"
    current_time = now or datetime.now(timezone.utc)
    cached = _read_sample_metadata(metadata_path)
    if (
        cached
        and cached.get("fingerprint") == fingerprint
        and audio_path.is_file()
        and audio_path.stat().st_size > 44
    ):
        cached["last_successful_use_at"] = _format_utc(current_time)
        cached["eligible_for_cleanup_at"] = _format_utc(
            current_time + timedelta(days=retention_days)
        )
        cached["reused"] = True
        atomic_write_json(metadata_path, cached)
        return cached

    audio_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".transcription-full.", suffix=".wav", dir=audio_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    command = [
        ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(media_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(temporary_path),
    ]
    started = time.monotonic()
    try:
        completed = runner(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        temporary_path.unlink(missing_ok=True)
        raise MediaError("没有找到 ffmpeg；无法准备完整转写音频") from exc
    except subprocess.TimeoutExpired as exc:
        temporary_path.unlink(missing_ok=True)
        raise MediaError(f"ffmpeg 提取完整音频超过 {timeout_seconds} 秒") from exc
    elapsed_seconds = time.monotonic() - started
    if completed.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        detail = completed.stderr.strip()[-1000:] or "没有错误详情"
        raise MediaError(f"ffmpeg 提取完整音频失败（退出码 {completed.returncode}）：{detail}")
    if not temporary_path.is_file() or temporary_path.stat().st_size <= 44:
        temporary_path.unlink(missing_ok=True)
        raise MediaError("ffmpeg 没有生成有效的完整转写音频")
    os.replace(temporary_path, audio_path)
    metadata = {
        "schema_version": FULL_AUDIO_SCHEMA_VERSION,
        "kind": "full_transcription_audio",
        "fingerprint": fingerprint,
        "backup_policy": "do_not_backup",
        "source": {"path": str(media_path), "sha256": source_sha256.casefold()},
        "audio": {
            "path": str(audio_path),
            "size_bytes": audio_path.stat().st_size,
            **profile,
        },
        "created_at": _format_utc(current_time),
        "last_successful_use_at": _format_utc(current_time),
        "eligible_for_cleanup_at": _format_utc(
            current_time + timedelta(days=retention_days)
        ),
        "retention_days": retention_days,
        "cleanup_policy": "eligible_after_last_successful_use; explicit cleanup only",
        "elapsed_seconds": elapsed_seconds,
        "reused": False,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def extract_embedded_subtitle(
    media_path: Path,
    *,
    stream_index: int,
    source_sha256: str,
    cache_root: Path,
    ffmpeg_command: str = "ffmpeg",
    timeout_seconds: int = 300,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Extract one text subtitle track to a recoverable cache file."""

    _validate_source_hash(source_sha256)
    if stream_index < 0:
        raise MediaError("内嵌字幕轨索引不能小于 0")
    subtitle_dir = cache_root / f"local-{source_sha256.casefold()[:16]}" / "media"
    subtitle_path = subtitle_dir / f"embedded-subtitle-{stream_index}.srt"
    metadata_path = subtitle_dir / f"embedded-subtitle-{stream_index}.json"
    fingerprint = hashlib.sha256(
        f"{EMBEDDED_SUBTITLE_SCHEMA_VERSION}\0{source_sha256.casefold()}\0{stream_index}".encode()
    ).hexdigest()
    cached = _read_sample_metadata(metadata_path)
    if (
        cached
        and cached.get("fingerprint") == fingerprint
        and subtitle_path.is_file()
        and subtitle_path.stat().st_size > 0
    ):
        cached["reused"] = True
        atomic_write_json(metadata_path, cached)
        return cached

    subtitle_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".embedded-subtitle.", suffix=".srt", dir=subtitle_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    command = [
        ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(media_path),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-an",
        "-c:s",
        "srt",
        "-f",
        "srt",
        str(temporary_path),
    ]
    try:
        completed = runner(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        temporary_path.unlink(missing_ok=True)
        raise MediaError("没有找到 ffmpeg；无法提取内嵌字幕") from exc
    except subprocess.TimeoutExpired as exc:
        temporary_path.unlink(missing_ok=True)
        raise MediaError(f"ffmpeg 提取内嵌字幕超过 {timeout_seconds} 秒") from exc
    if completed.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        detail = completed.stderr.strip()[-1000:] or "没有错误详情"
        raise MediaError(f"ffmpeg 提取内嵌字幕失败（退出码 {completed.returncode}）：{detail}")
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise MediaError("内嵌字幕轨没有生成有效 SRT")
    os.replace(temporary_path, subtitle_path)
    metadata = {
        "schema_version": EMBEDDED_SUBTITLE_SCHEMA_VERSION,
        "kind": "embedded_subtitle",
        "fingerprint": fingerprint,
        "path": str(subtitle_path),
        "stream_index": stream_index,
        "source_sha256": source_sha256.casefold(),
        "backup_policy": "do_not_backup",
        "reused": False,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def _validate_source_hash(source_sha256: str) -> None:
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256.casefold()
    ):
        raise MediaError("媒体缓存前必须完成有效的 SHA-256")


def _require_transcription_source(media: dict[str, Any]) -> None:
    text_source = media.get("text_source") if isinstance(media, dict) else None
    source_kind = text_source.get("kind") if isinstance(text_source, dict) else None
    if source_kind in {"external_srt", "embedded_subtitle"}:
        raise MediaError("已有可用字幕，不应提取完整转写音频")
    if source_kind == "unavailable_no_audio":
        raise MediaError("媒体没有可用字幕或音轨，无法准备完整转写音频")
    if source_kind not in {
        "audio_transcription_required",
        "unsupported_embedded_subtitle",
    }:
        raise MediaError("媒体检查结果不完整，无法安全准备完整转写音频")


def _read_sample_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _format_utc(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_stream(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MediaError("ffprobe streams 包含无效项目")
    codec_type = _optional_text(value.get("codec_type")) or "unknown"
    codec_name = _optional_text(value.get("codec_name"))
    tags = value.get("tags") if isinstance(value.get("tags"), dict) else {}
    disposition = value.get("disposition") if isinstance(value.get("disposition"), dict) else {}
    return {
        "index": _optional_int(value.get("index")),
        "type": codec_type,
        "codec": codec_name,
        "language": _optional_text(tags.get("language")),
        "title": _optional_text(tags.get("title")),
        "duration_seconds": _optional_float(value.get("duration")),
        "default": bool(disposition.get("default", 0)),
        "can_extract_srt": codec_type == "subtitle" and codec_name in TEXT_SUBTITLE_CODECS,
    }


def _optional_text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
