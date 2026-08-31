from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .evaluation import parse_srt
from .storage import atomic_write_json


LOCAL_ASR_CACHE_VERSION = 1


class LocalAsrError(RuntimeError):
    """A safe local CPU transcription failure."""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = "whisper_cpp"
        self.code = code
        self.retryable = retryable


def run_whisper_cpp_transcription(
    source: Path,
    settings: Settings,
    *,
    cache_root: Path,
    source_identity: str,
    progress: Callable[[str], None] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    binary = settings.local_asr_binary
    model = settings.local_asr_model
    if binary is None or model is None:
        raise LocalAsrError(
            "本地 CPU 回退尚未配置 whisper.cpp binary 和 model",
            code="not_configured",
        )
    if not binary.is_file() or not model.is_file():
        raise LocalAsrError(
            "本地 CPU 回退的 whisper.cpp 程序或模型文件不存在",
            code="missing_runtime",
        )
    if not source.is_file() or source.stat().st_size <= 44:
        raise LocalAsrError("本地转写音频不存在或为空", code="missing_source")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "cache_version": LOCAL_ASR_CACHE_VERSION,
                "source_identity": source_identity,
                "binary": _file_identity(binary),
                "model": _file_identity(model),
                "language": "zh",
                "threads": settings.local_asr_threads,
                "cpu_only": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    output_prefix = cache_root / "whisper-cpp"
    srt_path = output_prefix.with_suffix(".srt")
    metadata_path = cache_root / "whisper-cpp.json"
    cached = _read_json(metadata_path)
    if (
        cached
        and cached.get("fingerprint") == fingerprint
        and srt_path.is_file()
        and srt_path.stat().st_size > 0
    ):
        segments = parse_srt(srt_path.read_text(encoding="utf-8"))
        return {
            "provider": "local",
            "model": "whisper.cpp-small",
            "status": "complete",
            "reused": True,
            "segments": segments,
            "raw_transcript": {
                "engine": "whisper.cpp",
                "model_path": str(model),
                "srt": srt_path.read_text(encoding="utf-8"),
            },
            "usage": {},
            "elapsed_seconds": float(cached.get("elapsed_seconds") or 0),
            "estimated_max_cost_cny": 0.0,
            "files": [str(srt_path)],
            "cache_backup_policy": "do_not_backup",
        }

    if progress:
        progress("whisper.cpp：开始纯 CPU 转写")
    command = [
        str(binary),
        "-m",
        str(model),
        "-f",
        str(source),
        "-l",
        "zh",
        "-t",
        str(settings.local_asr_threads),
        "-ng",
        "-osrt",
        "-of",
        str(output_prefix),
        "-np",
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
        raise LocalAsrError("无法启动 whisper.cpp", code="missing_runtime") from exc
    except subprocess.TimeoutExpired as exc:
        raise LocalAsrError(
            f"whisper.cpp 转写超过 {timeout_seconds} 秒",
            code="timeout",
            retryable=True,
        ) from exc
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:] or "没有错误详情"
        raise LocalAsrError(
            f"whisper.cpp 转写失败（退出码 {completed.returncode}）：{detail}",
            code="process_failed",
            retryable=False,
        )
    if not srt_path.is_file() or srt_path.stat().st_size == 0:
        raise LocalAsrError("whisper.cpp 没有生成有效 SRT", code="empty_transcript")
    srt = srt_path.read_text(encoding="utf-8")
    segments = parse_srt(srt)
    atomic_write_json(
        metadata_path,
        {
            "cache_version": LOCAL_ASR_CACHE_VERSION,
            "fingerprint": fingerprint,
            "provider": "local",
            "model": "whisper.cpp-small",
            "elapsed_seconds": elapsed,
            "segment_count": len(segments),
            "backup_policy": "do_not_backup",
        },
    )
    return {
        "provider": "local",
        "model": "whisper.cpp-small",
        "status": "complete",
        "reused": False,
        "segments": segments,
        "raw_transcript": {
            "engine": "whisper.cpp",
            "model_path": str(model),
            "srt": srt,
        },
        "usage": {},
        "elapsed_seconds": elapsed,
        "estimated_max_cost_cny": 0.0,
        "files": [str(srt_path)],
        "cache_backup_policy": "do_not_backup",
    }


def _file_identity(path: Path) -> dict[str, Any]:
    status = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": status.st_size,
        "modified_ns": status.st_mtime_ns,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
