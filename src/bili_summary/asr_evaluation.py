from __future__ import annotations

import hashlib
import json
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .aliyun_asr import (
    AliyunAsrClient,
    AliyunAsrError,
    AliyunTemporaryUploadClient,
    COMPARISON_MODELS,
    PRICE_CNY_PER_SECOND,
    estimate_comparison_cost_cny,
    load_aliyun_asr_api_key,
)
from .bilibili import segments_to_srt
from .config import Settings
from .models import TranscriptSegment
from .storage import atomic_write_json, atomic_write_text


def run_aliyun_asr_comparison(
    source: Path,
    settings: Settings,
    *,
    output_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
    client_factory: Callable[[str], AliyunAsrClient] | None = None,
    uploader_factory: Callable[[str], AliyunTemporaryUploadClient] | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    audio = inspect_comparison_wav(source)
    source_sha256 = _sha256(source)
    costs = estimate_comparison_cost_cny(audio["duration_seconds"])
    if costs["total"] > settings.cost_submission_limit_cny:
        raise AliyunAsrError(
            f"两模型预计费用 {costs['total']:.6f} 元超过提交门槛 "
            f"{settings.cost_submission_limit_cny:.6f} 元",
            code="cost_limit_exceeded",
        )
    root = output_dir or (
        settings.data_root
        / "模型对比"
        / f"{source.stem}-语音转写-{source_sha256[:12]}"
    )
    root.mkdir(parents=True, exist_ok=True)
    key = load_aliyun_asr_api_key(settings)
    client = (
        client_factory(key)
        if client_factory
        else AliyunAsrClient(
            api_key=key,
            workspace_id=settings.aliyun_asr_workspace_id,
        )
    )
    uploader = (
        uploader_factory(key)
        if uploader_factory
        else AliyunTemporaryUploadClient(api_key=key, timeout_seconds=180)
    )
    results = []
    for model in COMPARISON_MODELS:
        results.append(
            _run_model(
                model=model,
                source=source,
                source_sha256=source_sha256,
                root=root,
                client=client,
                uploader=uploader,
                progress=progress,
            )
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            **audio,
        },
        "models": results,
        "estimated_max_cost_cny": costs,
        "actual_billing": "verify_in_console",
        "enhancements": {
            "hotwords": False,
            "prompt_context": False,
            "disfluency_removal": False,
            "speaker_diarization": False,
        },
        "data_policy": {
            "uploaded_audio": "private_model_scoped_temporary_storage",
            "temporary_url_hours": 48,
            "source_mp4_uploaded": False,
            "raw_results_backup": True,
            "cache_backup": False,
        },
    }
    atomic_write_json(root / "对比记录.json", manifest)
    return {"status": "complete", "output_dir": str(root), **manifest}


def run_aliyun_file_transcription(
    source: Path,
    settings: Settings,
    *,
    model: str,
    duration_seconds: float,
    cache_root: Path,
    progress: Callable[[str], None] | None = None,
    client_factory: Callable[[str], AliyunAsrClient] | None = None,
    uploader_factory: Callable[[str], AliyunTemporaryUploadClient] | None = None,
) -> dict[str, Any]:
    """Run one recoverable production file-transcription task from a prepared WAV."""

    if model not in COMPARISON_MODELS:
        raise AliyunAsrError(f"不支持的阿里云 ASR 模型：{model}", code="unsupported_model")
    if not source.is_file() or source.stat().st_size <= 44:
        raise AliyunAsrError(f"完整转写音频不存在或为空：{source}", code="missing_source")
    if duration_seconds <= 0 or duration_seconds > 12 * 60 * 60:
        raise AliyunAsrError("完整转写音频时长必须大于 0 且不超过 12 小时", code="invalid_audio_duration")
    estimated_cost = round(duration_seconds * PRICE_CNY_PER_SECOND[model], 6)
    if estimated_cost > settings.cost_submission_limit_cny:
        raise AliyunAsrError(
            f"{model} 预计费用 {estimated_cost:.6f} 元超过提交门槛 "
            f"{settings.cost_submission_limit_cny:.6f} 元",
            code="cost_limit_exceeded",
        )
    source_sha256 = _sha256(source)
    key = load_aliyun_asr_api_key(settings)
    client = (
        client_factory(key)
        if client_factory
        else AliyunAsrClient(
            api_key=key,
            workspace_id=settings.aliyun_asr_workspace_id,
        )
    )
    uploader = (
        uploader_factory(key)
        if uploader_factory
        else AliyunTemporaryUploadClient(api_key=key, timeout_seconds=1800)
    )
    result = _run_model(
        model=model,
        source=source,
        source_sha256=source_sha256,
        root=cache_root,
        client=client,
        uploader=uploader,
        progress=progress,
    )
    raw_path = cache_root / model / "原始响应.json"
    raw = _read_json(raw_path)
    if raw is None:
        raise AliyunAsrError("阿里云转写缓存缺少原始结果", code="invalid_transcript")
    segments = parse_aliyun_transcript(raw)
    return {
        "provider": "aliyun_bailian",
        "model": model,
        "status": "complete",
        "reused": result["reused"],
        "task_id": result["task_id"],
        "segments": segments,
        "raw_transcript": raw,
        "usage": result["usage"],
        "provider_timing": result.get("provider_timing") or {},
        "estimated_max_cost_cny": estimated_cost,
        "files": result["files"],
        "cache_backup_policy": "do_not_backup",
    }


def inspect_comparison_wav(source: Path) -> dict[str, Any]:
    if not source.is_file():
        raise AliyunAsrError(f"转写样本不存在：{source}", code="missing_source")
    try:
        with wave.open(str(source), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frame_count = stream.getnframes()
    except (wave.Error, EOFError) as exc:
        raise AliyunAsrError("转写样本不是有效 WAV", code="invalid_audio") from exc
    if channels != 1 or sample_rate != 16_000 or sample_width != 2:
        raise AliyunAsrError(
            "转写比较样本必须是单声道、16 kHz、16-bit PCM WAV",
            code="invalid_audio_profile",
        )
    duration = frame_count / sample_rate
    if not 300 <= duration <= 601:
        raise AliyunAsrError("转写比较样本必须为 5～10 分钟", code="invalid_audio_duration")
    return {
        "size_bytes": source.stat().st_size,
        "duration_seconds": round(duration, 3),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
    }


def parse_aliyun_transcript(payload: dict[str, Any]) -> tuple[TranscriptSegment, ...]:
    transcripts = payload.get("transcripts")
    if not isinstance(transcripts, list):
        raise AliyunAsrError("阿里云原始结果缺少 transcripts", code="invalid_transcript")
    segments = []
    for transcript in transcripts:
        sentences = transcript.get("sentences") if isinstance(transcript, dict) else None
        if not isinstance(sentences, list):
            continue
        for sentence in sentences:
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text") or "").strip()
            try:
                start_ms = int(sentence["begin_time"])
                end_ms = int(sentence["end_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if text and start_ms >= 0 and end_ms > start_ms:
                segments.append(TranscriptSegment(start_ms, end_ms, text))
    segments.sort(key=lambda item: (item.start_ms, item.end_ms))
    if not segments:
        raise AliyunAsrError("阿里云原始结果没有有效句级时间戳", code="empty_transcript")
    return tuple(segments)


def _run_model(
    *,
    model: str,
    source: Path,
    source_sha256: str,
    root: Path,
    client: AliyunAsrClient,
    uploader: AliyunTemporaryUploadClient,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    model_dir = root / model
    raw_path = model_dir / "原始响应.json"
    srt_path = model_dir / "字幕.srt"
    cache_path = root / ".comparison-cache" / f"{model}.json"
    cached = _read_json(cache_path)
    if (
        isinstance(cached, dict)
        and cached.get("source_sha256") == source_sha256
        and cached.get("status") == "complete"
        and raw_path.is_file()
        and srt_path.is_file()
    ):
        if progress:
            progress(f"{model}：复用已完成结果")
        return {
            "model": model,
            "status": "complete",
            "reused": True,
            "task_id": str(cached.get("task_id") or ""),
            "elapsed_seconds": float(cached.get("elapsed_seconds") or 0),
            "segment_count": int(cached.get("segment_count") or 0),
            "usage": cached.get("usage") or {},
            "provider_timing": cached.get("provider_timing") or {},
            "files": [str(raw_path), str(srt_path)],
        }

    started = time.monotonic()
    task_id = (
        str(cached.get("task_id"))
        if isinstance(cached, dict)
        and cached.get("source_sha256") == source_sha256
        and cached.get("task_id")
        else ""
    )
    if not task_id:
        upload = None
        if (
            isinstance(cached, dict)
            and cached.get("source_sha256") == source_sha256
            and cached.get("status") == "uploaded"
            and float(cached.get("upload_expires_at_epoch") or 0) > time.time()
            and isinstance(cached.get("temporary_upload"), dict)
        ):
            upload = cached["temporary_upload"]
            if progress:
                progress(f"{model}：复用已上传的模型绑定临时样本")
        if upload is None:
            if progress:
                progress(f"{model}：上传模型绑定的私有临时样本")
            upload = uploader.upload(model, source)
            atomic_write_json(
                cache_path,
                {
                    "schema_version": 1,
                    "source_sha256": source_sha256,
                    "model": model,
                    "status": "uploaded",
                    "temporary_upload": upload,
                    # 官方临时对象有效 48 小时；提前一小时停止复用，给任务留出时间。
                    "upload_expires_at_epoch": time.time() + 47 * 60 * 60,
                    "backup_policy": "do_not_backup",
                },
            )
        if progress:
            progress(f"{model}：提交异步转写")
        submitted = client.submit(model=model, file_url=str(upload["file_url"]))
        task_id = submitted["task_id"]
        atomic_write_json(
            cache_path,
            {
                "schema_version": 1,
                "source_sha256": source_sha256,
                "model": model,
                "status": "submitted",
                "task_id": task_id,
                "request_id": submitted["request_id"],
                "temporary_upload": upload,
                "backup_policy": "do_not_backup",
            },
        )
    response = client.wait_for_completion(
        task_id,
        progress=progress,
    )
    raw = client.download_transcription(response)
    segments = parse_aliyun_transcript(raw)
    atomic_write_json(raw_path, raw)
    atomic_write_text(srt_path, segments_to_srt(segments))
    elapsed = time.monotonic() - started
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    provider_timing = _provider_timing(response)
    complete = {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "model": model,
        "status": "complete",
        "task_id": task_id,
        "elapsed_seconds": elapsed,
        "segment_count": len(segments),
        "usage": usage,
        "provider_timing": provider_timing,
        "backup_policy": "do_not_backup",
    }
    atomic_write_json(cache_path, complete)
    return {
        "model": model,
        "status": "complete",
        "reused": False,
        "task_id": task_id,
        "elapsed_seconds": elapsed,
        "segment_count": len(segments),
        "usage": usage,
        "provider_timing": provider_timing,
        "files": [str(raw_path), str(srt_path)],
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provider_timing(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output")
    if not isinstance(output, dict):
        return {}
    timing = {
        key: output[key]
        for key in ("submit_time", "scheduled_time", "end_time")
        if isinstance(output.get(key), str)
    }
    try:
        submitted = datetime.strptime(timing["submit_time"], "%Y-%m-%d %H:%M:%S.%f")
        ended = datetime.strptime(timing["end_time"], "%Y-%m-%d %H:%M:%S.%f")
    except (KeyError, ValueError):
        return timing
    timing["service_elapsed_seconds"] = round((ended - submitted).total_seconds(), 3)
    return timing
