from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import CodexError, CodexStudyBackend
from .bilibili import BilibiliClient, segments_to_srt
from .config import Settings
from .models import InputSpec, StudyOutputs
from .naming import build_archive_path
from .storage import atomic_write_json, atomic_write_text


OUTPUT_FILES = (
    "字幕.srt",
    "完整整理稿.md",
    "审校报告.md",
    "学习总结.md",
    "source.json",
)


def run_bilibili_pipeline(
    spec: InputSpec,
    settings: Settings,
    *,
    subject: str,
    course: str | None,
    title_override: str | None,
    schema_path: Path,
    force: bool = False,
    backend: CodexStudyBackend | None = None,
    client: BilibiliClient | None = None,
) -> dict[str, Any]:
    platform_client = client or BilibiliClient(settings.bilibili_cookie_file)
    transcript = platform_client.fetch_transcript(spec)
    title = title_override or _result_title(transcript.video, transcript.page)
    output_dir = build_archive_path(
        settings.data_root,
        subject=subject,
        course=course,
        title=title,
    )
    source_path = output_dir / "source.json"
    cached = _read_completed_source(source_path)
    if cached and not force and all((output_dir / name).is_file() for name in OUTPUT_FILES):
        return {
            "status": "already_complete",
            "output_dir": str(output_dir),
            "files": [str(output_dir / name) for name in OUTPUT_FILES],
            "task_id": cached.get("task_id"),
            "notice": "检测到同一成果目录已完成；未重复调用 Codex。使用 --force 才会覆盖文本成果",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    srt = segments_to_srt(transcript.segments)
    task_id = _task_id(transcript.video, transcript.page, transcript.subtitle)
    record: dict[str, Any] = {
        "schema_version": 1,
        "task_id": task_id,
        "status": "subtitle_ready",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "input": transcript.input_spec.to_dict(),
        "platform": {
            "name": "bilibili",
            "video": transcript.video,
            "page": transcript.page,
            "subtitle": transcript.subtitle,
            "authenticated_request": platform_client.authenticated,
        },
        "processing": {
            "text_backend": "codex-exec",
            "audit_level": settings.audit_level,
            "codex": None,
        },
        "outputs": {},
    }
    atomic_write_text(output_dir / "字幕.srt", srt)
    atomic_write_json(output_dir / "原始字幕.json", transcript.raw_subtitle)
    record["outputs"] = {
        "subtitle_srt": "字幕.srt",
        "raw_subtitle": "原始字幕.json",
    }
    atomic_write_json(source_path, record)

    text_backend = backend or CodexStudyBackend(schema_path, model=settings.codex_model)
    context = {
        "title": transcript.video.get("title"),
        "part_title": transcript.page.get("title"),
        "source_url": transcript.input_spec.canonical,
        "audit_level": settings.audit_level,
        "subtitle_language": transcript.subtitle.get("lan_doc") or transcript.subtitle.get("lan"),
    }
    try:
        outputs = text_backend.process(srt, context)
    except CodexError:
        record["status"] = "codex_failed"
        record["updated_at"] = _utc_now()
        atomic_write_json(source_path, record)
        raise

    _write_study_outputs(output_dir, outputs)
    record["status"] = "complete"
    record["updated_at"] = _utc_now()
    record["processing"]["codex"] = {
        "calls": outputs.call_count,
        "usage": outputs.usage,
        "elapsed_seconds": round(outputs.elapsed_seconds, 3),
        "sandbox": "read-only",
        "ephemeral": True,
        "warnings": list(outputs.warnings),
        **outputs.backend_metadata,
    }
    record["outputs"].update(
        {
            "clean_transcript": "完整整理稿.md",
            "audit_report": "审校报告.md",
            "study_summary": "学习总结.md",
        }
    )
    atomic_write_json(source_path, record)
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "files": [str(output_dir / name) for name in OUTPUT_FILES] + [str(output_dir / "原始字幕.json")],
        "task_id": task_id,
        "codex": record["processing"]["codex"],
    }


def _write_study_outputs(output_dir: Path, outputs: StudyOutputs) -> None:
    atomic_write_text(output_dir / "完整整理稿.md", outputs.clean_transcript_markdown)
    atomic_write_text(output_dir / "审校报告.md", outputs.audit_markdown)
    atomic_write_text(output_dir / "学习总结.md", outputs.summary_markdown)


def _result_title(video: dict[str, Any], page: dict[str, Any]) -> str:
    video_title = str(video.get("title") or "未命名视频")
    if int(video.get("page_count") or 1) > 1:
        return f"{video_title} - P{page.get('part_number')} {page.get('title') or ''}".strip()
    return video_title


def _task_id(video: dict[str, Any], page: dict[str, Any], subtitle: dict[str, Any]) -> str:
    identity = json.dumps(
        [video.get("bvid"), page.get("cid"), subtitle.get("id")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "bili-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _read_completed_source(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("status") == "complete" else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
