from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANAGED_AUDIO_FILES = {
    "transcription-sample.json": "transcription-sample.wav",
    "transcription-full.json": "transcription-full.wav",
}


def inspect_cache(cache_root: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Return a read-only inventory without following symlinks."""

    current_time = _utc_now(now)
    root = cache_root.expanduser().resolve()
    total_bytes = _tree_size(root)
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    if root.is_dir():
        for metadata_path in sorted(root.glob("local-*/media/transcription-*.json")):
            entry, warning = _managed_audio_entry(root, metadata_path, current_time)
            if entry is not None:
                entries.append(entry)
            if warning is not None:
                warnings.append(warning)
    managed_bytes = sum(int(entry["size_bytes"]) for entry in entries)
    eligible_bytes = sum(
        int(entry["size_bytes"]) for entry in entries if entry["eligible_for_cleanup"]
    )
    return {
        "status": "cache_inventory",
        "cache_root": str(root),
        "scanned_at": _format_utc(current_time),
        "total_bytes": total_bytes,
        "managed_audio_bytes": managed_bytes,
        "eligible_bytes": eligible_bytes,
        "other_cache_bytes": max(0, total_bytes - managed_bytes),
        "managed_audio": entries,
        "warnings": warnings,
        "notice": (
            "这里只管理带生命周期元数据的临时转写音频；"
            "文本恢复缓存、ASR 原始响应和用户成果不会被列为自动清理目标。"
        ),
    }


def clean_cache(
    cache_root: Path,
    *,
    execute: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Preview or delete only eligible, validated managed audio files."""

    current_time = _utc_now(now)
    inventory = inspect_cache(cache_root, now=current_time)
    eligible = [
        entry for entry in inventory["managed_audio"] if entry["eligible_for_cleanup"]
    ]
    result: dict[str, Any] = {
        "status": "cache_cleanup_complete" if execute else "cache_cleanup_preview",
        "cache_root": inventory["cache_root"],
        "scanned_at": inventory["scanned_at"],
        "eligible_items": len(eligible),
        "eligible_bytes": sum(int(entry["size_bytes"]) for entry in eligible),
        "deleted_items": 0,
        "deleted_bytes": 0,
        "targets": eligible,
        "warnings": list(inventory["warnings"]),
    }
    if not execute:
        result["notice"] = "预览完成；未删除任何文件。使用 --execute 并确认后才会清理。"
        return result

    root = Path(str(inventory["cache_root"]))
    deleted: list[dict[str, Any]] = []
    for candidate in eligible:
        metadata_path = Path(str(candidate["metadata_path"]))
        refreshed, warning = _managed_audio_entry(root, metadata_path, current_time)
        if warning is not None:
            result["warnings"].append(warning)
        if refreshed is None or not refreshed["eligible_for_cleanup"]:
            continue
        audio_path = Path(str(refreshed["audio_path"]))
        try:
            audio_path.unlink()
        except OSError as exc:
            result["warnings"].append(f"音频清理失败：{audio_path}：{exc}")
            continue
        try:
            metadata_path.unlink()
        except OSError as exc:
            result["warnings"].append(
                f"音频已删除，但元数据清理失败：{metadata_path}：{exc}"
            )
        deleted.append(refreshed)

    result["deleted_items"] = len(deleted)
    result["deleted_bytes"] = sum(int(entry["size_bytes"]) for entry in deleted)
    result["deleted"] = deleted
    result["notice"] = (
        "只删除了达到5天期限且通过路径、文件名和元数据校验的临时音频及其元数据；"
        "没有删除成果、字幕、ASR 原始响应或文本恢复缓存。"
    )
    return result


def _managed_audio_entry(
    root: Path,
    metadata_path: Path,
    now: datetime,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        relative = metadata_path.relative_to(root)
    except ValueError:
        return None, f"忽略缓存根目录之外的元数据：{metadata_path}"
    parts = relative.parts
    if (
        len(parts) != 3
        or not parts[0].startswith("local-")
        or parts[1] != "media"
        or parts[2] not in MANAGED_AUDIO_FILES
    ):
        return None, f"忽略不符合受管目录结构的元数据：{metadata_path}"
    if metadata_path.is_symlink() or not _is_regular_file(metadata_path):
        return None, f"忽略符号链接或非普通元数据文件：{metadata_path}"
    if metadata_path.resolve() != metadata_path.absolute():
        return None, f"忽略经过符号链接目录的元数据：{metadata_path}"
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"无法读取缓存元数据：{metadata_path}：{exc}"
    if not isinstance(value, dict):
        return None, f"缓存元数据不是对象：{metadata_path}"
    if value.get("backup_policy") != "do_not_backup":
        return None, f"缓存元数据没有 do_not_backup 标记：{metadata_path}"
    if value.get("cleanup_policy") != "eligible_after_last_successful_use; explicit cleanup only":
        return None, f"缓存元数据没有受支持的清理策略：{metadata_path}"
    audio = value.get("audio")
    if not isinstance(audio, dict):
        return None, f"缓存元数据缺少 audio 对象：{metadata_path}"
    expected_audio = metadata_path.with_name(MANAGED_AUDIO_FILES[metadata_path.name])
    recorded_path = audio.get("path")
    if (
        not isinstance(recorded_path, str)
        or Path(recorded_path).expanduser().resolve() != expected_audio.resolve()
    ):
        return None, f"缓存音频路径与受管文件名不一致：{metadata_path}"
    if expected_audio.is_symlink() or not _is_regular_file(expected_audio):
        return None, f"缓存音频不存在、不是普通文件或是符号链接：{expected_audio}"
    if expected_audio.resolve() != expected_audio.absolute():
        return None, f"忽略经过符号链接目录的缓存音频：{expected_audio}"
    try:
        eligible_at = _parse_utc(value.get("eligible_for_cleanup_at"))
    except ValueError as exc:
        return None, f"缓存清理时间无效：{metadata_path}：{exc}"
    actual_size = expected_audio.stat(follow_symlinks=False).st_size
    return (
        {
            "kind": value.get("kind"),
            "audio_path": str(expected_audio),
            "metadata_path": str(metadata_path),
            "size_bytes": actual_size,
            "last_successful_use_at": value.get("last_successful_use_at"),
            "eligible_for_cleanup_at": _format_utc(eligible_at),
            "eligible_for_cleanup": now >= eligible_at,
        },
        None,
    )


def _tree_size(root: Path) -> int:
    if not root.is_dir():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            details = path.lstat()
        except OSError:
            continue
        if stat.S_ISREG(details.st_mode) and not stat.S_ISLNK(details.st_mode):
            total += details.st_size
    return total


def _is_regular_file(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(details.st_mode) and not stat.S_ISLNK(details.st_mode)


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("缺少时间")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("不是 ISO 8601 时间") from exc
    if parsed.tzinfo is None:
        raise ValueError("时间缺少时区")
    return parsed.astimezone(timezone.utc)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now 必须包含时区")
    return current.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
