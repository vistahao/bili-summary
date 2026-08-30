from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .models import InputSpec


BV_PATTERN = re.compile(r"(?<![0-9A-Za-z])(BV[0-9A-Za-z]{10})(?![0-9A-Za-z])")
WINDOWS_PATH_PATTERN = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<tail>.*)$")


class InputError(ValueError):
    """Raised when an input cannot be safely normalized."""


def parse_bilibili_input(value: str) -> InputSpec:
    original = value.strip()
    if not original:
        raise InputError("视频链接或 BV 号不能为空")

    bare_match = BV_PATTERN.fullmatch(original)
    if bare_match:
        bv_id = bare_match.group(1)
        canonical = f"https://www.bilibili.com/video/{bv_id}"
        return InputSpec(
            source_type="bilibili",
            original=original,
            canonical=canonical,
            display_title="待获取标题",
            identity=f"{bv_id}:p1",
            metadata={"bv_id": bv_id, "part": 1, "needs_network": True},
        )

    candidate = original
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlsplit(candidate)
    host = (parsed.hostname or "").lower()

    if host in {"b23.tv", "www.b23.tv"}:
        return InputSpec(
            source_type="bilibili_short",
            original=original,
            canonical=candidate,
            display_title="待解析短链接",
            metadata={"needs_network": True},
        )

    if host not in {"bilibili.com", "www.bilibili.com", "m.bilibili.com"}:
        raise InputError("只接受哔哩哔哩链接、b23.tv 短链接或 BV 号")

    bv_match = BV_PATTERN.search(parsed.path)
    if not bv_match:
        raise InputError("链接中没有找到有效的 BV 号")
    bv_id = bv_match.group(1)

    part = 1
    raw_part = parse_qs(parsed.query).get("p", ["1"])[0]
    try:
        part = int(raw_part)
    except ValueError as exc:
        raise InputError("分P参数 p 必须是正整数") from exc
    if part < 1:
        raise InputError("分P参数 p 必须大于等于 1")

    canonical = f"https://www.bilibili.com/video/{bv_id}"
    if part != 1:
        canonical = f"{canonical}?p={part}"
    return InputSpec(
        source_type="bilibili",
        original=original,
        canonical=canonical,
        display_title="待获取标题",
        identity=f"{bv_id}:p{part}",
        metadata={"bv_id": bv_id, "part": part, "needs_network": True},
    )


def normalize_local_path(value: str) -> Path:
    original = value.strip()
    if not original:
        raise InputError("本地文件路径不能为空")

    windows_match = WINDOWS_PATH_PATTERN.match(original)
    if windows_match:
        drive = windows_match.group("drive").lower()
        tail = windows_match.group("tail").replace("\\", "/")
        candidate = Path(f"/mnt/{drive}/{tail}")
    else:
        candidate = Path(original).expanduser()

    if not candidate.is_absolute():
        raise InputError("本地 MP4 必须使用 Windows 盘符路径或 WSL 绝对路径")
    return candidate.resolve(strict=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_local_mp4(value: str, *, compute_hash: bool = False) -> InputSpec:
    path = normalize_local_path(value)
    if path.suffix.lower() != ".mp4":
        raise InputError("第一版只支持单个 MP4 文件")
    if not path.exists():
        raise InputError(f"本地文件不存在：{path}")
    if not path.is_file():
        raise InputError(f"路径不是普通文件：{path}")

    stat_result = path.stat()
    file_hash = sha256_file(path) if compute_hash else None
    metadata = {
        "path": str(path),
        "size_bytes": stat_result.st_size,
        "modified_ns": stat_result.st_mtime_ns,
        "sha256": file_hash,
        "hash_status": "complete" if file_hash else "deferred",
        "media_probe_status": "deferred_until_ffprobe",
    }
    return InputSpec(
        source_type="local_mp4",
        original=value,
        canonical=str(path),
        display_title=path.stem,
        identity=f"sha256:{file_hash}" if file_hash else None,
        metadata=metadata,
    )
