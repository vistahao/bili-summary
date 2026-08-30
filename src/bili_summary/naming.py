from __future__ import annotations

import re
import unicodedata
from pathlib import Path


INVALID_FILENAME_CHARS = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
WHITESPACE = re.compile(r"\s+")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def sanitize_component(value: str, *, fallback: str = "未命名", max_length: int = 100) -> str:
    cleaned = unicodedata.normalize("NFKC", value)
    cleaned = INVALID_FILENAME_CHARS.sub("-", cleaned)
    cleaned = WHITESPACE.sub(" ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned[:max_length].rstrip(" .") or fallback


def build_archive_path(
    data_root: Path,
    *,
    subject: str,
    title: str,
    course: str | None = None,
) -> Path:
    result = data_root / sanitize_component(subject, fallback="未分类")
    if course:
        result /= sanitize_component(course)
    return result / sanitize_component(title)
