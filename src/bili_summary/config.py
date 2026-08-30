from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


VALID_AUDIT_LEVELS = {"off", "basic", "deep"}
VALID_TRANSCRIBER_MODES = {"auto", "local", "online"}


@dataclass(frozen=True)
class Settings:
    data_root: Path = Path("/home/dev/bili-summary-data")
    audit_level: str = "basic"
    transcriber_mode: str = "auto"
    cost_submission_limit_cny: float = 1.0
    bilibili_cookie_file: Path | None = None
    codex_model: str | None = None
    long_chunk_target_minutes: int = 15
    long_chunk_max_minutes: int = 20
    deep_chunk_target_minutes: int = 50
    deep_chunk_max_minutes: int = 55


def load_settings(path: Path | None = None) -> Settings:
    if path is None or not path.exists():
        return Settings()

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    cookie_value = parser.get("bilibili", "cookie_file", fallback="").strip()
    codex_model = parser.get("codex", "model", fallback="").strip()
    settings = Settings(
        data_root=Path(parser.get("storage", "data_root", fallback=str(Settings.data_root))).expanduser(),
        audit_level=parser.get("processing", "audit_level", fallback="basic").strip().lower(),
        transcriber_mode=parser.get("processing", "transcriber_mode", fallback="auto").strip().lower(),
        cost_submission_limit_cny=parser.getfloat(
            "processing", "cost_submission_limit_cny", fallback=1.0
        ),
        bilibili_cookie_file=Path(cookie_value).expanduser() if cookie_value else None,
        codex_model=codex_model or None,
        long_chunk_target_minutes=parser.getint(
            "long_processing", "chunk_target_minutes", fallback=15
        ),
        long_chunk_max_minutes=parser.getint(
            "long_processing", "chunk_max_minutes", fallback=20
        ),
        deep_chunk_target_minutes=parser.getint(
            "long_processing", "deep_chunk_target_minutes", fallback=50
        ),
        deep_chunk_max_minutes=parser.getint(
            "long_processing", "deep_chunk_max_minutes", fallback=55
        ),
    )
    if settings.audit_level not in VALID_AUDIT_LEVELS:
        raise ValueError(f"无效的 audit_level：{settings.audit_level}")
    if settings.transcriber_mode not in VALID_TRANSCRIBER_MODES:
        raise ValueError(f"无效的 transcriber_mode：{settings.transcriber_mode}")
    if settings.cost_submission_limit_cny < 0:
        raise ValueError("cost_submission_limit_cny 不能小于 0")
    if settings.long_chunk_target_minutes <= 0:
        raise ValueError("chunk_target_minutes 必须大于 0")
    if settings.long_chunk_max_minutes < settings.long_chunk_target_minutes:
        raise ValueError("chunk_max_minutes 不能小于 chunk_target_minutes")
    if settings.deep_chunk_target_minutes <= 0:
        raise ValueError("deep_chunk_target_minutes 必须大于 0")
    if settings.deep_chunk_max_minutes < settings.deep_chunk_target_minutes:
        raise ValueError("deep_chunk_max_minutes 不能小于 deep_chunk_target_minutes")
    return settings
