from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import TextProfile


VALID_AUDIT_LEVELS = {"off", "basic", "deep"}
VALID_CONTENT_MODES = {"lecture", "practice"}
VALID_TRANSCRIBER_MODES = {"auto", "local", "online"}
TEXT_TASKS = ("organize", "summary", "basic_audit", "deep_audit")
VALID_TEXT_DRIVERS = {"codex_exec", "deepseek_http"}
VALID_CODEX_REASONING = {"default", "low", "medium", "high", "xhigh", "max"}
VALID_DEEPSEEK_REASONING = {"off", "low", "high", "max"}
ALIYUN_WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


@dataclass(frozen=True)
class Settings:
    data_root: Path = Path("/home/dev/bili-summary-data")
    audit_level: str = "basic"
    content_mode: str = "lecture"
    transcriber_mode: str = "auto"
    cost_submission_limit_cny: float = 1.0
    bilibili_cookie_file: Path | None = None
    codex_model: str | None = None
    long_chunk_target_minutes: int = 15
    long_chunk_max_minutes: int = 20
    deep_chunk_target_minutes: int = 50
    deep_chunk_max_minutes: int = 55
    text_profiles: dict[str, TextProfile] = field(
        default_factory=lambda: {
            "codex_default": TextProfile(
                name="codex_default",
                driver="codex_exec",
                model=None,
                reasoning="default",
            ),
            "codex_quality": TextProfile(
                name="codex_quality",
                driver="codex_exec",
                model=None,
                reasoning="high",
            ),
            "codex_speed": TextProfile(
                name="codex_speed",
                driver="codex_exec",
                model=None,
                reasoning="low",
            ),
        }
    )
    text_routes: dict[str, str] = field(
        default_factory=lambda: {task: "codex_default" for task in TEXT_TASKS}
    )
    text_presets: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "quality": {task: "codex_quality" for task in TEXT_TASKS},
            "speed": {task: "codex_speed" for task in TEXT_TASKS},
        }
    )
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_api_key_file: Path | None = None
    aliyun_asr_workspace_id: str | None = None
    aliyun_asr_api_key_env: str = "DASHSCOPE_API_KEY"
    aliyun_asr_api_key_file: Path | None = None
    local_asr_binary: Path | None = None
    local_asr_model: Path | None = None
    local_asr_threads: int = 8


def load_settings(path: Path | None = None) -> Settings:
    parser = configparser.ConfigParser()
    if path is not None and path.exists():
        parser.read(path, encoding="utf-8")
    cookie_value = parser.get("bilibili", "cookie_file", fallback="").strip()
    codex_model = parser.get("codex", "model", fallback="").strip()
    deepseek_key_file = parser.get("deepseek", "api_key_file", fallback="").strip()
    aliyun_key_file = parser.get("aliyun_asr", "api_key_file", fallback="").strip()
    aliyun_workspace_id = parser.get(
        "aliyun_asr", "workspace_id", fallback=""
    ).strip()
    local_asr_binary = parser.get("local_asr", "binary", fallback="").strip()
    local_asr_model = parser.get("local_asr", "model", fallback="").strip()
    profiles = _load_text_profiles(parser, codex_model or None)
    routes = {
        task: parser.get("text_routes", task, fallback="codex_default").strip()
        for task in TEXT_TASKS
    }
    presets = _load_text_presets(parser)
    settings = Settings(
        data_root=Path(parser.get("storage", "data_root", fallback=str(Settings.data_root))).expanduser(),
        audit_level=parser.get("processing", "audit_level", fallback="basic").strip().lower(),
        content_mode=parser.get(
            "processing", "content_mode", fallback="lecture"
        ).strip().lower(),
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
        text_profiles=profiles,
        text_routes=routes,
        text_presets=presets,
        deepseek_base_url=parser.get(
            "deepseek", "base_url", fallback="https://api.deepseek.com"
        ).strip().rstrip("/"),
        deepseek_api_key_env=parser.get(
            "deepseek", "api_key_env", fallback="DEEPSEEK_API_KEY"
        ).strip(),
        deepseek_api_key_file=(
            Path(deepseek_key_file).expanduser() if deepseek_key_file else None
        ),
        aliyun_asr_workspace_id=aliyun_workspace_id or None,
        aliyun_asr_api_key_env=parser.get(
            "aliyun_asr", "api_key_env", fallback="DASHSCOPE_API_KEY"
        ).strip(),
        aliyun_asr_api_key_file=(
            Path(aliyun_key_file).expanduser() if aliyun_key_file else None
        ),
        local_asr_binary=(Path(local_asr_binary).expanduser() if local_asr_binary else None),
        local_asr_model=(Path(local_asr_model).expanduser() if local_asr_model else None),
        local_asr_threads=parser.getint("local_asr", "threads", fallback=8),
    )
    if settings.audit_level not in VALID_AUDIT_LEVELS:
        raise ValueError(f"无效的 audit_level：{settings.audit_level}")
    if settings.content_mode not in VALID_CONTENT_MODES:
        raise ValueError(f"无效的 content_mode：{settings.content_mode}")
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
    if not settings.aliyun_asr_api_key_env:
        raise ValueError("阿里云 ASR api_key_env 不能为空")
    if settings.aliyun_asr_workspace_id and not ALIYUN_WORKSPACE_ID_PATTERN.fullmatch(
        settings.aliyun_asr_workspace_id
    ):
        raise ValueError("阿里云 ASR workspace_id 只能包含字母、数字和连字符")
    if settings.local_asr_threads <= 0 or settings.local_asr_threads > 64:
        raise ValueError("本地 ASR threads 必须在 1 到 64 之间")
    _validate_text_settings(settings)
    return settings


def _load_text_profiles(
    parser: configparser.ConfigParser,
    legacy_codex_model: str | None,
) -> dict[str, TextProfile]:
    profiles: dict[str, TextProfile] = {
        "codex_default": TextProfile(
            name="codex_default",
            driver="codex_exec",
            model=legacy_codex_model,
            reasoning="default",
        ),
        "codex_quality": TextProfile(
            name="codex_quality",
            driver="codex_exec",
            model=legacy_codex_model,
            reasoning="high",
        ),
        "codex_speed": TextProfile(
            name="codex_speed",
            driver="codex_exec",
            model=legacy_codex_model,
            reasoning="low",
        ),
    }
    prefix = "text_profile."
    for section in parser.sections():
        if not section.startswith(prefix):
            continue
        name = section[len(prefix) :].strip()
        model = parser.get(section, "model", fallback="").strip() or None
        profiles[name] = TextProfile(
            name=name,
            driver=parser.get(section, "driver", fallback="").strip().lower(),
            model=model,
            reasoning=parser.get(section, "reasoning", fallback="default").strip().lower(),
            max_output_tokens=parser.getint(section, "max_output_tokens", fallback=32768),
        )
    return profiles


def _load_text_presets(parser: configparser.ConfigParser) -> dict[str, dict[str, str]]:
    presets: dict[str, dict[str, str]] = {
        "quality": {task: "codex_quality" for task in TEXT_TASKS},
        "speed": {task: "codex_speed" for task in TEXT_TASKS},
    }
    prefix = "text_preset."
    for section in parser.sections():
        if section.startswith(prefix):
            name = section[len(prefix) :].strip()
            presets[name] = {
                task: parser.get(section, task).strip()
                for task in TEXT_TASKS
                if parser.has_option(section, task)
            }
    return presets


def _validate_text_settings(settings: Settings) -> None:
    if not settings.deepseek_base_url.startswith("https://"):
        raise ValueError("DeepSeek base_url 必须使用 HTTPS")
    if not settings.deepseek_api_key_env:
        raise ValueError("DeepSeek api_key_env 不能为空")
    for name, profile in settings.text_profiles.items():
        if not name:
            raise ValueError("文本配置名称不能为空")
        if profile.driver not in VALID_TEXT_DRIVERS:
            raise ValueError(f"文本配置 {name} 使用了无效 driver：{profile.driver}")
        if not profile.model and profile.driver == "deepseek_http":
            raise ValueError(f"DeepSeek 文本配置 {name} 必须指定 model")
        valid_reasoning = (
            VALID_CODEX_REASONING
            if profile.driver == "codex_exec"
            else VALID_DEEPSEEK_REASONING
        )
        if profile.reasoning not in valid_reasoning:
            raise ValueError(f"文本配置 {name} 使用了无效 reasoning：{profile.reasoning}")
        if profile.max_output_tokens <= 0 or profile.max_output_tokens > 384_000:
            raise ValueError(f"文本配置 {name} 的 max_output_tokens 必须在 1 到 384000 之间")
    for task, profile_name in settings.text_routes.items():
        if task not in TEXT_TASKS:
            raise ValueError(f"无效文本任务：{task}")
        if profile_name not in settings.text_profiles:
            raise ValueError(f"任务 {task} 引用了不存在的文本配置：{profile_name}")
    for preset_name, routes in settings.text_presets.items():
        for task, profile_name in routes.items():
            if task not in TEXT_TASKS:
                raise ValueError(f"预设 {preset_name} 包含无效任务：{task}")
            if profile_name not in settings.text_profiles:
                raise ValueError(
                    f"预设 {preset_name} 的任务 {task} 引用了不存在的文本配置：{profile_name}"
                )
