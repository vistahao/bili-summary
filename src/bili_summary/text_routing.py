from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .adapters import CodexStructuredBackend, DeepSeekHttpBackend, TextBackendError
from .config import Settings, TEXT_TASKS, VALID_AUDIT_LEVELS
from .models import StructuredResult, TextExecutionPlan, TextProfile


class StructuredBackend(Protocol):
    def process(self, prompt: str, schema_path: Path) -> StructuredResult: ...


class TextSelectionCancelled(ValueError):
    pass


def parse_route_overrides(values: list[str] | None) -> dict[str, str]:
    routes: dict[str, str] = {}
    for value in values or []:
        task, separator, profile = value.partition("=")
        task = task.strip()
        profile = profile.strip()
        if not separator or task not in TEXT_TASKS or not profile:
            valid = ", ".join(TEXT_TASKS)
            raise ValueError(f"无效 --route：{value}；格式为 <任务>=<配置>，任务可用：{valid}")
        routes[task] = profile
    return routes


def resolve_text_plan(
    settings: Settings,
    *,
    audit_level: str,
    preset: str | None = None,
    route_overrides: Mapping[str, str] | None = None,
) -> TextExecutionPlan:
    if audit_level not in VALID_AUDIT_LEVELS:
        raise ValueError(f"无效审校档位：{audit_level}")
    routes = dict(settings.text_routes)
    if preset:
        if preset not in settings.text_presets:
            choices = ", ".join(sorted(settings.text_presets)) or "无"
            raise ValueError(f"不存在文本预设：{preset}；可用预设：{choices}")
        routes.update(settings.text_presets[preset])
    routes.update(route_overrides or {})
    resolved: dict[str, TextProfile] = {}
    for task in TEXT_TASKS:
        profile_name = routes.get(task)
        if not profile_name or profile_name not in settings.text_profiles:
            raise ValueError(f"任务 {task} 引用了不存在的文本配置：{profile_name or '空'}")
        profile = settings.text_profiles[profile_name]
        if (
            profile.driver == "codex_exec"
            and profile.model is None
            and settings.codex_model
        ):
            profile = TextProfile(
                name=profile.name,
                driver=profile.driver,
                model=settings.codex_model,
                reasoning=profile.reasoning,
                max_output_tokens=profile.max_output_tokens,
            )
        resolved[task] = profile
    return TextExecutionPlan(audit_level=audit_level, routes=resolved, preset=preset)


def active_tasks(audit_level: str) -> tuple[str, ...]:
    tasks = ["organize", "summary"]
    if audit_level in {"basic", "deep"}:
        tasks.append("basic_audit")
    if audit_level == "deep":
        tasks.append("deep_audit")
    return tuple(tasks)


def format_plan(plan: TextExecutionPlan) -> list[str]:
    labels = {
        "organize": "整理",
        "summary": "总结",
        "basic_audit": "Basic 审校",
        "deep_audit": "Deep 审校",
    }
    lines = [f"审校档位: {plan.audit_level}"]
    for task in active_tasks(plan.audit_level):
        profile = plan.routes[task]
        lines.append(
            f"{labels[task]}: {profile.name} / {profile.driver} / "
            f"{profile.model or '后端默认模型'} / {profile.reasoning}"
        )
    return lines


def choose_execution_plan(
    settings: Settings,
    *,
    preview: Mapping[str, object],
    audit_level: str,
    preset: str | None,
    route_overrides: Mapping[str, str],
    assume_yes: bool,
    interactive: bool,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> TextExecutionPlan:
    plan = resolve_text_plan(
        settings,
        audit_level=audit_level,
        preset=preset,
        route_overrides=route_overrides,
    )
    if assume_yes:
        validate_plan_credentials(plan, settings)
        return plan
    if not interactive:
        raise ValueError(
            "当前不是交互终端，尚未确认文本方案；请在原命令末尾增加 --yes，"
            "也可配合 --profile 或 --route 明确覆盖"
        )

    output_fn("\n文本处理方案确认")
    output_fn(f"视频: {preview.get('title', '未知标题')}")
    output_fn(f"时长: {preview.get('duration', '未知')}；文本来源: {preview.get('source', '未知')}")
    output_fn(f"预计调用: {preview.get('estimated_calls', '未知')}")
    for line in format_plan(plan):
        output_fn(line)
    output_fn("[Enter] 使用当前方案  [1] 修改本次任务  [2] 质量优先  [3] 速度优先  [q] 取消")
    choice = input_fn("选择: ").strip().lower()
    if choice == "q":
        raise TextSelectionCancelled("用户取消；未调用文本模型")
    if choice == "2":
        plan = resolve_text_plan(settings, audit_level=audit_level, preset="quality")
    elif choice == "3":
        plan = resolve_text_plan(settings, audit_level=audit_level, preset="speed")
    elif choice == "1":
        plan = _modify_plan(plan, settings, input_fn=input_fn, output_fn=output_fn)
    elif choice:
        raise ValueError(f"无效选择：{choice}")
    validate_plan_credentials(plan, settings)
    for line in format_plan(plan):
        output_fn(line)
    output_fn("本次文本方案已冻结；后续切片将沿用该方案。")
    return plan


def _modify_plan(
    plan: TextExecutionPlan,
    settings: Settings,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> TextExecutionPlan:
    audit = input_fn(f"审校档位 off/basic/deep [{plan.audit_level}]: ").strip().lower()
    audit_level = audit or plan.audit_level
    if audit_level not in VALID_AUDIT_LEVELS:
        raise ValueError(f"无效审校档位：{audit_level}")
    routes = {task: profile.name for task, profile in plan.routes.items()}
    available = ", ".join(sorted(settings.text_profiles))
    output_fn(f"可用文本配置: {available}")
    for task in active_tasks(audit_level):
        current = routes[task]
        selected = input_fn(f"{task} 配置 [{current}]: ").strip()
        if selected:
            routes[task] = selected
    return resolve_text_plan(settings, audit_level=audit_level, route_overrides=routes)


def validate_plan_credentials(plan: TextExecutionPlan, settings: Settings) -> None:
    if any(
        plan.routes[task].driver == "deepseek_http"
        for task in active_tasks(plan.audit_level)
    ):
        load_deepseek_api_key(settings)


def load_deepseek_api_key(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    from_environment = environment.get(settings.deepseek_api_key_env, "").strip()
    if from_environment:
        return from_environment
    path = settings.deepseek_api_key_file
    if path is None:
        raise ValueError(
            f"DeepSeek 配置需要密钥；请设置环境变量 {settings.deepseek_api_key_env}，"
            "或在 Git 忽略的权限 600 文件中配置 api_key_file"
        )
    if not path.is_file():
        raise ValueError(f"DeepSeek API Key 文件不存在：{path}")
    if path.stat().st_mode & 0o077:
        raise ValueError(f"DeepSeek API Key 文件权限过宽：{path}；请设置为 600")
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("DeepSeek API Key 文件必须只有一行非空内容")
    return value


def build_backend(profile: TextProfile, settings: Settings) -> StructuredBackend:
    if profile.driver == "codex_exec":
        return CodexStructuredBackend(
            model=profile.model,
            reasoning=profile.reasoning,
        )
    if profile.driver == "deepseek_http":
        return DeepSeekHttpBackend(
            model=profile.model or "",
            reasoning=profile.reasoning,
            api_key=load_deepseek_api_key(settings),
            base_url=settings.deepseek_base_url,
            max_output_tokens=profile.max_output_tokens,
        )
    raise TextBackendError(
        f"不支持的文本后端：{profile.driver}",
        provider="local",
        code="unsupported_backend",
    )
