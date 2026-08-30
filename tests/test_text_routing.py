import tempfile
import unittest
from pathlib import Path

from bili_summary.config import Settings
from bili_summary.models import TextProfile
from bili_summary.text_routing import (
    TextSelectionCancelled,
    choose_execution_plan,
    load_deepseek_api_key,
    parse_route_overrides,
    resolve_text_plan,
)


class TextRoutingTests(unittest.TestCase):
    def test_route_override_has_highest_noninteractive_priority(self) -> None:
        profiles = {
            "codex_default": TextProfile("codex_default", "codex_exec", "codex", "high"),
            "deepseek_fast": TextProfile("deepseek_fast", "deepseek_http", "flash", "low"),
        }
        settings = Settings(
            text_profiles=profiles,
            text_routes={task: "codex_default" for task in ("organize", "summary", "basic_audit", "deep_audit")},
            text_presets={
                "speed": {task: "deepseek_fast" for task in ("organize", "summary", "basic_audit", "deep_audit")}
            },
        )
        plan = resolve_text_plan(
            settings,
            audit_level="basic",
            preset="speed",
            route_overrides={"summary": "codex_default"},
        )
        self.assertEqual(plan.routes["organize"].name, "deepseek_fast")
        self.assertEqual(plan.routes["summary"].name, "codex_default")
        self.assertEqual(parse_route_overrides(["summary=codex_default"]), {"summary": "codex_default"})

    def test_noninteractive_requires_yes_and_cancel_sends_nothing(self) -> None:
        settings = Settings()
        arguments = {
            "settings": settings,
            "preview": {"title": "测试", "estimated_calls": 3},
            "audit_level": "basic",
            "preset": None,
            "route_overrides": {},
        }
        with self.assertRaisesRegex(ValueError, "--yes"):
            choose_execution_plan(**arguments, assume_yes=False, interactive=False)
        with self.assertRaises(TextSelectionCancelled):
            choose_execution_plan(
                **arguments,
                assume_yes=False,
                interactive=True,
                input_fn=lambda _prompt: "q",
                output_fn=lambda _message: None,
            )
        plan = choose_execution_plan(**arguments, assume_yes=True, interactive=False)
        self.assertEqual(plan.routes["organize"].name, "codex_default")

    def test_interactive_speed_preset_and_manual_audit_override(self) -> None:
        settings = Settings()
        base = {
            "settings": settings,
            "preview": {"title": "测试", "estimated_calls": 3},
            "audit_level": "basic",
            "preset": None,
            "route_overrides": {},
            "assume_yes": False,
            "interactive": True,
            "output_fn": lambda _message: None,
        }
        speed = choose_execution_plan(**base, input_fn=lambda _prompt: "3")
        self.assertEqual(speed.preset, "speed")

        answers = iter(["1", "deep", "", "", "", ""])
        manual = choose_execution_plan(**base, input_fn=lambda _prompt: next(answers))
        self.assertEqual(manual.audit_level, "deep")

    def test_secret_file_requires_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "deepseek.key"
            path.write_text("test-key\n", encoding="utf-8")
            path.chmod(0o644)
            settings = Settings(deepseek_api_key_file=path)
            with self.assertRaisesRegex(ValueError, "600"):
                load_deepseek_api_key(settings, environ={})
            path.chmod(0o600)
            self.assertEqual(load_deepseek_api_key(settings, environ={}), "test-key")


if __name__ == "__main__":
    unittest.main()
