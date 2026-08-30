import tempfile
import unittest
from pathlib import Path

from bili_summary.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_example_config_uses_approved_default_routes(self) -> None:
        settings = load_settings(Path("config.example.ini"))
        self.assertEqual(settings.text_routes["organize"], "deepseek_flash_high")
        self.assertEqual(settings.text_routes["summary"], "codex_default")
        self.assertEqual(settings.text_routes["basic_audit"], "deepseek_pro_high")
        self.assertEqual(settings.text_routes["deep_audit"], "deepseek_pro_high")
        self.assertEqual(settings.text_profiles["deepseek_flash_high"].reasoning, "high")

    def test_loads_non_secret_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.ini"
            config_path.write_text(
                "[storage]\n"
                "data_root = /tmp/study-data\n"
                "[processing]\n"
                "audit_level = deep\n"
                "transcriber_mode = local\n"
                "cost_submission_limit_cny = 0.5\n"
                "[bilibili]\n"
                "cookie_file = ~/.secrets/bili-cookie.txt\n"
                "[codex]\n"
                "model = gpt-test\n"
                "[long_processing]\n"
                "chunk_target_minutes = 10\n"
                "chunk_max_minutes = 12\n"
                "deep_chunk_target_minutes = 40\n"
                "deep_chunk_max_minutes = 45\n"
                "[text_routes]\n"
                "organize = codex_default\n"
                "summary = deepseek_test\n"
                "basic_audit = codex_default\n"
                "deep_audit = deepseek_test\n"
                "[text_profile.codex_default]\n"
                "driver = codex_exec\n"
                "model = gpt-test\n"
                "reasoning = high\n"
                "[text_profile.deepseek_test]\n"
                "driver = deepseek_http\n"
                "model = deepseek-test\n"
                "reasoning = low\n"
                "max_output_tokens = 12000\n"
                "[text_preset.quality]\n"
                "summary = deepseek_test\n"
                "deep_audit = deepseek_test\n",
                encoding="utf-8",
            )
            settings = load_settings(config_path)
            self.assertEqual(settings.data_root, Path("/tmp/study-data"))
            self.assertEqual(settings.audit_level, "deep")
            self.assertEqual(settings.transcriber_mode, "local")
            self.assertEqual(settings.cost_submission_limit_cny, 0.5)
            self.assertEqual(
                settings.bilibili_cookie_file,
                Path("~/.secrets/bili-cookie.txt").expanduser(),
            )
            self.assertEqual(settings.codex_model, "gpt-test")
            self.assertEqual(settings.long_chunk_target_minutes, 10)
            self.assertEqual(settings.deep_chunk_max_minutes, 45)
            self.assertEqual(settings.text_routes["summary"], "deepseek_test")
            self.assertEqual(settings.text_profiles["deepseek_test"].reasoning, "low")
            self.assertEqual(settings.text_profiles["deepseek_test"].max_output_tokens, 12000)
            self.assertEqual(settings.text_presets["quality"]["summary"], "deepseek_test")


if __name__ == "__main__":
    unittest.main()
