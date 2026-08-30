import tempfile
import unittest
from pathlib import Path

from bili_summary.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_loads_non_secret_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.ini"
            config_path.write_text(
                "[storage]\n"
                "data_root = /tmp/study-data\n"
                "[processing]\n"
                "audit_level = deep\n"
                "transcriber_mode = local\n"
                "cost_submission_limit_cny = 0.5\n",
                encoding="utf-8",
            )
            settings = load_settings(config_path)
            self.assertEqual(settings.data_root, Path("/tmp/study-data"))
            self.assertEqual(settings.audit_level, "deep")
            self.assertEqual(settings.transcriber_mode, "local")
            self.assertEqual(settings.cost_submission_limit_cny, 0.5)


if __name__ == "__main__":
    unittest.main()
