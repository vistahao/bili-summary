import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from bili_summary.cli import main


class CliTests(unittest.TestCase):
    def test_link_preview_is_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "run",
                    "BV1fKtN6DErG",
                    "--subject",
                    "计算机",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "preview_only")
        self.assertIn("/计算机/", result["archive_path"])

    def test_local_preview_does_not_hash_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "本地课程.mp4"
            sample.write_bytes(b"small-test")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["run-file", str(sample), "--json"])
            self.assertEqual(exit_code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["input"]["metadata"]["hash_status"], "deferred")


if __name__ == "__main__":
    unittest.main()
