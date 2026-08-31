import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from bili_summary.cli import main
from bili_summary.media import MediaError


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
        self.assertEqual(result["processing"]["content_mode"], "lecture")

    def test_preview_accepts_practice_content_mode(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                ["run", "BV1fKtN6DErG", "--content-mode", "practice", "--json"]
            )
        self.assertEqual(exit_code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["processing"]["content_mode"], "practice")

    def test_cache_commands_are_read_only_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.ini"
            config.write_text(
                f"[storage]\ndata_root = {root / 'data'}\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["--config", str(config), "cache-clean", "--json"])
            self.assertEqual(exit_code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "cache_cleanup_preview")
            self.assertEqual(result["deleted_items"], 0)

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

    def test_local_probe_reports_missing_ffprobe_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "sample.mp4"
            sample.write_bytes(b"sample")
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr), mock.patch(
                "bili_summary.cli.inspect_local_media",
                side_effect=MediaError("没有找到 ffprobe"),
            ):
                code = main(["run-file", str(sample), "--probe", "--json"])
            self.assertEqual(code, 2)
            self.assertIn("ffprobe", stderr.getvalue())

    def test_local_audio_sample_preparation_is_reported_separately_from_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "sample.mp4"
            sample.write_bytes(b"sample")
            output = io.StringIO()
            inspected = {
                "external_srt": None,
                "probe": {
                    "format": {"duration_seconds": 900.0},
                    "streams": [{"index": 1, "type": "audio", "codec": "aac"}],
                },
                "text_source": {"kind": "audio_transcription_required"},
            }
            prepared = {
                "kind": "transcription_sample",
                "audio": {"path": str(Path(temp_dir) / "sample.wav")},
                "reused": False,
            }
            with redirect_stdout(output), mock.patch(
                "bili_summary.cli.inspect_local_media", return_value=inspected
            ), mock.patch(
                "bili_summary.cli.prepare_transcription_sample", return_value=prepared
            ) as prepare:
                code = main(
                    [
                        "run-file",
                        str(sample),
                        "--prepare-audio-sample",
                        "--sample-start",
                        "30",
                        "--sample-minutes",
                        "5",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "audio_sample_ready")
            self.assertEqual(result["input"]["metadata"]["hash_status"], "complete")
            self.assertEqual(result["temporary_audio"]["kind"], "transcription_sample")
            self.assertEqual(prepare.call_args.kwargs["duration_seconds"], 300)

    def test_local_execute_hashes_probes_and_runs_pipeline_after_yes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "sample.mp4"
            sample.write_bytes(b"sample-video")
            inspected = {
                "external_srt": None,
                "probe": {
                    "format": {"duration_seconds": 100.0},
                    "streams": [{"index": 1, "type": "audio", "codec": "aac"}],
                },
                "text_source": {"kind": "audio_transcription_required"},
            }
            completed = {
                "status": "complete",
                "output_dir": str(Path(temp_dir) / "result"),
                "files": [],
            }
            output = io.StringIO()
            with redirect_stdout(output), mock.patch(
                "bili_summary.cli.inspect_local_media", return_value=inspected
            ), mock.patch(
                "bili_summary.cli.run_local_file_pipeline", return_value=completed
            ) as run:
                code = main(
                    [
                        "run-file",
                        str(sample),
                        "--execute",
                        "--yes",
                        "--audit-level",
                        "off",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "complete")
            spec = run.call_args.args[0]
            self.assertEqual(spec.metadata["hash_status"], "complete")
            self.assertEqual(run.call_args.kwargs["media"], inspected)


if __name__ == "__main__":
    unittest.main()
