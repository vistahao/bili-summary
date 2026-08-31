from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from bili_summary.aliyun_asr import AliyunAsrError
from bili_summary.asr_evaluation import (
    run_aliyun_asr_comparison,
    run_aliyun_file_transcription,
)
from bili_summary.config import load_settings


class _FakeUploader:
    def __init__(self) -> None:
        self.models: list[str] = []

    def upload(self, model: str, source: Path) -> dict[str, object]:
        self.models.append(model)
        return {
            "model": model,
            "file_url": f"oss://temporary/{model}/{source.name}",
            "size_bytes": source.stat().st_size,
            "expires_after_hours": 48,
            "backup_policy": "do_not_backup",
        }


class _FakeClient:
    def __init__(self) -> None:
        self.fail_first_submit = True
        self.submitted: list[str] = []

    def submit(self, *, model: str, file_url: str) -> dict[str, str]:
        self.submitted.append(model)
        if self.fail_first_submit:
            self.fail_first_submit = False
            raise AliyunAsrError("模拟响应丢失", code="network_error", retryable=True)
        return {
            "task_id": f"task-{len(self.submitted)}",
            "task_status": "PENDING",
            "request_id": f"request-{len(self.submitted)}",
        }

    def wait_for_completion(self, task_id: str, *, progress=None):  # type: ignore[no-untyped-def]
        return {
            "output": {
                "task_id": task_id,
                "task_status": "SUCCEEDED",
                "submit_time": "2026-08-31 10:00:00.000",
                "scheduled_time": "2026-08-31 10:00:00.010",
                "end_time": "2026-08-31 10:00:01.250",
            },
            "usage": {"seconds": 300},
        }

    def download_transcription(self, _response):  # type: ignore[no-untyped-def]
        return {
            "transcripts": [
                {
                    "sentences": [
                        {"begin_time": 0, "end_time": 1000, "text": "测试字幕。"}
                    ]
                }
            ]
        }


class AsrEvaluationTests(unittest.TestCase):
    def test_reuses_upload_after_submit_response_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\0" * (300 * 16_000 * 2))
            key_file = root / "key.txt"
            key_file.write_text("test-key", encoding="utf-8")
            key_file.chmod(0o600)
            config = root / "config.ini"
            config.write_text(
                "[storage]\n"
                f"data_root = {root / 'data'}\n"
                "[aliyun_asr]\n"
                f"api_key_file = {key_file}\n",
                encoding="utf-8",
            )
            settings = load_settings(config)
            client = _FakeClient()
            uploader = _FakeUploader()
            factories = {
                "client_factory": lambda _key: client,
                "uploader_factory": lambda _key: uploader,
            }

            with self.assertRaisesRegex(AliyunAsrError, "模拟响应丢失"):
                run_aliyun_asr_comparison(source, settings, **factories)
            result = run_aliyun_asr_comparison(source, settings, **factories)

            self.assertEqual(
                uploader.models,
                ["qwen3-asr-flash-filetrans", "paraformer-v2"],
            )
            self.assertEqual(
                client.submitted,
                [
                    "qwen3-asr-flash-filetrans",
                    "qwen3-asr-flash-filetrans",
                    "paraformer-v2",
                ],
            )
            self.assertTrue(all(model["status"] == "complete" for model in result["models"]))
            self.assertEqual(
                result["models"][0]["provider_timing"]["service_elapsed_seconds"],
                1.25,
            )
            second = run_aliyun_asr_comparison(source, settings, **factories)
            self.assertTrue(all(model["reused"] for model in second["models"]))
            self.assertEqual(len(client.submitted), 3)

    def test_single_model_transcription_obeys_cost_gate_and_returns_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "audio.wav"
            source.write_bytes(b"RIFF" + b"\0" * 64)
            key_file = root / "key.txt"
            key_file.write_text("test-key", encoding="utf-8")
            key_file.chmod(0o600)
            settings = load_settings(root / "missing.ini")
            settings = settings.__class__(
                **{
                    **settings.__dict__,
                    "data_root": root / "data",
                    "aliyun_asr_api_key_file": key_file,
                    "cost_submission_limit_cny": 0.05,
                }
            )
            with self.assertRaisesRegex(AliyunAsrError, "超过提交门槛"):
                run_aliyun_file_transcription(
                    source,
                    settings,
                    model="qwen3-asr-flash-filetrans",
                    duration_seconds=300,
                    cache_root=root / "cache",
                )

            client = _FakeClient()
            client.fail_first_submit = False
            uploader = _FakeUploader()
            settings = settings.__class__(
                **{**settings.__dict__, "cost_submission_limit_cny": 1.0}
            )
            result = run_aliyun_file_transcription(
                source,
                settings,
                model="qwen3-asr-flash-filetrans",
                duration_seconds=300,
                cache_root=root / "cache",
                client_factory=lambda _key: client,
                uploader_factory=lambda _key: uploader,
            )
            self.assertEqual(result["model"], "qwen3-asr-flash-filetrans")
            self.assertEqual(result["segments"][0].text, "测试字幕。")
            self.assertEqual(result["provider_timing"]["service_elapsed_seconds"], 1.25)


if __name__ == "__main__":
    unittest.main()
