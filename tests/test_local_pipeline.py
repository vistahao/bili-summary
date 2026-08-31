from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bili_summary.aliyun_asr import AliyunAsrError, PARAFORMER_V2_MODEL, QWEN_FILETRANS_MODEL
from bili_summary.config import Settings
from bili_summary.inputs import parse_local_mp4
from bili_summary.local_pipeline import run_local_file_pipeline
from bili_summary.models import StructuredResult, TranscriptSegment
from bili_summary.text_routing import resolve_text_plan


class _TextBackend:
    def process(self, _prompt, schema_path):  # type: ignore[no-untyped-def]
        if schema_path.name == "organize_outputs.schema.json":
            payload = {"clean_markdown": "## 内容 [00:00:00]\n\n测试。", "warnings": []}
        elif schema_path.name == "summary_notes.schema.json":
            payload = {"summary_notes_markdown": "- [00:00:00] 测试", "warnings": []}
        elif schema_path.name == "summary_outputs.schema.json":
            payload = {"summary_markdown": "# 学习总结\n\n测试。", "warnings": []}
        else:
            raise AssertionError(schema_path)
        return StructuredResult(
            payload=payload,
            usage={"input_tokens": 1, "output_tokens": 1},
            elapsed_seconds=0.01,
            backend_metadata={"model": "fake"},
        )


def _media(kind: str, *, path: Path | None = None, duration: float = 100.0) -> dict:
    text_source = {"kind": kind}
    if path is not None:
        text_source["path"] = str(path)
    return {
        "external_srt": str(path) if path else None,
        "probe": {
            "format": {"duration_seconds": duration, "size_bytes": 100},
            "streams": [{"index": 1, "type": "audio", "codec": "aac"}],
            "stream_counts": {"video": 1, "audio": 1, "subtitle": 0},
        },
        "text_source": text_source,
    }


class LocalPipelineTests(unittest.TestCase):
    def test_external_srt_reuses_long_pipeline_and_records_local_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "课程.mp4"
            media_path.write_bytes(b"video")
            subtitle = root / "课程.srt"
            subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n测试字幕。\n",
                encoding="utf-8",
            )
            spec = parse_local_mp4(str(media_path), compute_hash=True)
            settings = Settings(data_root=root / "data", audit_level="off")
            result = run_local_file_pipeline(
                spec,
                settings,
                subject="测试",
                course=None,
                title_override=None,
                schemas_dir=Path("schemas"),
                compare_deep=False,
                audit_level="off",
                media=_media("external_srt", path=subtitle),
                backend=_TextBackend(),
                plan_selector=lambda _preview: resolve_text_plan(settings, audit_level="off"),
            )
            self.assertEqual(result["status"], "complete")
            record = json.loads(
                (Path(result["output_dir"]) / "source.json").read_text(encoding="utf-8")
            )
            self.assertIn("local", record)
            self.assertNotIn("platform", record)
            self.assertEqual(record["local"]["subtitle_acquisition"]["kind"], "external_srt")
            self.assertEqual(record["task_id"], f"local-{spec.metadata['sha256'][:16]}")

    def test_qwen_failure_falls_back_to_paraformer_without_local_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "课程.mp4"
            media_path.write_bytes(b"video")
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"RIFF" + b"\0" * 64)
            spec = parse_local_mp4(str(media_path), compute_hash=True)
            settings = Settings(data_root=root / "data", audit_level="off")
            models = []

            def online_runner(_source, _settings, *, model, **_kwargs):  # type: ignore[no-untyped-def]
                models.append(model)
                if model == QWEN_FILETRANS_MODEL:
                    raise AliyunAsrError("模拟 Qwen 故障", code="internal_error", retryable=True)
                return {
                    "provider": "aliyun_bailian",
                    "model": model,
                    "task_id": "para-task",
                    "reused": False,
                    "segments": (TranscriptSegment(0, 1000, "备用字幕。"),),
                    "raw_transcript": {"transcripts": []},
                    "usage": {"duration": 100},
                    "provider_timing": {},
                    "estimated_max_cost_cny": 0.008,
                }

            result = run_local_file_pipeline(
                spec,
                settings,
                subject="测试",
                course=None,
                title_override=None,
                schemas_dir=Path("schemas"),
                compare_deep=False,
                audit_level="off",
                media=_media("audio_transcription_required"),
                backend=_TextBackend(),
                plan_selector=lambda _preview: resolve_text_plan(settings, audit_level="off"),
                online_runner=online_runner,
                local_runner=lambda *_args, **_kwargs: self.fail("不应调用本地 CPU"),
                audio_preparer=lambda *_args, **_kwargs: {
                    "kind": "full_transcription_audio",
                    "audio": {"path": str(audio_path)},
                    "backup_policy": "do_not_backup",
                },
            )
            self.assertEqual(models, [QWEN_FILETRANS_MODEL, PARAFORMER_V2_MODEL])
            record = json.loads(
                (Path(result["output_dir"]) / "source.json").read_text(encoding="utf-8")
            )
            acquisition = record["local"]["subtitle_acquisition"]
            self.assertEqual(acquisition["model"], PARAFORMER_V2_MODEL)
            self.assertEqual(acquisition["attempt_errors"][0]["code"], "internal_error")

    def test_cost_gate_stops_before_complete_audio_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "课程.mp4"
            media_path.write_bytes(b"video")
            spec = parse_local_mp4(str(media_path), compute_hash=True)
            settings = Settings(
                data_root=root / "data",
                audit_level="off",
                cost_submission_limit_cny=0.01,
            )
            with self.assertRaisesRegex(AliyunAsrError, "尚未提取或上传"):
                run_local_file_pipeline(
                    spec,
                    settings,
                    subject="测试",
                    course=None,
                    title_override=None,
                    schemas_dir=Path("schemas"),
                    compare_deep=False,
                    audit_level="off",
                    media=_media("audio_transcription_required", duration=1000),
                    backend=_TextBackend(),
                    plan_selector=lambda _preview: resolve_text_plan(
                        settings, audit_level="off"
                    ),
                    audio_preparer=lambda *_args, **_kwargs: self.fail("门槛前不应提取"),
                )


if __name__ == "__main__":
    unittest.main()
