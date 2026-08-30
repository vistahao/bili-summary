import json
import tempfile
import unittest
from pathlib import Path

from bili_summary.config import Settings
from bili_summary.inputs import parse_bilibili_input
from bili_summary.models import StudyOutputs
from bili_summary.pipeline import run_bilibili_pipeline
from test_bilibili import FakeBilibiliClient


class FakeStudyBackend:
    def process(self, transcript_srt, source_context):  # type: ignore[no-untyped-def]
        self.transcript_srt = transcript_srt
        self.source_context = source_context
        return StudyOutputs(
            clean_transcript_markdown="# 完整整理稿\n\n正文。\n",
            audit_markdown="# 审校报告\n\n未发现。\n",
            summary_markdown="# 学习总结\n\n要点。\n",
            warnings=(),
            usage={"input_tokens": 100, "output_tokens": 50},
            call_count=1,
            elapsed_seconds=0.25,
            backend_metadata={
                "cli_version": "codex-cli test",
                "model": "test-model",
                "model_source": "test",
            },
        )


class PipelineTests(unittest.TestCase):
    def test_writes_traceable_outputs_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_root=Path(temp_dir))
            result = run_bilibili_pipeline(
                parse_bilibili_input("BV1fKtN6DErG"),
                settings,
                subject="测试学科",
                course=None,
                title_override=None,
                schema_path=Path("schemas/study_outputs.schema.json"),
                backend=FakeStudyBackend(),
                client=FakeBilibiliClient(),
            )
            self.assertEqual(result["status"], "complete")
            output_dir = Path(result["output_dir"])
            for name in ("字幕.srt", "原始字幕.json", "完整整理稿.md", "审校报告.md", "学习总结.md", "source.json"):
                self.assertTrue((output_dir / name).is_file(), name)
            source = json.loads((output_dir / "source.json").read_text(encoding="utf-8"))
            self.assertEqual(source["status"], "complete")
            self.assertEqual(source["processing"]["codex"]["usage"]["input_tokens"], 100)
            self.assertNotIn("cookie", json.dumps(source).lower())


if __name__ == "__main__":
    unittest.main()
