import json
import tempfile
import unittest
from pathlib import Path

from bili_summary.adapters import CodexError
from bili_summary.config import Settings
from bili_summary.inputs import parse_bilibili_input
from bili_summary.long_pipeline import run_bilibili_long_pipeline, split_transcript
from bili_summary.models import StructuredResult, TranscriptSegment
from test_bilibili import FakeBilibiliClient


class FakeLongBackend:
    def __init__(self, *, fail_on_summary: bool = False) -> None:
        self.calls = []
        self.fail_on_summary = fail_on_summary

    def process(self, prompt, schema_path):  # type: ignore[no-untyped-def]
        self.calls.append(schema_path.name)
        if schema_path.name == "chunk_outputs.schema.json":
            payload = {
                "clean_markdown": "## 第一部分 [00:00:00]\n\n完整内容。",
                "basic_audit_items": [
                    {
                        "category": "疑似字幕错误",
                        "timestamp": "[00:00:01]",
                        "quote": "原句",
                        "concern": "测试风险。",
                    }
                ],
                "summary_notes_markdown": "- [00:00:00] 测试笔记",
                "warnings": [],
            }
        elif schema_path.name == "summary_outputs.schema.json":
            if self.fail_on_summary:
                raise CodexError("模拟中断")
            payload = {"summary_markdown": "# 学习总结\n\n总结。", "warnings": []}
        elif schema_path.name == "deep_audit.schema.json":
            payload = {
                "audit_items": [],
                "coverage_statement": "已覆盖整个测试片段。",
                "warnings": [],
            }
        else:
            raise AssertionError(schema_path)
        return StructuredResult(
            payload=payload,
            usage={"input_tokens": 10, "output_tokens": 5},
            elapsed_seconds=0.1,
            backend_metadata={
                "cli_version": "codex-cli test",
                "model": "test-model",
                "model_source": "test",
            },
        )


class LongPipelineTests(unittest.TestCase):
    def test_split_preserves_every_segment_and_merges_tiny_tail(self) -> None:
        segments = tuple(
            TranscriptSegment(index * 1000, (index + 1) * 1000, f"句子{index}")
            for index in range(25)
        )
        chunks = split_transcript(segments, target_ms=10_000, max_ms=12_000, semantic_gap_ms=0)
        flattened = tuple(segment for chunk in chunks for segment in chunk.segments)
        self.assertEqual(flattened, segments)
        self.assertGreaterEqual(chunks[-1].end_ms - chunks[-1].start_ms, 5_000)

    def test_resumes_after_failure_without_repeating_completed_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_root=Path(temp_dir), codex_model="test-model")
            arguments = {
                "subject": "计算机",
                "course": "长课程",
                "title_override": None,
                "schemas_dir": Path("schemas"),
                "compare_deep": True,
                "client": FakeBilibiliClient(),
            }
            failing = FakeLongBackend(fail_on_summary=True)
            with self.assertRaisesRegex(CodexError, "模拟中断"):
                run_bilibili_long_pipeline(
                    parse_bilibili_input("BV1fKtN6DErG"),
                    settings,
                    backend=failing,
                    **arguments,
                )
            self.assertEqual(failing.calls, ["chunk_outputs.schema.json", "summary_outputs.schema.json"])

            resumed = FakeLongBackend()
            result = run_bilibili_long_pipeline(
                parse_bilibili_input("BV1fKtN6DErG"),
                settings,
                backend=resumed,
                **arguments,
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(
                resumed.calls,
                ["summary_outputs.schema.json", "deep_audit.schema.json"],
            )
            output_dir = Path(result["output_dir"])
            for name in (
                "字幕.srt",
                "完整整理稿.md",
                "审校报告.md",
                "审校报告-deep.md",
                "审校对比.md",
                "学习总结.md",
                "source.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
            source = json.loads((output_dir / "source.json").read_text(encoding="utf-8"))
            self.assertEqual(source["processing"]["codex"]["calls"], 3)
            self.assertEqual(source["processing"]["codex"]["calls_reused_this_run"], 1)
            cache_dir = Path(source["processing"]["strategy"]["cache_dir"])
            self.assertTrue((cache_dir / "primary-001.json").is_file())
            self.assertNotIn("primary_chunk", (cache_dir / "primary-001.json").read_text(encoding="utf-8"))

            no_repeat = FakeLongBackend()
            repeated = run_bilibili_long_pipeline(
                parse_bilibili_input("BV1fKtN6DErG"),
                settings,
                backend=no_repeat,
                **arguments,
            )
            self.assertEqual(repeated["status"], "already_complete")
            self.assertEqual(no_repeat.calls, [])


if __name__ == "__main__":
    unittest.main()
