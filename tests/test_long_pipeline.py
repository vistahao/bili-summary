import json
import tempfile
import unittest
from pathlib import Path

from bili_summary.adapters import CodexError, TextBackendError
from bili_summary.config import Settings
from bili_summary.inputs import parse_bilibili_input
from bili_summary.long_pipeline import (
    TranscriptChunk,
    _basic_prompt,
    _cached_invoke,
    _deep_prompt,
    run_bilibili_long_pipeline,
    split_transcript,
)
from bili_summary.models import StructuredResult, TextProfile, TranscriptSegment
from bili_summary.text_routing import resolve_text_plan
from test_bilibili import FakeBilibiliClient


class FakeLongBackend:
    def __init__(self, *, fail_on_summary: bool = False) -> None:
        self.calls = []
        self.prompts = []
        self.fail_on_summary = fail_on_summary

    def process(self, prompt, schema_path):  # type: ignore[no-untyped-def]
        self.calls.append(schema_path.name)
        self.prompts.append((schema_path.name, prompt))
        if schema_path.name == "organize_outputs.schema.json":
            payload = {
                "clean_markdown": "## 第一部分 [00:00:00]\n\n完整内容。",
                "warnings": [],
            }
        elif schema_path.name == "summary_notes.schema.json":
            payload = {"summary_notes_markdown": "- [00:00:00] 测试笔记", "warnings": []}
        elif schema_path.name == "basic_audit.schema.json":
            payload = {
                "audit_items": [
                    {
                        "category": "疑似字幕错误",
                        "timestamp": "[00:00:01]",
                        "quote": "原句",
                        "concern": "测试风险。",
                    }
                ],
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
            self.assertEqual(
                failing.calls,
                [
                    "basic_audit.schema.json",
                    "organize_outputs.schema.json",
                    "summary_notes.schema.json",
                    "summary_outputs.schema.json",
                ],
            )

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
            self.assertEqual(source["processing"]["text"]["calls"], 5)
            self.assertEqual(source["processing"]["text"]["calls_reused_this_run"], 3)
            cache_dir = Path(source["processing"]["strategy"]["cache_dir"])
            self.assertTrue((cache_dir / "organize-001.json").is_file())
            cached = (cache_dir / "organize-001.json").read_text(encoding="utf-8")
            self.assertNotIn("primary_chunk", cached)
            self.assertEqual(source["processing"]["text_plan"]["routes"]["organize"]["name"], "codex_default")

            no_repeat = FakeLongBackend()
            repeated = run_bilibili_long_pipeline(
                parse_bilibili_input("BV1fKtN6DErG"),
                settings,
                backend=no_repeat,
                **arguments,
            )
            self.assertEqual(repeated["status"], "already_complete")
            self.assertEqual(no_repeat.calls, [])

    def test_cache_fingerprint_separates_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "summary.json"
            schema_path = Path("schemas/summary_outputs.schema.json")
            backend = FakeLongBackend()
            codex = TextProfile("codex", "codex_exec", "model-a", "high")
            deepseek = TextProfile("deepseek", "deepseek_http", "model-b", "low")
            _first, first_reused = _cached_invoke(
                cache_path,
                "summary",
                codex,
                "prompt",
                schema_path,
                backend,
                force=False,
            )
            _second, second_reused = _cached_invoke(
                cache_path,
                "summary",
                deepseek,
                "prompt",
                schema_path,
                backend,
                force=False,
            )
            _third, third_reused = _cached_invoke(
                cache_path,
                "summary",
                deepseek,
                "prompt",
                schema_path,
                backend,
                force=False,
            )
            self.assertEqual((first_reused, second_reused, third_reused), (False, False, True))
            self.assertEqual(backend.calls, ["summary_outputs.schema.json"] * 2)

    def test_audit_prompts_filter_style_and_colloquial_guesses(self) -> None:
        segment = TranscriptSegment(0, 2000, "这是一个口语化测试。")
        chunk = TranscriptChunk(index=1, segments=(segment,))
        for prompt in (_basic_prompt(chunk, 1, {}), _deep_prompt(chunk, 1, {})):
            self.assertIn("知识", prompt)
            self.assertIn("逻辑", prompt)
            self.assertIn("不要报告口语表达", prompt)
            self.assertIn("不要猜测讲者", prompt)
            self.assertIn("疑似知识或逻辑错误", prompt)

    def test_rejects_audit_category_outside_knowledge_scope(self) -> None:
        from bili_summary.long_pipeline import _validate_audit_payload

        chunk = TranscriptChunk(
            index=1,
            segments=(TranscriptSegment(0, 2000, "测试。"),),
        )
        payload = {
            "audit_items": [
                {
                    "category": "口语不严谨",
                    "timestamp": "[00:00:01]",
                    "quote": "测试",
                    "concern": "只是文风判断。",
                }
            ],
            "warnings": [],
        }
        with self.assertRaisesRegex(TextBackendError, "无效类别"):
            _validate_audit_payload(payload, chunk, "audit_items")

    def test_routes_each_task_to_its_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_profile = TextProfile("codex_profile", "codex_exec", "codex-test", "high")
            deepseek_profile = TextProfile(
                "deepseek_profile", "deepseek_http", "deepseek-test", "low"
            )
            routes = {
                "organize": "codex_profile",
                "summary": "deepseek_profile",
                "basic_audit": "codex_profile",
                "deep_audit": "deepseek_profile",
            }
            settings = Settings(
                data_root=Path(temp_dir),
                text_profiles={
                    codex_profile.name: codex_profile,
                    deepseek_profile.name: deepseek_profile,
                },
                text_routes=routes,
            )
            codex_backend = FakeLongBackend()
            deepseek_backend = FakeLongBackend()
            result = run_bilibili_long_pipeline(
                parse_bilibili_input("BV1fKtN6DErG"),
                settings,
                subject="测试",
                course=None,
                title_override=None,
                schemas_dir=Path("schemas"),
                compare_deep=False,
                audit_level="basic",
                client=FakeBilibiliClient(),
                backends={
                    codex_profile.name: codex_backend,
                    deepseek_profile.name: deepseek_backend,
                },
                plan_selector=lambda _preview: resolve_text_plan(
                    settings, audit_level="basic"
                ),
            )
            self.assertEqual(
                codex_backend.calls,
                ["basic_audit.schema.json", "organize_outputs.schema.json"],
            )
            self.assertEqual(
                deepseek_backend.calls,
                ["summary_notes.schema.json", "summary_outputs.schema.json"],
            )
            source = json.loads(
                (Path(result["output_dir"]) / "source.json").read_text(encoding="utf-8")
            )
            self.assertEqual(source["processing"]["text"]["by_task"]["summary"]["calls"], 2)
            self.assertEqual(source["processing"]["content_mode"], "lecture")
            self.assertNotIn("secret", json.dumps(source).lower())

            organize_prompt = next(
                prompt
                for schema, prompt in codex_backend.prompts
                if schema == "organize_outputs.schema.json"
            )
            notes_prompt = next(
                prompt
                for schema, prompt in deepseek_backend.prompts
                if schema == "summary_notes.schema.json"
            )
            final_prompt = next(
                prompt
                for schema, prompt in deepseek_backend.prompts
                if schema == "summary_outputs.schema.json"
            )
            self.assertIn("测试风险", organize_prompt)
            self.assertIn("完整内容", notes_prompt)
            self.assertNotIn("<subtitle_chunk>", notes_prompt)
            self.assertIn("测试风险", final_prompt)

    def test_practice_mode_defines_learning_scope_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_root=Path(temp_dir))
            backend = FakeLongBackend()
            result = run_bilibili_long_pipeline(
                parse_bilibili_input("BV1fKtN6DErG"),
                settings,
                subject="测试",
                course=None,
                title_override=None,
                schemas_dir=Path("schemas"),
                compare_deep=False,
                audit_level="basic",
                content_mode="practice",
                client=FakeBilibiliClient(),
                backend=backend,
            )
            organize_prompt = next(
                prompt
                for schema, prompt in backend.prompts
                if schema == "organize_outputs.schema.json"
            )
            notes_prompt = next(
                prompt
                for schema, prompt in backend.prompts
                if schema == "summary_notes.schema.json"
            )
            final_prompt = next(
                prompt
                for schema, prompt in backend.prompts
                if schema == "summary_outputs.schema.json"
            )
            self.assertIn("本次内容模式是 practice", organize_prompt)
            self.assertIn("不得复述或解释歌词", organize_prompt)
            self.assertIn("题意、答案、教师推理、选项辨析", organize_prompt)
            self.assertIn("不得从原始字幕补回", notes_prompt)
            self.assertIn("不要为歌曲", final_prompt)
            source = json.loads(
                (Path(result["output_dir"]) / "source.json").read_text(encoding="utf-8")
            )
            self.assertEqual(source["processing"]["content_mode"], "practice")
            self.assertEqual(source["processing"]["pipeline_version"], "content-aware-v1")

    def test_switching_to_practice_reuses_content_independent_audit_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_root=Path(temp_dir))
            arguments = {
                "subject": "测试",
                "course": None,
                "title_override": None,
                "schemas_dir": Path("schemas"),
                "compare_deep": False,
                "audit_level": "basic",
                "client": FakeBilibiliClient(),
            }
            first_backend = FakeLongBackend()
            run_bilibili_long_pipeline(
                parse_bilibili_input("BV1fKtN6DErG"),
                settings,
                content_mode="lecture",
                backend=first_backend,
                **arguments,
            )

            practice_backend = FakeLongBackend()
            result = run_bilibili_long_pipeline(
                parse_bilibili_input("BV1fKtN6DErG"),
                settings,
                content_mode="practice",
                backend=practice_backend,
                **arguments,
            )
            self.assertEqual(
                practice_backend.calls,
                [
                    "organize_outputs.schema.json",
                    "summary_notes.schema.json",
                    "summary_outputs.schema.json",
                ],
            )
            self.assertEqual(result["text"]["calls_reused_this_run"], 1)
            self.assertEqual(result["content_mode"], "practice")

    def test_missing_deepseek_key_stops_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            profile = TextProfile("deepseek", "deepseek_http", "deepseek-test", "low")
            settings = Settings(
                data_root=data_root,
                text_profiles={"deepseek": profile},
                text_routes={
                    task: "deepseek"
                    for task in ("organize", "summary", "basic_audit", "deep_audit")
                },
                deepseek_api_key_env="BILI_SUMMARY_TEST_KEY_NOT_SET",
            )
            with self.assertRaisesRegex(ValueError, "DeepSeek 配置需要密钥"):
                run_bilibili_long_pipeline(
                    parse_bilibili_input("BV1fKtN6DErG"),
                    settings,
                    subject="测试",
                    course=None,
                    title_override=None,
                    schemas_dir=Path("schemas"),
                    compare_deep=False,
                    audit_level="basic",
                    client=FakeBilibiliClient(),
                    plan_selector=lambda _preview: resolve_text_plan(
                        settings, audit_level="basic"
                    ),
                )
            self.assertEqual(list(data_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
