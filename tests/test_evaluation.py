import json
import tempfile
import unittest
from pathlib import Path

from bili_summary.config import Settings
from bili_summary.evaluation import parse_srt, run_text_profile_comparison
from bili_summary.models import StructuredResult, TextProfile


class FakeEvaluationBackend:
    def __init__(self) -> None:
        self.calls = []

    def process(self, _prompt, schema_path):  # type: ignore[no-untyped-def]
        self.calls.append(schema_path.name)
        if schema_path.name == "organize_outputs.schema.json":
            payload = {"clean_markdown": "## 整理 [00:00:00]\n\n正文。", "warnings": []}
        elif schema_path.name == "summary_outputs.schema.json":
            payload = {"summary_markdown": "# 学习总结\n\n总结。", "warnings": []}
        elif schema_path.name == "basic_audit.schema.json":
            payload = {"audit_items": [], "warnings": []}
        elif schema_path.name == "deep_audit.schema.json":
            payload = {"audit_items": [], "coverage_statement": "覆盖全部。", "warnings": []}
        else:
            raise AssertionError(schema_path)
        return StructuredResult(
            payload=payload,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            elapsed_seconds=0.1,
            backend_metadata={
                "provider": "test",
                "model": "test-model",
                "reasoning": "low",
                "estimated_cost_usd": "0.0001",
            },
        )


class EvaluationTests(unittest.TestCase):
    def test_parses_srt_and_creates_resumable_comparison(self) -> None:
        srt = "1\n00:00:00,000 --> 00:00:02,000\n测试字幕。\n"
        segments = parse_srt(srt)
        self.assertEqual(segments[0].end_ms, 2000)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "字幕.srt"
            source.write_text(srt, encoding="utf-8")
            profile = TextProfile("test_profile", "deepseek_http", "test-model", "low")
            settings = Settings(
                data_root=root / "data",
                text_profiles={profile.name: profile},
            )
            backend = FakeEvaluationBackend()
            arguments = {
                "profile_names": (profile.name,),
                "schemas_dir": Path("schemas"),
                "backends": {profile.name: backend},
            }
            first = run_text_profile_comparison(source, settings, **arguments)
            self.assertEqual(first["evaluations"], 4)
            self.assertEqual(first["calls"], 4)
            self.assertEqual(first["reused"], 0)
            output_dir = Path(first["output_dir"])
            metrics = json.loads((output_dir / "对比指标.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics["results"]), 4)
            self.assertEqual(metrics["schema_version"], 2)
            self.assertEqual(metrics["profile_totals"][0]["input"], 40)
            self.assertEqual(metrics["profile_totals"][0]["output"], 20)
            self.assertEqual(metrics["profile_totals"][0]["cost"], "0.00040000")
            self.assertGreater(metrics["profile_totals"][0]["characters"], 0)
            self.assertEqual(metrics["profile_totals"][0]["audit_items"], 0)
            self.assertNotIn("reasoning_content", json.dumps(metrics))
            index = (output_dir / "对比说明.md").read_text(encoding="utf-8")
            self.assertIn("## 配置合计", index)
            self.assertIn("test_profile | 40 | 20", index)

            second_backend = FakeEvaluationBackend()
            second = run_text_profile_comparison(
                source,
                settings,
                profile_names=(profile.name,),
                schemas_dir=Path("schemas"),
                backends={profile.name: second_backend},
            )
            self.assertEqual(second["output_dir"], first["output_dir"])
            self.assertEqual(second["evaluations"], 4)
            self.assertEqual(second["calls"], 0)
            self.assertEqual(second["reused"], 4)
            self.assertEqual(second_backend.calls, [])


if __name__ == "__main__":
    unittest.main()
