import json
import unittest
from pathlib import Path

from bili_summary.adapters import (
    MockTextBackend,
    MockTranscriber,
    build_codex_exec_command,
    parse_codex_jsonl,
)


class AdapterTests(unittest.TestCase):
    def test_mock_backends_are_deterministic(self) -> None:
        segments = MockTranscriber().transcribe(Path("sample.mp4"))
        self.assertEqual(segments[0].start_ms, 0)
        self.assertIn("sample.mp4", segments[0].text)
        result = MockTextBackend().process("summary", "hello")
        self.assertEqual(result.task, "summary")
        self.assertEqual(result.content, "模拟结果：hello")

    def test_codex_command_is_read_only_and_ephemeral(self) -> None:
        command = build_codex_exec_command(Path("schema.json"))
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertIn("--ephemeral", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertIn("--json", command)
        self.assertEqual(command[-1], "-")

        explicit = build_codex_exec_command(Path("schema.json"), model="test-model")
        self.assertEqual(explicit[explicit.index("--model") + 1], "test-model")

    def test_parses_structured_codex_events_and_usage(self) -> None:
        final = {
            "clean_transcript_markdown": "# 整理稿",
            "audit_markdown": "# 审校",
            "summary_markdown": "# 总结",
            "warnings": [],
        }
        events = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "test"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": json.dumps(final)},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 123, "output_tokens": 45},
                    }
                ),
            ]
        )
        payload, usage = parse_codex_jsonl(events)
        self.assertEqual(payload["summary_markdown"], "# 总结")
        self.assertEqual(usage["input_tokens"], 123)


if __name__ == "__main__":
    unittest.main()
