import unittest
from pathlib import Path

from bili_summary.adapters import MockTextBackend, MockTranscriber, build_codex_exec_command


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
        self.assertEqual(command[-1], "-")


if __name__ == "__main__":
    unittest.main()
