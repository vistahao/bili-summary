from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from bili_summary.config import Settings
from bili_summary.local_asr import LocalAsrError, run_whisper_cpp_transcription


class LocalAsrTests(unittest.TestCase):
    def test_requires_explicit_runtime_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio.wav"
            source.write_bytes(b"RIFF" + b"\0" * 64)
            with self.assertRaisesRegex(LocalAsrError, "尚未配置"):
                run_whisper_cpp_transcription(
                    source,
                    Settings(),
                    cache_root=Path(directory) / "cache",
                    source_identity="source",
                )

    def test_runs_cpu_only_and_reuses_srt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "whisper-cli"
            model = root / "ggml-small.bin"
            source = root / "audio.wav"
            binary.write_bytes(b"binary")
            model.write_bytes(b"model")
            source.write_bytes(b"RIFF" + b"\0" * 64)
            calls = []

            def runner(command, **_kwargs):  # type: ignore[no-untyped-def]
                calls.append(command)
                prefix = Path(command[command.index("-of") + 1])
                prefix.with_suffix(".srt").write_text(
                    "1\n00:00:00,000 --> 00:00:01,000\n本地字幕。\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            settings = Settings(
                local_asr_binary=binary,
                local_asr_model=model,
                local_asr_threads=6,
            )
            first = run_whisper_cpp_transcription(
                source,
                settings,
                cache_root=root / "cache",
                source_identity="source-sha",
                runner=runner,
            )
            second = run_whisper_cpp_transcription(
                source,
                settings,
                cache_root=root / "cache",
                source_identity="source-sha",
                runner=lambda *_args, **_kwargs: self.fail("不应重复执行"),
            )
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertIn("-ng", calls[0])
            self.assertEqual(calls[0][calls[0].index("-t") + 1], "6")


if __name__ == "__main__":
    unittest.main()
