from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from .models import TextResult, TranscriptSegment


class Transcriber(Protocol):
    name: str

    def transcribe(self, source: Path) -> Sequence[TranscriptSegment]: ...


class TextBackend(Protocol):
    name: str

    def process(self, task: str, text: str) -> TextResult: ...


class MockTranscriber:
    name = "mock-transcriber"

    def transcribe(self, source: Path) -> Sequence[TranscriptSegment]:
        return (TranscriptSegment(0, 1000, f"模拟转写：{source.name}"),)


class MockTextBackend:
    name = "mock-text-backend"

    def process(self, task: str, text: str) -> TextResult:
        return TextResult(task=task, content=f"模拟结果：{text}")


def build_codex_exec_command(schema_path: Path) -> list[str]:
    """Build the future command without invoking Codex."""
    return [
        "codex",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema_path),
        "-",
    ]
