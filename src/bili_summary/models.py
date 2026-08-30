from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class InputSpec:
    source_type: str
    original: str
    canonical: str
    display_title: str
    identity: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class TextResult:
    task: str
    content: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlatformTranscript:
    input_spec: InputSpec
    video: dict[str, Any]
    page: dict[str, Any]
    subtitle: dict[str, Any]
    raw_subtitle: dict[str, Any]
    segments: tuple[TranscriptSegment, ...]


@dataclass(frozen=True)
class StudyOutputs:
    clean_transcript_markdown: str
    audit_markdown: str
    summary_markdown: str
    warnings: tuple[str, ...]
    usage: dict[str, int]
    call_count: int
    elapsed_seconds: float
    backend_metadata: dict[str, str]


@dataclass(frozen=True)
class StructuredResult:
    payload: dict[str, Any]
    usage: dict[str, int]
    elapsed_seconds: float
    backend_metadata: dict[str, str]
