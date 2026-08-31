import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from bili_summary.media import (
    MediaError,
    extract_embedded_subtitle,
    find_external_srt,
    inspect_local_media,
    prepare_full_transcription_audio,
    prepare_transcription_sample,
    probe_media,
)


class MediaTests(unittest.TestCase):
    def test_external_srt_has_priority_over_embedded_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media = root / "课程.mp4"
            subtitle = root / "课程.srt"
            media.write_bytes(b"mp4")
            subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n测试\n", encoding="utf-8")
            result = inspect_local_media(media, runner=_fake_runner(_probe_payload()))
            self.assertEqual(find_external_srt(media), subtitle)
            self.assertEqual(result["text_source"]["kind"], "external_srt")
            self.assertEqual(result["probe"]["stream_counts"]["subtitle"], 1)

    def test_selects_extractable_embedded_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            media = Path(temp_dir) / "课程.mp4"
            media.write_bytes(b"mp4")
            result = inspect_local_media(media, runner=_fake_runner(_probe_payload()))
            self.assertEqual(result["text_source"]["kind"], "embedded_subtitle")
            self.assertEqual(result["text_source"]["stream_index"], 2)
            self.assertTrue(result["probe"]["read_only"])

    def test_reports_audio_required_without_subtitles(self) -> None:
        payload = _probe_payload()
        payload["streams"] = payload["streams"][:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            media = Path(temp_dir) / "课程.mp4"
            media.write_bytes(b"mp4")
            result = inspect_local_media(media, runner=_fake_runner(payload))
            self.assertEqual(result["text_source"]["kind"], "audio_transcription_required")

    def test_reports_unavailable_when_media_has_no_audio_or_subtitles(self) -> None:
        payload = _probe_payload()
        payload["streams"] = payload["streams"][:1]
        with tempfile.TemporaryDirectory() as temp_dir:
            media = Path(temp_dir) / "课程.mp4"
            media.write_bytes(b"mp4")
            result = inspect_local_media(media, runner=_fake_runner(payload))
            self.assertEqual(result["text_source"]["kind"], "unavailable_no_audio")

    def test_rejects_ambiguous_case_insensitive_external_subtitles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media = root / "Course.mp4"
            media.write_bytes(b"mp4")
            (root / "Course.srt").write_text("one", encoding="utf-8")
            (root / "course.SRT").write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(MediaError, "多个同名 SRT"):
                find_external_srt(media)

    def test_prepares_and_reuses_lossless_sample_with_five_day_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_path = root / "课程.mp4"
            media_path.write_bytes(b"source")
            cache_root = root / "cache"
            inspected = inspect_local_media(
                media_path,
                runner=_fake_runner(_audio_only_probe_payload()),
            )
            calls = []

            def extract_runner(command, **_kwargs):  # type: ignore[no-untyped-def]
                calls.append(command)
                Path(command[-1]).write_bytes(b"RIFF-test-audio")
                return subprocess.CompletedProcess(command, 0, "", "")

            first = prepare_transcription_sample(
                media_path,
                media=inspected,
                source_sha256="a" * 64,
                cache_root=cache_root,
                start_seconds=30,
                duration_seconds=300,
                runner=extract_runner,
                now=datetime(2026, 8, 31, 4, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(first["reused"])
            self.assertEqual(first["eligible_for_cleanup_at"], "2026-09-05T04:00:00Z")
            self.assertEqual(first["audio"]["sample_rate_hz"], 16_000)
            self.assertEqual(first["audio"]["channels"], 1)
            self.assertIn("pcm_s16le", calls[0])
            self.assertTrue(Path(first["audio"]["path"]).is_file())

            def should_not_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("matching sample should be reused")

            second = prepare_transcription_sample(
                media_path,
                media=inspected,
                source_sha256="a" * 64,
                cache_root=cache_root,
                start_seconds=30,
                duration_seconds=300,
                runner=should_not_run,
                now=datetime(2026, 9, 1, 4, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(second["reused"])
            self.assertEqual(second["eligible_for_cleanup_at"], "2026-09-06T04:00:00Z")

    def test_does_not_prepare_audio_when_subtitles_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_path = root / "课程.mp4"
            media_path.write_bytes(b"source")
            subtitle_path = root / "课程.srt"
            subtitle_path.write_text("subtitle", encoding="utf-8")
            inspected = inspect_local_media(
                media_path,
                runner=_fake_runner(_audio_only_probe_payload()),
            )
            with self.assertRaisesRegex(MediaError, "已有可用字幕"):
                prepare_transcription_sample(
                    media_path,
                    media=inspected,
                    source_sha256="b" * 64,
                    cache_root=root / "cache",
                    duration_seconds=300,
                )

    def test_prepares_and_reuses_complete_audio_without_copying_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_path = root / "课程.mp4"
            media_path.write_bytes(b"source-video")
            inspected = inspect_local_media(
                media_path,
                runner=_fake_runner(_audio_only_probe_payload()),
            )
            calls = []

            def extract_runner(command, **_kwargs):  # type: ignore[no-untyped-def]
                calls.append(command)
                Path(command[-1]).write_bytes(b"RIFF" + b"\0" * 64)
                return subprocess.CompletedProcess(command, 0, "", "")

            first = prepare_full_transcription_audio(
                media_path,
                media=inspected,
                source_sha256="c" * 64,
                cache_root=root / "cache",
                runner=extract_runner,
                now=datetime(2026, 8, 31, 4, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(first["reused"])
            self.assertEqual(first["audio"]["duration_seconds"], 900.0)
            self.assertNotIn("-t", calls[0])
            self.assertEqual(media_path.read_bytes(), b"source-video")

            second = prepare_full_transcription_audio(
                media_path,
                media=inspected,
                source_sha256="c" * 64,
                cache_root=root / "cache",
                runner=lambda *_args, **_kwargs: self.fail("不应重复提取"),
                now=datetime(2026, 9, 1, 4, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(second["reused"])
            self.assertEqual(second["eligible_for_cleanup_at"], "2026-09-06T04:00:00Z")

    def test_extracts_and_reuses_embedded_srt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_path = root / "课程.mp4"
            media_path.write_bytes(b"source")
            calls = []

            def extract_runner(command, **_kwargs):  # type: ignore[no-untyped-def]
                calls.append(command)
                Path(command[-1]).write_text(
                    "1\n00:00:00,000 --> 00:00:01,000\n测试\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            first = extract_embedded_subtitle(
                media_path,
                stream_index=2,
                source_sha256="d" * 64,
                cache_root=root / "cache",
                runner=extract_runner,
            )
            self.assertFalse(first["reused"])
            self.assertIn("0:2", calls[0])
            second = extract_embedded_subtitle(
                media_path,
                stream_index=2,
                source_sha256="d" * 64,
                cache_root=root / "cache",
                runner=lambda *_args, **_kwargs: self.fail("不应重复提取"),
            )
            self.assertTrue(second["reused"])

    def test_rejects_failed_or_invalid_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            media = Path(temp_dir) / "课程.mp4"
            media.write_bytes(b"mp4")
            failed = lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "bad media")
            with self.assertRaisesRegex(MediaError, "bad media"):
                probe_media(media, runner=failed)
            invalid = lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "not json", "")
            with self.assertRaisesRegex(MediaError, "有效 JSON"):
                probe_media(media, runner=invalid)


def _probe_payload() -> dict:  # type: ignore[type-arg]
    return {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "12.5",
            "size": "12345",
            "bit_rate": "98765",
        },
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac"},
            {
                "index": 2,
                "codec_type": "subtitle",
                "codec_name": "mov_text",
                "tags": {"language": "chi", "title": "中文"},
                "disposition": {"default": 1},
            },
        ],
    }


def _audio_only_probe_payload() -> dict:  # type: ignore[type-arg]
    payload = _probe_payload()
    payload["streams"] = payload["streams"][:2]
    payload["format"]["duration"] = "900.0"
    return payload


def _fake_runner(payload: dict):  # type: ignore[type-arg]
    def runner(command, **_kwargs):  # type: ignore[no-untyped-def]
        if command[1:3] != ["-v", "error"]:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    return runner


if __name__ == "__main__":
    unittest.main()
