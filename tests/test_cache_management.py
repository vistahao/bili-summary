from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from bili_summary.cache_management import clean_cache, inspect_cache


NOW = datetime(2026, 9, 6, 0, 0, tzinfo=timezone.utc)


def _managed_audio(root: Path, *, eligible_at: str = "2026-09-05T00:00:00Z") -> tuple[Path, Path]:
    media = root / "local-0123456789abcdef" / "media"
    media.mkdir(parents=True)
    audio = media / "transcription-full.wav"
    audio.write_bytes(b"RIFF" + b"x" * 100)
    metadata = media / "transcription-full.json"
    metadata.write_text(
        json.dumps(
            {
                "kind": "full_transcription_audio",
                "backup_policy": "do_not_backup",
                "cleanup_policy": "eligible_after_last_successful_use; explicit cleanup only",
                "last_successful_use_at": "2026-08-31T00:00:00Z",
                "eligible_for_cleanup_at": eligible_at,
                "audio": {"path": str(audio), "size_bytes": audio.stat().st_size},
            }
        ),
        encoding="utf-8",
    )
    return audio, metadata


class CacheManagementTests(unittest.TestCase):
    def test_inventory_reports_managed_and_unmanaged_bytes_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            audio, metadata = _managed_audio(root)
            other = root / "local-0123456789abcdef" / "asr" / "原始响应.json"
            other.parent.mkdir(parents=True)
            other.write_bytes(b"other-cache")

            result = inspect_cache(root, now=NOW)

            self.assertEqual(result["status"], "cache_inventory")
            self.assertEqual(result["managed_audio_bytes"], audio.stat().st_size)
            self.assertGreaterEqual(result["other_cache_bytes"], other.stat().st_size)
            self.assertEqual(result["eligible_bytes"], audio.stat().st_size)
            self.assertTrue(result["managed_audio"][0]["eligible_for_cleanup"])
            self.assertTrue(audio.is_file())
            self.assertTrue(metadata.is_file())

    def test_cleanup_preview_is_read_only_and_execute_preserves_other_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            audio, metadata = _managed_audio(root)
            other = root / "local-0123456789abcdef" / "asr" / "原始响应.json"
            other.parent.mkdir(parents=True)
            other.write_bytes(b"keep")

            preview = clean_cache(root, execute=False, now=NOW)
            self.assertEqual(preview["eligible_items"], 1)
            self.assertTrue(audio.is_file())
            self.assertTrue(metadata.is_file())

            completed = clean_cache(root, execute=True, now=NOW)
            self.assertEqual(completed["deleted_items"], 1)
            self.assertFalse(audio.exists())
            self.assertFalse(metadata.exists())
            self.assertTrue(other.is_file())

    def test_not_yet_eligible_audio_is_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            audio, metadata = _managed_audio(root, eligible_at="2026-09-07T00:00:00Z")

            result = clean_cache(root, execute=True, now=NOW)

            self.assertEqual(result["deleted_items"], 0)
            self.assertTrue(audio.is_file())
            self.assertTrue(metadata.is_file())

    def test_path_mismatch_is_reported_and_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            audio, metadata = _managed_audio(root)
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["audio"]["path"] = str(Path(directory) / "outside.wav")
            metadata.write_text(json.dumps(payload), encoding="utf-8")

            result = clean_cache(root, execute=True, now=NOW)

            self.assertEqual(result["deleted_items"], 0)
            self.assertTrue(audio.is_file())
            self.assertTrue(metadata.is_file())
            self.assertTrue(any("路径" in warning for warning in result["warnings"]))

    def test_symlinked_audio_is_rejected_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            audio, metadata = _managed_audio(root)
            outside = Path(directory) / "outside.wav"
            outside.write_bytes(b"keep-outside")
            audio.unlink()
            audio.symlink_to(outside)

            result = clean_cache(root, execute=True, now=NOW)

            self.assertEqual(result["deleted_items"], 0)
            self.assertTrue(audio.is_symlink())
            self.assertEqual(outside.read_bytes(), b"keep-outside")
            self.assertTrue(metadata.is_file())
            self.assertTrue(any("符号链接" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
