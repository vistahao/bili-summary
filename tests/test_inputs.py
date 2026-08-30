import tempfile
import unittest
from pathlib import Path

from bili_summary.inputs import (
    InputError,
    normalize_local_path,
    parse_bilibili_input,
    parse_local_mp4,
)


class BilibiliInputTests(unittest.TestCase):
    def test_parses_url_and_part(self) -> None:
        result = parse_bilibili_input(
            "https://www.bilibili.com/video/BV1fKtN6DErG/?p=3&spm_id_from=333"
        )
        self.assertEqual(result.metadata["bv_id"], "BV1fKtN6DErG")
        self.assertEqual(result.metadata["part"], 3)
        self.assertEqual(result.canonical, "https://www.bilibili.com/video/BV1fKtN6DErG?p=3")

    def test_parses_bare_bv(self) -> None:
        result = parse_bilibili_input("BV1fKtN6DErG")
        self.assertEqual(result.identity, "BV1fKtN6DErG:p1")

    def test_recognizes_short_link_without_network(self) -> None:
        result = parse_bilibili_input("https://b23.tv/abc123")
        self.assertEqual(result.source_type, "bilibili_short")
        self.assertTrue(result.metadata["needs_network"])

    def test_rejects_other_hosts(self) -> None:
        with self.assertRaises(InputError):
            parse_bilibili_input("https://example.com/video/BV1fKtN6DErG")


class LocalInputTests(unittest.TestCase):
    def test_normalizes_windows_path(self) -> None:
        result = normalize_local_path(r"D:\Downloads\课程\第一课.mp4")
        self.assertEqual(result, Path("/mnt/d/Downloads/课程/第一课.mp4"))

    def test_reads_metadata_and_hashes_small_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "课程 01.MP4"
            sample.write_bytes(b"stage-one-test")
            result = parse_local_mp4(str(sample), compute_hash=True)
            self.assertEqual(result.source_type, "local_mp4")
            self.assertEqual(result.display_title, "课程 01")
            self.assertEqual(result.metadata["hash_status"], "complete")
            self.assertTrue(result.identity.startswith("sha256:"))

    def test_rejects_non_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "课程.mkv"
            sample.write_bytes(b"test")
            with self.assertRaises(InputError):
                parse_local_mp4(str(sample))


if __name__ == "__main__":
    unittest.main()
