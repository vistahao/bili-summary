import unittest
from pathlib import Path

from bili_summary.naming import build_archive_path, sanitize_component


class NamingTests(unittest.TestCase):
    def test_sanitizes_windows_invalid_characters(self) -> None:
        self.assertEqual(sanitize_component('第一课: "输入/输出"'), "第一课- -输入-输出-")

    def test_protects_reserved_name(self) -> None:
        self.assertEqual(sanitize_component("CON"), "_CON")

    def test_builds_subject_course_title_path(self) -> None:
        result = build_archive_path(
            Path("/data"), subject="公考事业编", course="行测", title="片段刷题4"
        )
        self.assertEqual(result, Path("/data/公考事业编/行测/片段刷题4"))


if __name__ == "__main__":
    unittest.main()
