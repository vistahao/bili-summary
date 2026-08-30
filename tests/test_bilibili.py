import tempfile
import unittest
from pathlib import Path

from bili_summary.bilibili import BilibiliClient, NoSubtitleError, segments_to_srt
from bili_summary.inputs import parse_bilibili_input


class FakeBilibiliClient(BilibiliClient):
    def __init__(self, *, with_subtitle: bool = True) -> None:
        super().__init__()
        self.with_subtitle = with_subtitle

    def _get_json(self, url):  # type: ignore[no-untyped-def]
        if "web-interface/view" in url:
            return {
                "code": 0,
                "data": {
                    "aid": 10,
                    "title": "短课程",
                    "desc": "测试",
                    "duration": 4,
                    "owner": {"name": "讲者"},
                    "pubdate": 100,
                    "pages": [{"cid": 20, "page": 1, "part": "第一节", "duration": 4}],
                },
            }
        if "player/wbi/v2" in url:
            subtitles = []
            if self.with_subtitle:
                subtitles = [
                    {
                        "id": 30,
                        "lan": "zh-CN",
                        "lan_doc": "中文（自动生成）",
                        "subtitle_url": "//aisubtitle.hdslb.com/subtitle.json",
                        "type": 1,
                    }
                ]
            return {"code": 0, "data": {"subtitle": {"subtitles": subtitles}}}
        return {
            "body": [
                {"from": 0.125, "to": 1.5, "content": "第一句"},
                {"from": 1.5, "to": 3.75, "content": "第二句"},
            ]
        }


class BilibiliTests(unittest.TestCase):
    def test_fetches_platform_subtitle_and_formats_srt(self) -> None:
        transcript = FakeBilibiliClient().fetch_transcript(parse_bilibili_input("BV1fKtN6DErG"))
        self.assertEqual(transcript.video["title"], "短课程")
        self.assertEqual(len(transcript.segments), 2)
        srt = segments_to_srt(transcript.segments)
        self.assertIn("00:00:00,125 --> 00:00:01,500", srt)
        self.assertIn("第二句", srt)

    def test_stops_before_text_processing_when_no_subtitle(self) -> None:
        with self.assertRaises(NoSubtitleError):
            FakeBilibiliClient(with_subtitle=False).fetch_transcript(
                parse_bilibili_input("BV1fKtN6DErG")
            )

    def test_cookie_file_must_be_one_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie = Path(temp_dir) / "cookie.txt"
            cookie.write_text("SESSDATA=one\nsecond=line\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "只有一行"):
                BilibiliClient(cookie)

    def test_cookie_is_not_sent_to_subtitle_cdn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie = Path(temp_dir) / "cookie.txt"
            cookie.write_text("SESSDATA=secret", encoding="utf-8")
            client = BilibiliClient(cookie)
            self.assertIn("Cookie", client._headers("https://api.bilibili.com/x/player/v2"))
            self.assertNotIn(
                "Cookie", client._headers("https://aisubtitle.hdslb.com/subtitle.json")
            )


if __name__ == "__main__":
    unittest.main()
