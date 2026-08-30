from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .inputs import InputError, parse_bilibili_input
from .models import InputSpec, PlatformTranscript, TranscriptSegment


API_ROOT = "https://api.bilibili.com"
ALLOWED_REDIRECT_HOSTS = {"b23.tv", "www.b23.tv", "bilibili.com", "www.bilibili.com", "m.bilibili.com"}


class BilibiliError(RuntimeError):
    """Raised when public Bilibili metadata or subtitles cannot be obtained."""


class NoSubtitleError(BilibiliError):
    """Raised before Codex is called when no platform subtitle is visible."""


class BilibiliClient:
    def __init__(self, cookie_file: Path | None = None, *, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds
        self.cookie = _read_cookie(cookie_file)

    @property
    def authenticated(self) -> bool:
        return bool(self.cookie)

    def fetch_transcript(self, spec: InputSpec) -> PlatformTranscript:
        normalized = self._resolve_short_link(spec) if spec.source_type == "bilibili_short" else spec
        bvid = str(normalized.metadata["bv_id"])
        part = int(normalized.metadata["part"])
        view = self._get_json(
            f"{API_ROOT}/x/web-interface/view?{urllib.parse.urlencode({'bvid': bvid})}"
        )
        data = _api_data(view, "获取视频元数据")
        pages = data.get("pages")
        if not isinstance(pages, list) or not pages:
            raise BilibiliError("视频元数据没有可处理的分P")
        if part > len(pages):
            raise BilibiliError(f"请求第 {part} P，但视频只有 {len(pages)} P")
        page = pages[part - 1]
        cid = page.get("cid")
        player = self._get_json(
            f"{API_ROOT}/x/player/wbi/v2?"
            + urllib.parse.urlencode({"bvid": bvid, "cid": cid})
        )
        player_data = _api_data(player, "获取播放器字幕列表")
        tracks = player_data.get("subtitle", {}).get("subtitles", [])
        if not tracks:
            login_hint = "当前已使用 Cookie，但该视频仍未返回字幕" if self.authenticated else "公开接口未返回字幕；AI 字幕可能需要登录 Cookie"
            raise NoSubtitleError(f"{login_hint}。流程已在调用 Codex 前停止")
        track = _select_subtitle(tracks)
        subtitle_url = _normalize_subtitle_url(track.get("subtitle_url"))
        raw_subtitle = self._get_json(subtitle_url)
        segments = _parse_segments(raw_subtitle)
        if not segments:
            raise NoSubtitleError("字幕文件存在，但没有有效句段；流程已在调用 Codex 前停止")

        safe_video = {
            "bvid": bvid,
            "aid": data.get("aid"),
            "title": data.get("title"),
            "description": data.get("desc"),
            "duration_seconds": data.get("duration"),
            "owner": (data.get("owner") or {}).get("name"),
            "published_at": data.get("pubdate"),
            "page_count": len(pages),
        }
        safe_page = {
            "part_number": part,
            "cid": cid,
            "title": page.get("part"),
            "duration_seconds": page.get("duration"),
        }
        safe_subtitle = {
            key: track.get(key)
            for key in ("id", "lan", "lan_doc", "type", "ai_type", "ai_status")
            if key in track
        }
        safe_subtitle["source"] = "bilibili_platform"
        return PlatformTranscript(
            input_spec=normalized,
            video=safe_video,
            page=safe_page,
            subtitle=safe_subtitle,
            raw_subtitle=raw_subtitle,
            segments=segments,
        )

    def _resolve_short_link(self, spec: InputSpec) -> InputSpec:
        request = urllib.request.Request(spec.canonical, headers=self._headers(spec.canonical))
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                final_url = response.geturl()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BilibiliError(f"短链接解析失败：{exc}") from exc
        host = (urllib.parse.urlsplit(final_url).hostname or "").lower()
        if host not in ALLOWED_REDIRECT_HOSTS:
            raise BilibiliError(f"短链接跳转到了不允许的域名：{host or '未知'}")
        try:
            return parse_bilibili_input(final_url)
        except InputError as exc:
            raise BilibiliError(f"短链接目标无有效 BV 号：{final_url}") from exc

    def _get_json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers=self._headers(url))
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise BilibiliError(f"哔哩哔哩请求失败：HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BilibiliError(f"哔哩哔哩网络请求失败：{exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BilibiliError("哔哩哔哩返回了无法解析的 JSON") from exc
        if not isinstance(payload, dict):
            raise BilibiliError("哔哩哔哩返回的 JSON 不是对象")
        return payload

    def _headers(self, url: str) -> dict[str, str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
            "Accept": "application/json,text/plain,*/*",
        }
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if self.cookie and (host == "bilibili.com" or host.endswith(".bilibili.com")):
            headers["Cookie"] = self.cookie
        return headers


def segments_to_srt(segments: tuple[TranscriptSegment, ...]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.replace("\r\n", "\n").replace("\r", "\n").strip()
        blocks.append(
            f"{index}\n{_srt_timestamp(segment.start_ms)} --> {_srt_timestamp(segment.end_ms)}\n{text}"
        )
    return "\n\n".join(blocks) + "\n"


def _read_cookie(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise BilibiliError(f"配置的 Cookie 文件不存在：{path}") from exc
    if not value:
        raise BilibiliError(f"配置的 Cookie 文件为空：{path}")
    if "\r" in value or "\n" in value:
        raise BilibiliError("Cookie 文件必须只有一行")
    if len(value) > 16384:
        raise BilibiliError("Cookie 文件异常大，已拒绝读取")
    return value


def _api_data(payload: dict[str, Any], action: str) -> dict[str, Any]:
    if payload.get("code") != 0:
        raise BilibiliError(f"{action}失败：{payload.get('message') or payload.get('code')}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise BilibiliError(f"{action}失败：响应缺少 data")
    return data


def _select_subtitle(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    chinese = [track for track in tracks if str(track.get("lan", "")).lower().startswith("zh")]
    return chinese[0] if chinese else tracks[0]


def _normalize_subtitle_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise BilibiliError("字幕轨没有下载地址")
    if value.startswith("//"):
        value = "https:" + value
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https":
        raise BilibiliError("字幕地址不是 HTTPS")
    host = (parsed.hostname or "").lower()
    trusted = (
        host == "hdslb.com"
        or host.endswith(".hdslb.com")
        or host == "bilibili.com"
        or host.endswith(".bilibili.com")
    )
    if not trusted:
        raise BilibiliError(f"字幕地址不属于可信的 B 站域名：{host or '未知'}")
    return value


def _parse_segments(payload: dict[str, Any]) -> tuple[TranscriptSegment, ...]:
    body = payload.get("body")
    if not isinstance(body, list):
        raise BilibiliError("字幕 JSON 缺少 body 数组")
    result = []
    for item in body:
        if not isinstance(item, dict):
            continue
        text = str(item.get("content", "")).strip()
        if not text:
            continue
        try:
            start_ms = max(0, round(float(item["from"]) * 1000))
            end_ms = max(start_ms + 1, round(float(item["to"]) * 1000))
        except (KeyError, TypeError, ValueError):
            continue
        result.append(TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text=text))
    return tuple(result)


def _srt_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
