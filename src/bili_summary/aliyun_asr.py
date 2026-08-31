from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import Settings


QWEN_FILETRANS_MODEL = "qwen3-asr-flash-filetrans"
PARAFORMER_V2_MODEL = "paraformer-v2"
COMPARISON_MODELS = (QWEN_FILETRANS_MODEL, PARAFORMER_V2_MODEL)
WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
PRICE_CNY_PER_SECOND = {
    QWEN_FILETRANS_MODEL: 0.00022,
    PARAFORMER_V2_MODEL: 0.00008,
}


class AliyunAsrError(RuntimeError):
    """A normalized Alibaba Cloud ASR error that is safe to display."""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = "aliyun_bailian"
        self.code = code
        self.retryable = retryable


class AliyunAsrClient:
    """Minimal HTTP client shared by the two asynchronous file ASR models."""

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str | None = None,
        timeout_seconds: int = 60,
        transport: Callable[[urllib.request.Request, float], bytes] | None = None,
    ) -> None:
        if not api_key.strip():
            raise AliyunAsrError("阿里云百炼 API Key 为空", code="missing_credentials")
        if workspace_id and not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
            raise AliyunAsrError("无效的阿里云 Workspace ID", code="invalid_workspace_id")
        self.api_key = api_key.strip()
        self.workspace_id = workspace_id
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _urlopen_bytes

    @property
    def base_url(self) -> str:
        if self.workspace_id:
            return f"https://{self.workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1"
        return "https://dashscope.aliyuncs.com/api/v1"

    def submit(
        self,
        *,
        model: str,
        file_url: str,
    ) -> dict[str, str]:
        if model not in COMPARISON_MODELS:
            raise AliyunAsrError(f"不支持的阿里云 ASR 模型：{model}", code="unsupported_model")
        if not file_url.startswith(("https://", "http://", "oss://")):
            raise AliyunAsrError("转写文件必须使用 HTTP、HTTPS 或临时 oss:// URL", code="invalid_url")
        if model == QWEN_FILETRANS_MODEL:
            input_payload: dict[str, Any] = {"file_url": file_url}
            parameters: dict[str, Any] = {
                "channel_id": [0],
                "enable_itn": False,
                "enable_words": True,
            }
        else:
            input_payload = {"file_urls": [file_url]}
            parameters = {
                "channel_id": [0],
                "diarization_enabled": False,
            }
        payload = {
            "model": model,
            "input": input_payload,
            "parameters": parameters,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        if file_url.startswith("oss://"):
            headers["X-DashScope-OssResourceResolve"] = "enable"
        request = urllib.request.Request(
            f"{self.base_url}/services/audio/asr/transcription",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response = self._request_json(request)
        output = response.get("output")
        if not isinstance(output, dict):
            raise AliyunAsrError("阿里云提交响应缺少 output", code="invalid_response")
        task_id = output.get("task_id")
        status = output.get("task_status")
        if not isinstance(task_id, str) or not task_id:
            raise AliyunAsrError("阿里云提交响应缺少 task_id", code="invalid_response")
        return {
            "task_id": task_id,
            "task_status": str(status or "UNKNOWN"),
            "request_id": str(response.get("request_id") or ""),
        }

    def fetch(self, task_id: str) -> dict[str, Any]:
        if not task_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            for character in task_id
        ):
            raise AliyunAsrError("无效的阿里云转写 task_id", code="invalid_task_id")
        request = urllib.request.Request(
            f"{self.base_url}/tasks/{urllib.parse.quote(task_id, safe='')}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        response = self._request_json(request)
        output = response.get("output")
        if not isinstance(output, dict) or not isinstance(output.get("task_status"), str):
            raise AliyunAsrError("阿里云查询响应缺少任务状态", code="invalid_response")
        return response

    def wait_for_completion(
        self,
        task_id: str,
        *,
        poll_seconds: float = 2.0,
        timeout_seconds: float = 900.0,
        sleep: Callable[[float], None] = time.sleep,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            response = self.fetch(task_id)
            output = response["output"]
            status = str(output["task_status"]).upper()
            if progress:
                progress(f"阿里云任务 {task_id}：{status}")
            if status == "SUCCEEDED":
                return response
            if status in {"FAILED", "UNKNOWN", "CANCELED"}:
                code = str(output.get("code") or "task_failed")
                message = str(output.get("message") or status)
                raise AliyunAsrError(
                    f"阿里云转写任务失败（{code}）：{message}",
                    code=code,
                    retryable=_retryable_code(code),
                )
            if time.monotonic() - started >= timeout_seconds:
                raise AliyunAsrError(
                    f"阿里云转写任务等待超过 {timeout_seconds:.0f} 秒",
                    code="timeout",
                    retryable=True,
                )
            sleep(poll_seconds)

    def download_transcription(self, task_response: Mapping[str, Any]) -> dict[str, Any]:
        output = task_response.get("output")
        if not isinstance(output, Mapping):
            raise AliyunAsrError("阿里云任务结果缺少 output", code="invalid_response")
        result = output.get("result")
        if isinstance(result, Mapping):
            transcription_url = result.get("transcription_url")
        else:
            results = output.get("results")
            first = results[0] if isinstance(results, list) and results else None
            if isinstance(first, Mapping) and str(first.get("subtask_status", "")).upper() == "FAILED":
                code = str(first.get("code") or "subtask_failed")
                message = str(first.get("message") or code)
                raise AliyunAsrError(
                    f"阿里云转写子任务失败（{code}）：{message}",
                    code=code,
                    retryable=_retryable_code(code),
                )
            transcription_url = first.get("transcription_url") if isinstance(first, Mapping) else None
        if not isinstance(transcription_url, str):
            raise AliyunAsrError("阿里云任务结果缺少安全的转写下载 URL", code="invalid_response")
        transcription_url = _secure_aliyun_result_url(transcription_url)
        request = urllib.request.Request(transcription_url, method="GET")
        return self._request_json(request)

    def _request_json(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            raw = self.transport(request, float(self.timeout_seconds))
        except urllib.error.HTTPError as exc:
            raise _http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise AliyunAsrError(
                f"阿里云网络请求失败：{exc.reason}",
                code="network_error",
                retryable=True,
            ) from exc
        except TimeoutError as exc:
            raise AliyunAsrError(
                f"阿里云请求超过 {self.timeout_seconds} 秒",
                code="timeout",
                retryable=True,
            ) from exc
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AliyunAsrError("阿里云返回内容不是有效 JSON", code="invalid_response") from exc
        if not isinstance(payload, dict):
            raise AliyunAsrError("阿里云返回结构必须是 JSON 对象", code="invalid_response")
        if payload.get("code"):
            code = str(payload["code"])
            message = str(payload.get("message") or code)
            raise AliyunAsrError(
                f"阿里云请求失败（{code}）：{message}",
                code=code,
                retryable=_retryable_code(code),
            )
        return payload


class AliyunTemporaryUploadClient:
    """Request model-scoped private upload policies without exposing their credentials."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: int = 60,
        transport: Callable[[urllib.request.Request, float], bytes] | None = None,
    ) -> None:
        if not api_key.strip():
            raise AliyunAsrError("阿里云百炼 API Key 为空", code="missing_credentials")
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _urlopen_bytes

    def check_model_upload_access(self, model: str) -> dict[str, Any]:
        response = self._request_policy(model)
        data = response["data"]
        return {
            "model": model,
            "upload_policy_available": True,
            "expire_in_seconds": _optional_positive_int(data.get("expire_in_seconds")),
            "max_file_size_mb": _optional_positive_int(data.get("max_file_size_mb")),
            "request_id": str(response.get("request_id") or ""),
        }

    def upload(self, model: str, source: Path) -> dict[str, Any]:
        if not source.is_file():
            raise AliyunAsrError(f"待上传音频不存在：{source}", code="missing_source")
        response = self._request_policy(model)
        data = response["data"]
        maximum_mb = _optional_positive_int(data.get("max_file_size_mb"))
        size_bytes = source.stat().st_size
        if maximum_mb is not None and size_bytes > maximum_mb * 1024 * 1024:
            raise AliyunAsrError(
                f"音频大小 {size_bytes} 字节超过临时上传上限 {maximum_mb} MiB",
                code="file_too_large",
            )
        upload_host = str(data["upload_host"])
        parsed_host = urllib.parse.urlparse(upload_host)
        if (
            parsed_host.scheme != "https"
            or not parsed_host.hostname
            or not parsed_host.hostname.endswith(".aliyuncs.com")
        ):
            raise AliyunAsrError("阿里云返回了不安全的上传地址", code="invalid_response")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", source.name) or "sample.wav"
        object_key = f"{str(data['upload_dir']).rstrip('/')}/{safe_name}"
        fields = [
            ("OSSAccessKeyId", str(data["oss_access_key_id"])),
            ("Signature", str(data["signature"])),
            ("policy", str(data["policy"])),
            ("x-oss-object-acl", str(data.get("x_oss_object_acl") or "private")),
            (
                "x-oss-forbid-overwrite",
                str(data.get("x_oss_forbid_overwrite") or "true"),
            ),
            ("key", object_key),
            ("success_action_status", "200"),
        ]
        boundary = f"bili-summary-{secrets.token_hex(16)}"
        body = _multipart_body(boundary, fields, source)
        request = urllib.request.Request(
            upload_host,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            self.transport(request, float(self.timeout_seconds))
        except urllib.error.HTTPError as exc:
            raise _http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise AliyunAsrError(
                f"阿里云临时音频上传失败：{exc.reason}",
                code="network_error",
                retryable=True,
            ) from exc
        return {
            "model": model,
            "file_url": f"oss://{object_key}",
            "size_bytes": size_bytes,
            "expires_after_hours": 48,
            "backup_policy": "do_not_backup",
        }

    def _request_policy(self, model: str) -> dict[str, Any]:
        if model not in COMPARISON_MODELS:
            raise AliyunAsrError(f"不支持的阿里云 ASR 模型：{model}", code="unsupported_model")
        query = urllib.parse.urlencode({"action": "getPolicy", "model": model})
        request = urllib.request.Request(
            f"https://dashscope.aliyuncs.com/api/v1/uploads?{query}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="GET",
        )
        try:
            raw = self.transport(request, float(self.timeout_seconds))
        except urllib.error.HTTPError as exc:
            raise _http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise AliyunAsrError(
                f"阿里云上传预检失败：{exc.reason}",
                code="network_error",
                retryable=True,
            ) from exc
        try:
            response = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AliyunAsrError("阿里云上传预检返回内容不是有效 JSON", code="invalid_response") from exc
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            code = (
                str(response.get("code") or "invalid_response")
                if isinstance(response, dict)
                else "invalid_response"
            )
            message = (
                str(response.get("message") or "响应缺少 data")
                if isinstance(response, dict)
                else "响应不是 JSON 对象"
            )
            raise AliyunAsrError(
                f"阿里云上传预检失败（{code}）：{message}",
                code=code,
                retryable=_retryable_code(code),
            )
        required = (
            "policy",
            "signature",
            "upload_dir",
            "upload_host",
            "oss_access_key_id",
        )
        if any(not data.get(key) for key in required):
            raise AliyunAsrError("阿里云上传预检响应缺少必要凭据字段", code="invalid_response")
        return response


def load_aliyun_asr_api_key(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    value = environment.get(settings.aliyun_asr_api_key_env, "").strip()
    if value:
        return value
    path = settings.aliyun_asr_api_key_file
    if path is None:
        raise ValueError(
            f"阿里云 ASR 需要密钥；请设置环境变量 {settings.aliyun_asr_api_key_env}，"
            "或在 Git 忽略的权限 600 文件中配置 api_key_file"
        )
    if not path.is_file():
        raise ValueError(f"阿里云 ASR API Key 文件不存在：{path}")
    if path.stat().st_mode & 0o077:
        raise ValueError(f"阿里云 ASR API Key 文件权限过宽：{path}；请设置为 600")
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("阿里云 ASR API Key 文件必须只有一行非空内容")
    return value


def estimate_comparison_cost_cny(duration_seconds: float) -> dict[str, float]:
    if duration_seconds < 0:
        raise ValueError("音频时长不能小于 0")
    costs = {
        model: round(duration_seconds * PRICE_CNY_PER_SECOND[model], 6)
        for model in COMPARISON_MODELS
    }
    costs["total"] = round(sum(costs.values()), 6)
    return costs


def _http_error(exc: urllib.error.HTTPError) -> AliyunAsrError:
    try:
        payload = json.loads(exc.read())
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    code = str(payload.get("code") or f"http_{exc.code}")
    message = str(payload.get("message") or exc.reason or "没有错误详情")
    if exc.code in {401, 403}:
        normalized = "authentication_failed"
    elif exc.code == 429:
        normalized = "rate_limited"
    else:
        normalized = code
    return AliyunAsrError(
        f"阿里云请求失败（HTTP {exc.code}，{code}）：{message}",
        code=normalized,
        retryable=exc.code == 429 or exc.code >= 500,
    )


def _retryable_code(code: str) -> bool:
    lowered = code.lower()
    return any(token in lowered for token in ("throttl", "timeout", "internal", "unavailable"))


def _optional_positive_int(value: Any) -> int | None:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _secure_aliyun_result_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or not parsed.hostname.endswith(".aliyuncs.com")
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise AliyunAsrError("阿里云任务结果缺少安全的转写下载 URL", code="invalid_response")
    # 北京旧端点目前可能返回 http:// 的签名下载地址；OSS 同一主机支持
    # HTTPS，因此在下载完整转写正文前强制升级传输协议。
    return urllib.parse.urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, ""))


def _multipart_body(
    boundary: str,
    fields: list[tuple[str, str]],
    source: Path,
) -> bytes:
    chunks: list[bytes] = []
    delimiter = f"--{boundary}\r\n".encode()
    for name, value in fields:
        chunks.extend(
            (
                delimiter,
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", source.name) or "sample.wav"
    content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    chunks.extend(
        (
            delimiter,
            (
                f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode(),
            source.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks)


def _urlopen_bytes(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()
