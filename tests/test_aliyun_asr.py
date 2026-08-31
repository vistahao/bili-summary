import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from bili_summary.aliyun_asr import (
    AliyunAsrClient,
    AliyunAsrError,
    AliyunTemporaryUploadClient,
    PARAFORMER_V2_MODEL,
    QWEN_FILETRANS_MODEL,
    estimate_comparison_cost_cny,
    load_aliyun_asr_api_key,
)
from bili_summary.config import Settings


class AliyunAsrTests(unittest.TestCase):
    def test_both_models_use_same_async_endpoint_without_leaking_key(self) -> None:
        requests = []

        def transport(request, _timeout):  # type: ignore[no-untyped-def]
            requests.append(request)
            return json.dumps(
                {
                    "output": {"task_status": "PENDING", "task_id": "task-123"},
                    "request_id": "request-123",
                }
            ).encode()

        client = AliyunAsrClient(
            api_key="secret-test-key",
            workspace_id="llm-test",
            transport=transport,
        )
        for model in (QWEN_FILETRANS_MODEL, PARAFORMER_V2_MODEL):
            result = client.submit(model=model, file_url="https://example.test/sample.wav")
            self.assertEqual(result["task_id"], "task-123")

        for request in requests:
            self.assertEqual(request.method, "POST")
            self.assertEqual(
                request.full_url,
                "https://llm-test.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/asr/transcription",
            )
            payload = json.loads(request.data)
            self.assertIn(payload["model"], (QWEN_FILETRANS_MODEL, PARAFORMER_V2_MODEL))
            self.assertEqual(payload["parameters"]["channel_id"], [0])
            self.assertNotIn("secret-test-key", request.full_url)
            self.assertNotIn("secret-test-key", request.data.decode())

        qwen_payload = json.loads(requests[0].data)
        self.assertEqual(qwen_payload["input"], {"file_url": "https://example.test/sample.wav"})
        self.assertTrue(qwen_payload["parameters"]["enable_words"])
        paraformer_payload = json.loads(requests[1].data)
        self.assertEqual(
            paraformer_payload["input"],
            {"file_urls": ["https://example.test/sample.wav"]},
        )
        self.assertFalse(paraformer_payload["parameters"]["diarization_enabled"])

    def test_qwen_temporary_url_enables_oss_resolution(self) -> None:
        captured = []

        def transport(request, _timeout):  # type: ignore[no-untyped-def]
            captured.append(request)
            return b'{"output":{"task_status":"PENDING","task_id":"task-1"}}'

        client = AliyunAsrClient(api_key="secret", transport=transport)
        client.submit(
            model=QWEN_FILETRANS_MODEL,
            file_url="oss://dashscope-instant/sample.wav",
        )
        request = captured[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], QWEN_FILETRANS_MODEL)
        self.assertEqual(payload["input"]["file_url"], "oss://dashscope-instant/sample.wav")
        self.assertEqual(request.get_header("X-dashscope-ossresourceresolve"), "enable")

    def test_fetch_validates_and_parses_task_status(self) -> None:
        def transport(request, _timeout):  # type: ignore[no-untyped-def]
            self.assertEqual(request.method, "GET")
            return b'{"output":{"task_status":"SUCCEEDED","task_id":"task-1"}}'

        client = AliyunAsrClient(api_key="secret", transport=transport)
        result = client.fetch("task-1")
        self.assertEqual(result["output"]["task_status"], "SUCCEEDED")
        with self.assertRaisesRegex(AliyunAsrError, "task_id"):
            client.fetch("../secret")
        with self.assertRaisesRegex(AliyunAsrError, "Workspace"):
            AliyunAsrClient(api_key="secret", workspace_id="invalid.example.com")

    def test_normalizes_authentication_error(self) -> None:
        def transport(_request, _timeout):  # type: ignore[no-untyped-def]
            raise urllib.error.HTTPError(
                "https://example.test",
                403,
                "Forbidden",
                {},
                io.BytesIO(b'{"code":"InvalidApiKey","message":"invalid"}'),
            )

        client = AliyunAsrClient(api_key="bad", transport=transport)
        with self.assertRaises(AliyunAsrError) as context:
            client.submit(model=PARAFORMER_V2_MODEL, file_url="https://example.test/a.wav")
        self.assertEqual(context.exception.code, "authentication_failed")
        self.assertNotIn("bad", str(context.exception))

    def test_secret_file_permissions_and_cost_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "key.txt"
            path.write_text("secret\n", encoding="utf-8")
            path.chmod(0o644)
            settings = Settings(aliyun_asr_api_key_file=path)
            with self.assertRaisesRegex(ValueError, "600"):
                load_aliyun_asr_api_key(settings, environ={})
            path.chmod(0o600)
            self.assertEqual(load_aliyun_asr_api_key(settings, environ={}), "secret")

        costs = estimate_comparison_cost_cny(600)
        self.assertEqual(costs[QWEN_FILETRANS_MODEL], 0.132)
        self.assertEqual(costs[PARAFORMER_V2_MODEL], 0.048)
        self.assertEqual(costs["total"], 0.18)

    def test_upload_policy_preflight_returns_only_safe_metadata(self) -> None:
        def transport(request, _timeout):  # type: ignore[no-untyped-def]
            self.assertEqual(request.method, "GET")
            self.assertIn("model=qwen3-asr-flash-filetrans", request.full_url)
            return json.dumps(
                {
                    "request_id": "request-1",
                    "data": {
                        "policy": "private-policy",
                        "signature": "private-signature",
                        "upload_dir": "dashscope-instant/test",
                        "upload_host": "https://example.oss-cn-beijing.aliyuncs.com",
                        "expire_in_seconds": "300",
                        "max_file_size_mb": "100",
                        "oss_access_key_id": "temporary-access-key",
                    },
                }
            ).encode()

        client = AliyunTemporaryUploadClient(api_key="secret", transport=transport)
        result = client.check_model_upload_access(QWEN_FILETRANS_MODEL)
        self.assertEqual(result["max_file_size_mb"], 100)
        self.assertEqual(result["expire_in_seconds"], 300)
        serialized = json.dumps(result)
        self.assertNotIn("private", serialized)
        self.assertNotIn("temporary-access-key", serialized)

    def test_temporary_upload_keeps_file_private_and_model_scoped(self) -> None:
        requests = []

        def transport(request, _timeout):  # type: ignore[no-untyped-def]
            requests.append(request)
            if request.method == "GET":
                return json.dumps(
                    {
                        "data": {
                            "policy": "policy",
                            "signature": "signature",
                            "upload_dir": "dashscope-instant/test",
                            "upload_host": "https://upload.oss-cn-beijing.aliyuncs.com",
                            "expire_in_seconds": 300,
                            "max_file_size_mb": 100,
                            "oss_access_key_id": "temporary-key",
                            "x_oss_object_acl": "private",
                            "x_oss_forbid_overwrite": "true",
                        }
                    }
                ).encode()
            return b""

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.wav"
            source.write_bytes(b"RIFF-audio")
            result = AliyunTemporaryUploadClient(
                api_key="secret",
                transport=transport,
            ).upload(QWEN_FILETRANS_MODEL, source)
        self.assertEqual(
            result["file_url"],
            "oss://dashscope-instant/test/sample.wav",
        )
        upload_request = requests[1]
        self.assertEqual(upload_request.method, "POST")
        self.assertIn(b'name="x-oss-object-acl"\r\n\r\nprivate', upload_request.data)
        self.assertTrue(upload_request.data.endswith(b"--\r\n"))

    def test_waits_for_completion_and_downloads_both_result_shapes(self) -> None:
        responses = iter(
            (
                b'{"output":{"task_status":"RUNNING","task_id":"task-1"}}',
                b'{"output":{"task_status":"SUCCEEDED","task_id":"task-1",'
                b'"result":{"transcription_url":"https://result.aliyuncs.com/qwen.json"}}}',
                b'{"transcripts":[]}',
            )
        )

        def transport(_request, _timeout):  # type: ignore[no-untyped-def]
            return next(responses)

        client = AliyunAsrClient(api_key="secret", transport=transport)
        task = client.wait_for_completion("task-1", poll_seconds=0, sleep=lambda _value: None)
        self.assertEqual(client.download_transcription(task), {"transcripts": []})

        paraformer_task = {
            "output": {
                "task_status": "SUCCEEDED",
                "results": [
                    {
                        "subtask_status": "SUCCEEDED",
                        "transcription_url": "https://result.aliyuncs.com/paraformer.json",
                    }
                ],
            }
        }
        paraformer_client = AliyunAsrClient(
            api_key="secret",
            transport=lambda _request, _timeout: b'{"transcripts":[]}',
        )
        self.assertEqual(
            paraformer_client.download_transcription(paraformer_task),
            {"transcripts": []},
        )

    def test_upgrades_aliyun_http_result_url_to_https(self) -> None:
        requests = []

        def transport(request, _timeout):  # type: ignore[no-untyped-def]
            requests.append(request)
            return b'{"transcripts":[]}'

        client = AliyunAsrClient(api_key="secret", transport=transport)
        payload = {
            "output": {
                "result": {
                    "transcription_url": (
                        "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/"
                        "result.json?token=x"
                    )
                }
            }
        }
        self.assertEqual(client.download_transcription(payload), {"transcripts": []})
        self.assertEqual(
            requests[0].full_url,
            "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/"
            "result.json?token=x",
        )


if __name__ == "__main__":
    unittest.main()
