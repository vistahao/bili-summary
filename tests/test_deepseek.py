import io
import json
import unittest
import urllib.error
from pathlib import Path

from bili_summary.adapters import DeepSeekError, DeepSeekHttpBackend


class DeepSeekTests(unittest.TestCase):
    def test_sends_json_mode_and_does_not_persist_reasoning_content(self) -> None:
        captured = {}

        def transport(request, timeout):  # type: ignore[no-untyped-def]
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return json.dumps(
                {
                    "model": "deepseek-test",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps({"summary_markdown": "# 总结", "warnings": []}),
                                "reasoning_content": "不应保存",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                }
            ).encode()

        backend = DeepSeekHttpBackend(
            model="deepseek-test",
            reasoning="high",
            api_key="secret-test-key",
            transport=transport,
        )
        result = backend.process("测试", Path("schemas/summary_outputs.schema.json"))
        self.assertEqual(result.payload["summary_markdown"], "# 总结")
        self.assertEqual(result.usage["reasoning_tokens"], 3)
        self.assertEqual(captured["body"]["thinking"], {"type": "enabled"})
        self.assertEqual(captured["body"]["reasoning_effort"], "high")
        self.assertEqual(captured["body"]["max_tokens"], 32768)
        self.assertNotIn("reasoning_content", json.dumps(result.backend_metadata))
        self.assertIn("estimated_cost_usd", result.backend_metadata)

    def test_rejects_empty_and_truncated_content(self) -> None:
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
            {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        ]
        expected = ["empty_content", "truncated"]
        for response, code in zip(responses, expected):
            with self.subTest(code=code):
                backend = DeepSeekHttpBackend(
                    model="test",
                    reasoning="off",
                    api_key="key",
                    transport=lambda _request, _timeout, value=response: json.dumps(value).encode(),
                )
                with self.assertRaises(DeepSeekError) as raised:
                    backend.process("测试", Path("schemas/summary_outputs.schema.json"))
                self.assertEqual(raised.exception.code, code)

    def test_rejects_empty_response_and_invalid_json(self) -> None:
        cases = [(b"", "empty_response"), (b"not-json", "invalid_response")]
        for raw, code in cases:
            with self.subTest(code=code):
                backend = DeepSeekHttpBackend(
                    model="test",
                    reasoning="off",
                    api_key="key",
                    transport=lambda _request, _timeout, value=raw: value,
                )
                with self.assertRaises(DeepSeekError) as raised:
                    backend.process("测试", Path("schemas/summary_outputs.schema.json"))
                self.assertEqual(raised.exception.code, code)

    def test_normalizes_http_errors(self) -> None:
        cases = {
            402: ("insufficient_balance", False),
            429: ("rate_limited", True),
            500: ("server_error", True),
            503: ("overloaded", True),
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                def fail(_request, _timeout, code=status):  # type: ignore[no-untyped-def]
                    raise urllib.error.HTTPError(
                        "https://api.deepseek.com/chat/completions",
                        code,
                        "test",
                        {},
                        io.BytesIO(b"{}"),
                    )

                backend = DeepSeekHttpBackend(
                    model="test",
                    reasoning="off",
                    api_key="key",
                    transport=fail,
                )
                with self.assertRaises(DeepSeekError) as raised:
                    backend.process("测试", Path("schemas/summary_outputs.schema.json"))
                self.assertEqual((raised.exception.code, raised.exception.retryable), expected)


if __name__ == "__main__":
    unittest.main()
