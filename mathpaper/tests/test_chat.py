"""Provider protocol and public quota boundaries, without credentials or live LLM calls."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from mathpaper.app import app
from mathpaper.limits import ChatLimitError, PublicChatGuard
from mathpaper.llm import ChatModel, ChatServiceError

GEMINI = {"MATHPAPER_LLM_PROVIDER": "gemini", "MATHPAPER_CHAT_MODEL": "gemini-3.5-flash-lite", "MATHPAPER_LLM_URL": "https://generativelanguage.googleapis.com/v1beta", "MATHPAPER_API_KEY": "fixture-not-a-real-key"}
TOOLS = [{"type": "function", "function": {"name": "search_passages", "description": "Find evidence", "parameters": {"type": "object", "properties": {"paper_ids": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}}, "required": []}}}]


class GeminiProtocolTests(unittest.TestCase):
    def client_factory(self, handler):
        client_class = httpx.AsyncClient
        return lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs)

    def test_parallel_tools_preserve_signatures_order_ids_and_nullable_schema(self):
        original_parts = [
            {"text": "private reasoning fixture", "thought": True},
            {"functionCall": {"name": "search_passages", "id": "first-call", "args": {"paper_ids": None}}, "thoughtSignature": "opaque-first-signature"},
            {"functionCall": {"name": "search_passages", "id": "second-call", "args": {"paper_ids": ["a"]}}},
        ]
        requests = []

        def handler(request):
            self.assertEqual(request.headers["x-goog-api-key"], GEMINI["MATHPAPER_API_KEY"])
            self.assertNotIn(GEMINI["MATHPAPER_API_KEY"], str(request.url))
            requests.append(json.loads(request.content))
            parts = original_parts if len(requests) == 1 else [{"text": '{"claims":[],"abstain":true}'}]
            return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 25}})

        async def run():
            model = ChatModel()
            messages = [{"role": "system", "content": "Use paper evidence."}, {"role": "user", "content": "An unrelated question"}]
            first = await model.complete(messages, tools=TOOLS)
            self.assertEqual(first["content"], "")
            self.assertEqual(len(first["tool_calls"]), 2)
            messages.append(first)
            for call in first["tool_calls"]:
                messages.append({"role": "tool", "name": call["function"]["name"], "tool_call_id": call["id"], "content": '{"sources":[]}'})
            final = await model.complete(messages, tools=TOOLS)
            self.assertTrue(json.loads(final["content"])["abstain"])
            self.assertEqual(model.calls, 2)
            self.assertEqual(model.usage, [{"totalTokenCount": 25}, {"totalTokenCount": 25}])

        with patch.dict(os.environ, GEMINI), patch("mathpaper.llm.httpx.AsyncClient", side_effect=self.client_factory(handler)):
            asyncio.run(run())
        self.assertEqual(requests[0]["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"], TOOLS[0]["function"]["parameters"])
        self.assertEqual(requests[1]["contents"][1]["parts"], original_parts)
        responses = requests[1]["contents"][2]["parts"]
        self.assertEqual([p["functionResponse"]["id"] for p in responses], ["first-call", "second-call"])
        self.assertEqual([p["functionResponse"]["name"] for p in responses], ["search_passages", "search_passages"])

    def test_checker_requests_json_without_removed_sampling_parameters(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
            self.assertNotIn("temperature", payload["generationConfig"])
            self.assertNotIn("tools", payload)
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"verdicts":[]}'}]}, "finishReason": "STOP"}]})
        with patch.dict(os.environ, GEMINI), patch("mathpaper.llm.httpx.AsyncClient", side_effect=self.client_factory(handler)):
            result = asyncio.run(ChatModel().complete([{"role": "user", "content": "Check claims"}], json_output=True))
        self.assertEqual(json.loads(result["content"]), {"verdicts": []})

    def test_key_is_required_and_sent_only_to_the_google_api(self):
        with patch.dict(os.environ, {**GEMINI, "MATHPAPER_API_KEY": ""}):
            self.assertFalse(ChatModel().configured())
            with self.assertRaisesRegex(ChatServiceError, "API key"):
                asyncio.run(ChatModel().complete([]))
        with patch.dict(os.environ, {**GEMINI, "MATHPAPER_LLM_URL": "https://example.com"}):
            with self.assertRaisesRegex(ChatServiceError, "invalid"):
                asyncio.run(ChatModel().complete([]))

    def test_provider_quota_errors_do_not_echo_secrets_and_call_budget_stops_requests(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(429, json={"error": "secret="+GEMINI["MATHPAPER_API_KEY"]})
        with patch.dict(os.environ, {**GEMINI, "MATHPAPER_MAX_MODEL_CALLS": "2"}), patch("mathpaper.llm.httpx.AsyncClient", side_effect=self.client_factory(handler)):
            model = ChatModel()
            for _ in range(2):
                with self.assertRaisesRegex(ChatServiceError, "free quota") as caught:
                    asyncio.run(model.complete([]))
                self.assertNotIn(GEMINI["MATHPAPER_API_KEY"], str(caught.exception))
            with self.assertRaisesRegex(ChatServiceError, "model-call budget"):
                asyncio.run(model.complete([]))
        self.assertEqual(len(requests), 2)


class PublicChatLimitTests(unittest.TestCase):
    def test_active_answer_cooldown_and_daily_quota_are_shared_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MATHPAPER_CHAT_DAILY_LIMIT": "2", "MATHPAPER_CHAT_COOLDOWN": "20"}):
            root = Path(directory)
            first = PublicChatGuard(root)
            token = first.reserve(now=100_000)
            other = PublicChatGuard(root)
            with self.assertRaisesRegex(ChatLimitError, "another question"):
                other.reserve(now=100_001)
            first.release(token)
            with self.assertRaisesRegex(ChatLimitError, "wait a few seconds"):
                other.reserve(now=100_005)
            token = other.reserve(now=100_020)
            other.release(token)
            with self.assertRaisesRegex(ChatLimitError, "daily answer limit"):
                first.reserve(now=100_040)
            token = first.reserve(now=186_401)
            first.release(token)

    def test_abandoned_request_lease_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = PublicChatGuard(Path(directory))
            guard.reserve(now=100_000)
            self.assertTrue(guard.reserve(now=100_211))

    def test_missing_key_disables_chat_before_reserving_any_quota(self):
        with patch.dict(os.environ, {**GEMINI, "MATHPAPER_API_KEY": "", "MATHPAPER_PUBLIC_DEMO": "1", "MATHPAPER_PUBLIC_CHAT": "1"}), patch("mathpaper.app.run_agent", new_callable=AsyncMock) as agent, patch("mathpaper.app.PublicChatGuard.reserve") as reserve, TestClient(app) as client:
            self.assertFalse(client.get("/api/status").json()["chat_configured"])
            self.assertEqual(client.post("/api/chat", json={"question": "A question", "thread_id": "test"}).status_code, 503)
            agent.assert_not_called()
            reserve.assert_not_called()

    def test_endpoint_releases_lease_and_returns_retry_header_without_second_model_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {**GEMINI, "MATHPAPER_DATA": directory, "MATHPAPER_PUBLIC_DEMO": "1", "MATHPAPER_PUBLIC_CHAT": "1", "MATHPAPER_CHAT_COOLDOWN": "20"}), patch("mathpaper.app.run_agent", new_callable=AsyncMock, return_value={"claims": [], "abstained": True}) as agent, TestClient(app) as client:
            payload = {"question": "A question", "thread_id": "test"}
            self.assertEqual(client.post("/api/chat", json=payload).status_code, 200)
            second = client.post("/api/chat", json=payload)
            self.assertEqual(second.status_code, 429)
            self.assertGreater(int(second.headers["retry-after"]), 0)
            self.assertIn("wait a few seconds", second.json()["detail"])
            agent.assert_awaited_once()

    def test_unexpected_exception_details_are_never_published(self):
        with patch.dict(os.environ, {**GEMINI, "MATHPAPER_PUBLIC_DEMO": "0"}), patch("mathpaper.app.run_agent", new_callable=AsyncMock, side_effect=RuntimeError("private credential fixture")), TestClient(app) as client:
            result = client.post("/api/chat", json={"question": "A question", "thread_id": "test"})
            self.assertEqual(result.status_code, 503)
            self.assertNotIn("private credential", result.text)


if __name__ == "__main__":
    unittest.main()
