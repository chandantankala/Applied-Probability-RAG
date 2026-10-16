"""Server-side answering models. Provider secrets and reasoning stay off the wire to visitors."""
from __future__ import annotations

import json
import os
import re
import uuid

import httpx

from .networking import is_loopback


class ChatServiceError(ValueError):
    """A message safe to display without reflecting provider bodies or request headers."""


def access_error_code(response: httpx.Response) -> str:
    """Extract only known machine codes; provider messages/metadata may contain secrets."""
    allowed = {"API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_NOT_FOUND", "API_KEY_SERVICE_BLOCKED", "API_KEY_HTTP_REFERRER_BLOCKED", "API_KEY_IP_ADDRESS_BLOCKED", "API_KEY_ANDROID_APP_BLOCKED", "API_KEY_IOS_APP_BLOCKED", "SERVICE_DISABLED", "BILLING_DISABLED", "CONSUMER_INVALID", "IAM_PERMISSION_DENIED", "ACCESS_TOKEN_TYPE_UNSUPPORTED", "AUTH_API_KEY_NOT_ALLOWED"}
    try:
        error = response.json().get("error", {})
        if not isinstance(error, dict):
            return ""
        for detail in error.get("details", []):
            if isinstance(detail, dict) and detail.get("reason") in allowed:
                return detail["reason"]
        # Some key errors are delivered only as text. Recognize fixed phrases,
        # while returning a fixed code rather than any part of that text.
        message = str(error.get("message", "")).casefold()
        for marker, code in [("reported as leaked", "API_KEY_REPORTED_LEAKED"), ("api key not valid", "API_KEY_INVALID"), ("unregistered callers", "API_KEY_MISSING"), ("api keys are not supported", "API_KEY_NOT_SUPPORTED"), ("user location is not supported", "LOCATION_UNSUPPORTED"), ("unrestricted api keys", "API_KEY_UNRESTRICTED")]:
            if marker in message:
                return code
        status = error.get("status")
        return status if status in {"PERMISSION_DENIED", "UNAUTHENTICATED", "INVALID_ARGUMENT", "FAILED_PRECONDITION"} else ""
    except (ValueError, TypeError, AttributeError):
        return ""


class ChatModel:
    def __init__(self):
        self.provider = os.getenv("MATHPAPER_LLM_PROVIDER", "ollama")
        self.model = os.getenv("MATHPAPER_CHAT_MODEL", "").strip()
        default_url = "https://generativelanguage.googleapis.com/v1beta" if self.provider == "gemini" else "http://127.0.0.1:11434"
        self.base = os.getenv("MATHPAPER_LLM_URL", default_url).rstrip("/")
        self.key = os.getenv("MATHPAPER_API_KEY", "").strip()
        self.calls = 0
        self.usage = []

    def configured(self) -> bool:
        return bool(self.model) and self.provider in {"ollama", "openai-compatible", "gemini"} and (self.provider != "gemini" or bool(self.key))

    async def _post(self, endpoint: str, payload: dict, headers: dict) -> dict:
        if len(json.dumps(payload, ensure_ascii=False)) > 240_000:
            raise ChatServiceError("The evidence exceeds this demo's context budget. Start a new conversation or ask a narrower question.")
        try:
            async with httpx.AsyncClient(timeout=70, trust_env=not is_loopback(self.base)) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
        except (httpx.HTTPError, ValueError) as error:
            raise ChatServiceError("The answering service could not be reached. Please try again shortly; Find evidence remains available.") from error
        if response.status_code == 429:
            raise ChatServiceError("The answering model's free quota is temporarily exhausted. Try again later or use Find evidence.")
        if response.status_code in {401, 403}:
            code = access_error_code(response)
            diagnostic = f"HTTP {response.status_code}" + (f"; {code}" if code else "")
            raise ChatServiceError(f"The answering service did not accept its configuration ({diagnostic}). The site owner needs to check the API key and project access.")
        if not response.is_success:
            raise ChatServiceError(f"The answering service is unavailable (HTTP {response.status_code}). Please try again later or use Find evidence.")
        try:
            return response.json()
        except ValueError as error:
            raise ChatServiceError("The answering service returned an unreadable response. Please try again.") from error

    async def complete(self, messages: list[dict], tools=None, json_output=False) -> dict:
        if not self.model:
            raise ChatServiceError("Choose a chat model and set MATHPAPER_CHAT_MODEL. Evidence search works without a chat model.")
        limit = min(12, max(2, int(os.getenv("MATHPAPER_MAX_MODEL_CALLS", "10"))))
        if self.calls >= limit:
            raise ChatServiceError("This question reached the demo's model-call budget. Try a more specific question or use Find evidence.")
        self.calls += 1
        if self.provider == "gemini":
            return await self._gemini(messages, tools, json_output)
        payload = {"model": self.model, "messages": messages, "stream": False}
        headers = {}
        if tools:
            payload["tools"] = tools
        if self.provider == "ollama":
            converted = []
            for message in messages:
                item = dict(message)
                if item.get("tool_calls"):
                    item["tool_calls"] = [{"function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}} for c in item["tool_calls"]]
                if item["role"] == "tool":
                    item["tool_name"] = item.pop("name", "")
                    item.pop("tool_call_id", None)
                converted.append(item)
            payload["messages"] = converted
            payload["options"] = {"temperature": 0, "num_predict": 1400}
            if json_output:
                payload["format"] = "json"
            endpoint = self.base + "/api/chat"
        elif self.provider == "openai-compatible":
            payload.update({"temperature": 0, "max_tokens": 1400})
            if json_output:
                payload["response_format"] = {"type": "json_object"}
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
            endpoint = self.base + "/chat/completions"
        else:
            raise ChatServiceError("Use LLM provider gemini, ollama, or openai-compatible.")
        body = await self._post(endpoint, payload, headers)
        try:
            result = body["message"] if self.provider == "ollama" else body["choices"][0]["message"]
            answer = {"role": "assistant", "content": result.get("content") or ""}
            if result.get("tool_calls"):
                calls = []
                for call in result["tool_calls"]:
                    function = call["function"]
                    arguments = function["arguments"]
                    calls.append({"id": call.get("id") or "call_"+uuid.uuid4().hex[:12], "type": "function", "function": {"name": function["name"], "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}})
                answer["tool_calls"] = calls
            return answer
        except (KeyError, TypeError, IndexError) as error:
            raise ChatServiceError("The answering service returned an incomplete response. Please try again.") from error

    async def _gemini(self, messages: list[dict], tools, json_output: bool) -> dict:
        if not self.key:
            raise ChatServiceError("The site owner still needs to connect the answering model's API key. Find evidence is ready to use.")
        # Keep keys on the documented Google host; never put them in URLs or browser JavaScript.
        if self.base != "https://generativelanguage.googleapis.com/v1beta" or not re.fullmatch(r"[A-Za-z0-9._-]+", self.model):
            raise ChatServiceError("The Gemini API URL or model configuration is invalid.")
        contents, system, function_ids = [], [], {}
        for message in messages:
            role = message["role"]
            if role == "system":
                system.append({"text": message["content"]})
                continue
            if role == "tool":
                result = {"name": message["name"], "response": json.loads(message["content"])}
                original_id = function_ids.get(message["tool_call_id"])
                if original_id:
                    result["id"] = original_id
                parts, target_role = [{"functionResponse": result}], "user"
            elif role == "assistant":
                # Replay the entire original response, including opaque signatures, in order.
                # These parts exist only during this request, never in published history or traces.
                parts = message.get("_gemini_parts") or [{"text": message.get("content") or ""}]
                for call in message.get("tool_calls", []):
                    original = next((p["functionCall"] for p in parts if p.get("functionCall", {}).get("id") == call["id"]), None)
                    function_ids[call["id"]] = original.get("id") if original else None
                target_role = "model"
            else:
                parts, target_role = [{"text": message["content"]}], "user"
            if contents and contents[-1]["role"] == target_role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": target_role, "parts": list(parts)})
        payload = {"contents": contents, "systemInstruction": {"parts": system}, "generationConfig": {"maxOutputTokens": 4096}}
        # Gemini 3.5 removed sampling controls. Medium thinking supports multi-step tools.
        if self.model.startswith("gemini-3"):
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "minimal" if json_output else "medium"}
        if tools:
            payload["tools"] = [{"functionDeclarations": [{"name": t["function"]["name"], "description": t["function"].get("description", ""), "parametersJsonSchema": t["function"]["parameters"]} for t in tools]}]
        elif json_output:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        body = await self._post(self.base+"/models/"+self.model+":generateContent", payload, {"x-goog-api-key": self.key})
        try:
            candidate = body["candidates"][0]
            if candidate.get("finishReason") not in {None, "STOP"}:
                raise ChatServiceError("The model could not complete a checked answer. Try a narrower question or use Find evidence.")
            parts = candidate["content"]["parts"]
            answer = {"role": "assistant", "content": "".join(p.get("text", "") for p in parts if not p.get("thought")), "_gemini_parts": parts}
            calls = []
            for part in parts:
                if "functionCall" in part:
                    call = part["functionCall"]
                    calls.append({"id": call.get("id") or "call_"+uuid.uuid4().hex[:12], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call.get("args", {}))}})
            if calls:
                answer["tool_calls"] = calls
            usage = body.get("usageMetadata", {})
            self.usage.append({name: usage[name] for name in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount") if isinstance(usage.get(name), int)})
            return answer
        except (KeyError, TypeError, IndexError) as error:
            raise ChatServiceError("The answering model did not return an answer. Try a different question or use Find evidence.") from error
