"""Live HTTP/MCP checks plus a scripted-model agent wiring test.

Start `mathpaper serve` and `mathpaper mcp`, then run this script from the
project directory. The model fixture below tests control flow; it does not
measure real generative answer quality.
"""
import asyncio
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
from fastmcp import Client

from mathpaper.agent import conversation, run_agent
from mathpaper.networking import mcp_transport
from mathpaper.store import data_path

MEAN_FIELD = "391b111dc9c0fe55"
GAUSSIAN = "b0646b9f5fd8b361"


class ScriptedModel:
    """Deterministic wiring fixture, explicitly not a functioning chat model."""
    model = "test-fixture"

    def __init__(self, invent_citation=False):
        self.step = 0
        self.invent = invent_citation
        self.checker_calls = 0

    async def complete(self, messages, tools=None, json_output=False):
        if json_output:
            self.checker_calls += 1
            return {"role": "assistant", "content": '{"verdicts":[{"index":0,"status":"supported"}]}'}
        self.step += 1
        if self.step == 1:
            # Deliberately request a paper outside the selected corpus. The
            # controller must replace this filter before sending the MCP call.
            calls = [("search_passages", {"query": "What assumptions are imposed in Theorem 10 for pairwise interaction energy?", "paper_ids": [GAUSSIAN], "candidates": 20, "top_k": 3})]
        elif self.step == 2 and not self.invent:
            calls = [("get_statement", {"paper_id": MEAN_FIELD, "label": f"Assumption {n}"}) for n in (4, 5, 6)]
        else:
            responses = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            source = next(s for result in responses for s in result.get("sources", []) if s.get("label") == "Theorem 10")
            claim = {"text": "Theorem 10 states its pairwise interaction result under Assumptions 4, 5 and 6.", "source_ids": ["invented-id" if self.invent else source["source_id"]]}
            return {"role": "assistant", "content": json.dumps({"claims": [claim], "abstain": False})}
        return {"role": "assistant", "content": "", "tool_calls": [{"id": f"fixture-{self.step}-{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}} for i, (name, arguments) in enumerate(calls)]}


async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=90, trust_env=False) as api:
        status = (await api.get("/api/status")).json()
        assert len(status["papers"]) >= 3 and status["pages"] >= 78 and status["passages"] >= 411
        for paper in status["papers"]:
            original = await api.get("/api/papers/"+paper["id"]+"/pdf")
            assert original.status_code == 200 and hashlib.sha256(original.content).hexdigest() == paper["sha256"]
        for route in ["/", "/assets/app.js", "/assets/style.css"]:
            assert (await api.get(route)).status_code == 200
        assert (await api.get("/api/papers/missing/pdf")).status_code == 404
        assert (await api.post("/api/search", json={"question": "test", "top_k": 10, "candidates": 5})).status_code == 409
        if not status["chat_configured"]:
            chat = await api.post("/api/chat", json={"question": "What is Theorem 1?", "thread_id": "no-model-smoke"})
            assert chat.status_code == 409 and "Choose a chat model" in chat.json()["detail"]
        print("PASS: HTTP assets, source PDF byte hashes, corpus metadata, invalid settings and chat readiness.", flush=True)

    async with Client(mcp_transport("http://127.0.0.1:8001/mcp")) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools} == {"search_passages", "get_statement", "get_context"}
        assert all(getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) for t in tools)
        result = await client.call_tool("search_passages", {"query": "What assumptions are imposed in Theorem 10 for pairwise interaction energy?", "paper_ids": [MEAN_FIELD], "candidates": 20, "top_k": 3})
        payload = result.data if isinstance(result.data, dict) else result.structured_content
        assert payload["sources"] and all(s["paper_id"] == MEAN_FIELD for s in payload["sources"])
        result = await client.call_tool("get_statement", {"paper_id": GAUSSIAN, "label": "Assumption 1"})
        body = result.data if isinstance(result.data, dict) else result.structured_content
        assert {s["page"] for s in body["sources"]} == {4, 5} and not body["truncated"]
        result = await client.call_tool("get_context", {"source_id": payload["sources"][0]["source_id"]})
        body = result.data if isinstance(result.data, dict) else result.structured_content
        assert {"Assumption 4", "Assumption 5", "Assumption 6"} <= set(body["referenced_labels"])
        print("PASS: Live MCP discovery, hybrid + reranked search, cross-page assumption lookup, context references.", flush=True)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shutil.copyfile(data_path()/"corpus.sqlite", root/"corpus.sqlite")
        good = ScriptedModel()
        with patch("mathpaper.agent.ChatModel", return_value=good):
            result = await run_agent("What assumptions does Theorem 10 use?", "good-test", [MEAN_FIELD], root)
        assert not result["abstained"] and result["tool_calls"] == 4 and good.checker_calls == 1
        assert {s["paper_id"] for s in result["sources"]} == {MEAN_FIELD}
        assert result["trace"][0]["arguments"]["paper_ids"] == [MEAN_FIELD]
        bad = ScriptedModel(invent_citation=True)
        with patch("mathpaper.agent.ChatModel", return_value=bad):
            result = await run_agent("What assumptions does Theorem 10 use?", "bad-test", [MEAN_FIELD], root)
        assert result["abstained"] and result["claims"] == [] and result["sources"] == []
        assert bad.checker_calls == 0
        assert "invented-id" not in json.dumps(conversation(root, "bad-test"))
        print("PASS: Scripted-model agent wiring over live MCP, selected-paper enforcement, citation rejection, published-only memory.", flush=True)
    print("No live generative model was called; answer quality remains unmeasured.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
