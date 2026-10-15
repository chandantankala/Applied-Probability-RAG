"""Exercise the container entry point on Linux/macOS without a Docker daemon."""
import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from fastmcp import Client

from mathpaper.networking import mcp_transport


async def check(process):
    deadline = time.monotonic()+25
    async with httpx.AsyncClient(base_url="http://127.0.0.1:7862", trust_env=False, timeout=30) as api:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Deployment supervisor exited during startup")
            try:
                if (await api.get("/healthz")).status_code == 200:
                    with socket.socket() as sock:
                        if sock.connect_ex(("127.0.0.1", 8002)) == 0:
                            break
            except httpx.TransportError:
                pass
            await asyncio.sleep(.1)
        else:
            raise RuntimeError("Deployment startup timed out")
        status = (await api.get("/api/status")).json()
        assert status["public_demo"] and not status["can_upload"] and not status["chat_configured"]
        assert (await api.post("/api/upload", files={"file": ("sample.pdf", b"%PDF-placeholder", "application/pdf")})).status_code == 403
        assert (await api.post("/api/chat", json={"question": "test", "thread_id": "test"})).status_code == 403
        result = await api.post("/api/search", json={"question": "What assumptions are imposed in Theorem 10 for pairwise interaction energy?", "candidates": 20, "top_k": 3})
        assert result.status_code == 200 and result.json()["candidates_scored"] == 20
        assert any(s["paper_id"] == "391b111dc9c0fe55" and s["label"] == "Theorem 10" for s in result.json()["sources"])
        print("PASS: Configurable public port, readiness, public restrictions, and real MiniLM search.", flush=True)
    async with Client(mcp_transport("http://127.0.0.1:8002/mcp")) as client:
        result = await client.call_tool("get_statement", {"paper_id": "1a24af6ab83d882c", "label": "Theorem 1"})
        body = result.data if isinstance(result.data, dict) else result.structured_content
        assert body["sources"] and {s["page"] for s in body["sources"]} == {2}
        print("PASS: Internal MCP service reaches the API on the configured deployment port.", flush=True)


def main():
    project = Path(__file__).resolve().parents[1]
    for port in (7862, 8002):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"Test port {port} is already in use.")
    environment = {**os.environ, "PORT": "7862", "MATHPAPER_MCP_PORT": "8002", "MATHPAPER_PUBLIC_DEMO": "1", "MATHPAPER_PUBLIC_CHAT": "0"}
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen([sys.executable, "deployment/start.py"], cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            asyncio.run(check(process))
        except Exception:
            log.seek(0)
            print(log.read()[-5000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if os.name == "posix":
            assert process.returncode == 0, "Supervisor did not exit gracefully"
        for port in (7862, 8002):
            with socket.socket() as sock:
                assert sock.connect_ex(("127.0.0.1", port)) != 0, "A child service was left running"
        print("PASS: Supervisor stops both services when terminated.", flush=True)
    print("The Docker image and remote hosting have not been tested by this script.", flush=True)


if __name__ == "__main__":
    main()
