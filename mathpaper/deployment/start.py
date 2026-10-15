"""Supervise one API worker and its loopback-only MCP service in a container."""
from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import time


def main() -> int:
    port = int(os.environ.get("PORT", "7860"))
    mcp_port = int(os.environ.get("MATHPAPER_MCP_PORT", "8001"))
    if not 1 <= port <= 65535 or not 1 <= mcp_port <= 65535 or port == mcp_port:
        raise ValueError("Choose distinct valid ports for the API and internal MCP service.")
    environment = os.environ.copy()
    environment["MATHPAPER_BACKEND_URL"] = f"http://127.0.0.1:{port}"
    environment["MATHPAPER_MCP_URL"] = f"http://127.0.0.1:{mcp_port}/mcp"
    processes = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        processes.append(subprocess.Popen([sys.executable, "-m", "mathpaper", "serve", "--host", os.getenv("MATHPAPER_HOST", "0.0.0.0"), "--port", str(port)], env=environment))
        processes.append(subprocess.Popen([sys.executable, "-m", "mathpaper", "mcp"], env=environment))
        while not stopping:
            if any(process.poll() is not None for process in processes):
                return 1
            time.sleep(.2)
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if os.getenv("MATHPAPER_PROFILE_MEMORY") == "1":
            import resource
            print("MATHPAPER_MEMORY "+json.dumps({"role": "supervisor", "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
