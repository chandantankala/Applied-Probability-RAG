"""Measure an INT8 deployment process tree; does not emulate Render's CPU quota."""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

from deployment_smoke import check


async def exercise(process, project):
    await check(process)
    questions = json.loads((project/"evaluation/questions.json").read_text())
    runs = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:7862", timeout=90, trust_env=False) as api:
        for candidates in (20, 50):
            for question in questions:
                response = await api.post("/api/search", json={"question": question["question"], "candidates": candidates, "top_k": 5})
                response.raise_for_status()
                result = response.json()
                hit = any(s["paper_id"] == question["gold_paper"] and s["page"] in question["gold_pages"] for s in result["sources"])
                runs.append({"id": question["id"], "candidates": candidates, "page_hit_at_5": hit, "timing_ms": result["timing_ms"]})
            print(f"INT8 N={candidates}: {sum(r['page_hit_at_5'] for r in runs if r['candidates']==candidates)}/10 page hits", flush=True)
        assert (await api.post("/api/search", json={"question": "test", "candidates": 51})).status_code == 409
        # Exercise the upper pair length with batch size 1 as well as short queries.
        response = await api.post("/api/search", json={"question": "What assumptions " + "are required "*110 + "for Theorem 10?", "candidates": 5, "top_k": 1})
        response.raise_for_status()
    await asyncio.sleep(1.1)
    return runs


def main():
    if sys.platform != "linux":
        raise SystemExit("This memory measurement expects Linux resource accounting.")
    project = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PORT": "7862", "MATHPAPER_MCP_PORT": "8002", "MATHPAPER_PUBLIC_DEMO": "1", "MATHPAPER_PUBLIC_CHAT": "0", "MATHPAPER_MODEL_PRECISION": "int8", "MATHPAPER_ORT_THREADS": "1", "MATHPAPER_INFERENCE_BATCH_SIZE": "1", "MATHPAPER_MAX_CANDIDATES": "50", "MATHPAPER_PROFILE_MEMORY": "1"}
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen([sys.executable, "deployment/start.py"], cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT)

        try:
            runs = asyncio.run(exercise(process, project))
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
        assert process.returncode == 0
        log.seek(0)
        events = [json.loads(line.split("MATHPAPER_MEMORY ", 1)[1]) for line in log.read().splitlines() if line.startswith("MATHPAPER_MEMORY ")]
        peaks = {}
        for event in events:
            peaks[event["role"]] = max(peaks.get(event["role"], 0), event["peak_rss_kib"])
        memory = [{"role": role, "peak_rss_kib": value} for role, value in peaks.items()]
        assert {m["role"] for m in memory} == {"serve", "mcp", "supervisor"}
    report = {"precision": "int8", "batch_size": 1, "ort_threads": 1, "top_k": 5,
              "sum_of_process_peak_rss_mib": round(sum(m["peak_rss_kib"] for m in memory)/1024, 1), "processes": memory, "runs": runs,
              "limits": "Local sum of each process's peak RSS, including API, MCP and supervisor. Peaks need not coincide; shared pages can be double-counted. Some container overhead/page cache is excluded. This is not a cgroup memory-cap test or a measurement on Render's 0.1 CPU. These 10 questions are development seeds, not held-out evaluation."}
    (project/"evaluation/free_tier_results.json").write_text(json.dumps(report, indent=2)+'\n')
    print("Sum of process peak RSS:", report["sum_of_process_peak_rss_mib"], "MiB", flush=True)
    print("No remote resources or paid services were created.", flush=True)


if __name__ == "__main__":
    main()
