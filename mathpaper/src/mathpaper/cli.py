from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="mathpaper")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Preserve and extract an uploaded PDF")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--title")
    ingest.add_argument("--notes", default="")
    sub.add_parser("models", help="Download official ONNX models and record provenance")
    sub.add_parser("index", help="Build token-bounded passages and normalized dense vectors")
    search = sub.add_parser("search")
    search.add_argument("question")
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid", "hybrid-rerank"], default="hybrid-rerank")
    search.add_argument("--candidates", type=int, default=50)
    search.add_argument("--top-k", type=int, default=5)
    sub.add_parser("status")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    sub.add_parser("mcp")
    args = parser.parse_args()
    if os.getenv("MATHPAPER_PROFILE_MEMORY") == "1":
        import atexit
        import resource
        import threading
        import time

        def report_memory():
            print("MATHPAPER_MEMORY "+json.dumps({"role": args.command, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)

        def monitor_memory():
            # Servers can re-raise SIGTERM after graceful shutdown, bypassing
            # atexit handlers. Record high-water marks during the test as well.
            while True:
                time.sleep(1)
                report_memory()

        atexit.register(report_memory)
        threading.Thread(target=monitor_memory, daemon=True).start()
    if args.command == "ingest":
        from .ingest import ingest_pdf
        print(json.dumps(ingest_pdf(args.pdf, args.title, args.notes), indent=2))
    elif args.command == "models":
        from .models import prepare_models
        print(json.dumps(prepare_models(), indent=2))
    elif args.command == "index":
        from .ingest import build_chunks
        from .retrieval import ResearchIndex
        index = ResearchIndex()
        print(json.dumps({"chunks": build_chunks(index.models.tokenizer()), "vectors": index.build_vectors()}))
    elif args.command == "search":
        from .retrieval import ResearchIndex
        print(json.dumps(ResearchIndex().search(args.question, candidates=args.candidates, top_k=args.top_k, mode=args.mode), indent=2, ensure_ascii=False))
    elif args.command == "status":
        from .store import chunks, papers
        print(json.dumps({"papers": papers(), "passages": len(chunks())}, indent=2))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("mathpaper.app:app", host=args.host, port=args.port)
    elif args.command == "mcp":
        from .mcp_server import main as start
        start()


if __name__ == "__main__":
    main()
