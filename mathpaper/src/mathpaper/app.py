from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import run_agent
from .limits import ChatLimitError, PublicChatGuard, daily_limit
from .llm import ChatModel, ChatServiceError
from .ingest import build_chunks, ingest_pdf
from .retrieval import ResearchIndex
from .store import chunks, data_path, papers, records

app = FastAPI(title="MathPaper", description="Research over exact uploaded PDF versions")
index = ResearchIndex()
web = Path(__file__).parent / "web"


def public_demo() -> bool:
    return os.getenv("MATHPAPER_PUBLIC_DEMO", "0") == "1"


def chat_allowed() -> bool:
    return not public_demo() or os.getenv("MATHPAPER_PUBLIC_CHAT", "0") == "1"


def chat_configured() -> bool:
    return chat_allowed() and ChatModel().configured() and bool(importlib.util.find_spec("fastmcp"))


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    paper_ids: list[str] = Field(default_factory=list)
    candidates: int = Field(default=50, ge=1, le=150)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = "hybrid-rerank"
    max_length: int = Field(default=512, ge=64, le=512)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(min_length=1, max_length=100)
    paper_ids: list[str] = Field(default_factory=list)


@app.get("/")
def home():
    return FileResponse(web / "index.html")


@app.get("/evaluation")
def evaluation():
    return FileResponse(web / "evaluation.html")


@app.get("/api/status")
def status():
    corpus = papers()
    return {"papers": corpus, "pages": sum(p["pages"] for p in corpus), "passages": len(chunks()), "models_ready": index.models.ready(), "index_ready": index.vector_path.exists(), "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2", "model_precision": index.models.precision, "max_candidates": min(150, max(1, int(os.getenv("MATHPAPER_MAX_CANDIDATES", "150")))), "default_candidates": int(os.getenv("MATHPAPER_DEFAULT_CANDIDATES", "50")), "chat_model": os.getenv("MATHPAPER_CHAT_MODEL", ""), "agent_installed": bool(importlib.util.find_spec("fastmcp")), "chat_configured": chat_configured(), "answer_provider": os.getenv("MATHPAPER_LLM_PROVIDER", "ollama"), "chat_daily_limit": daily_limit() if public_demo() else None, "public_demo": public_demo(), "can_upload": not public_demo()}


@app.get("/healthz")
def health():
    if not index.models.ready() or not index.vector_path.exists():
        raise HTTPException(503, "The retrieval models or vector index are not ready.")
    return {"status": "ready"}


@app.get("/api/papers/{paper_id}/pdf")
def original(paper_id: str):
    matches = records("SELECT * FROM papers WHERE id=?", (paper_id,))
    if not matches:
        raise HTTPException(404, "Unknown paper.")
    return FileResponse(data_path()/"papers"/(paper_id+".pdf"), media_type="application/pdf", filename=matches[0]["filename"], content_disposition_type="inline")


@app.post("/api/search")
def search(request: SearchRequest):
    try:
        return index.search(request.question, request.paper_ids, request.candidates, request.top_k, request.mode, request.max_length)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(409, str(error)) from error


@app.get("/api/statement")
def statement(paper_id: str, label: str):
    return index.get_statement(paper_id, label)


@app.get("/api/context")
def context(source_id: str):
    return index.get_context(source_id)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not chat_allowed():
        raise HTTPException(403, "Generated answers are not enabled on this public demo. Find evidence is available.")
    if not chat_configured():
        raise HTTPException(503, "The answering model is not connected yet. Find evidence is ready to use.")
    if not request.question.strip():
        raise HTTPException(422, "Enter a research question.")
    guard, token = PublicChatGuard(), None
    try:
        if public_demo():
            token = guard.reserve()
        return await asyncio.wait_for(run_agent(request.question.strip(), request.thread_id, request.paper_ids), timeout=180)
    except ChatLimitError as error:
        raise HTTPException(429, str(error), headers={"Retry-After": str(error.retry_after)}) from error
    except ChatServiceError as error:
        raise HTTPException(503, str(error)) from error
    except TimeoutError as error:
        raise HTTPException(504, "The answer took too long. Try a narrower question or use Find evidence.") from error
    except Exception as error:
        # Never reflect provider exception text, tool content, headers or credentials.
        logging.getLogger(__name__).warning("Agent request failed (%s)", type(error).__name__)
        raise HTTPException(503, "The answer could not be completed. Please try again or use Find evidence.") from error
    finally:
        if token:
            guard.release(token)


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    if public_demo():
        raise HTTPException(403, "Uploads are disabled on this public demo. Explore the included papers.")
    if not file.filename or not file.filename.casefold().endswith(".pdf"):
        raise HTTPException(400, "Choose a PDF file.")
    content = file.file.read(25*1024*1024+1)
    if len(content) > 25*1024*1024:
        raise HTTPException(413, "The prototype accepts PDFs up to 25 MB.")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / Path(file.filename).name
        path.write_bytes(content)
        try:
            result = ingest_pdf(path)
            if not result["already_ingested"] and index.models.ready():
                build_chunks(index.models.tokenizer(), budget=220)
                index.build_vectors()
            return result
        except Exception as error:
            raise HTTPException(400, "Could not ingest the PDF: "+str(error)[:300]) from error


app.mount("/assets", StaticFiles(directory=web), name="assets")
