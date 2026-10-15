"""MCP tools delegate to the local API so neural models load in one process."""
import os

import httpx
from fastmcp import FastMCP

from .networking import is_loopback

mcp = FastMCP("MathPaper research tools")
BACKEND = os.getenv("MATHPAPER_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


@mcp.tool
async def search_passages(query: str, paper_ids: list[str] | None = None, candidates: int = 50, top_k: int = 5) -> dict:
    """Search the selected uploaded mathematics papers with BM25 + dense fusion and MiniLM reranking. Returns original passages with source IDs, PDF pages, and file fingerprints. Scores indicate relevance, not truth."""
    async with httpx.AsyncClient(timeout=90, trust_env=not is_loopback(BACKEND)) as client:
        response = await client.post(BACKEND+"/api/search", json={"question": query, "paper_ids": paper_ids or [], "candidates": candidates, "top_k": top_k, "mode": "hybrid-rerank"})
        response.raise_for_status()
        return response.json()


@mcp.tool
async def get_statement(paper_id: str, label: str) -> dict:
    """Fetch a labelled theorem, lemma, definition, or assumption from one uploaded PDF fingerprint, with neighboring blocks that may continue the statement. For example label='Assumption 1'."""
    async with httpx.AsyncClient(timeout=30, trust_env=not is_loopback(BACKEND)) as client:
        response = await client.get(BACKEND+"/api/statement", params={"paper_id": paper_id, "label": label})
        response.raise_for_status()
        return response.json()


@mcp.tool
async def get_context(source_id: str) -> dict:
    """Expand an already retrieved source into neighboring original blocks. Returns referenced labels for explicit follow-up lookup; this does not prove a mathematical assertion."""
    async with httpx.AsyncClient(timeout=30, trust_env=not is_loopback(BACKEND)) as client:
        response = await client.get(BACKEND+"/api/context", params={"source_id": source_id})
        response.raise_for_status()
        return response.json()


def main():
    mcp.run(transport="http", host="127.0.0.1", port=int(os.getenv("MATHPAPER_MCP_PORT", "8001")), path="/mcp")


if __name__ == "__main__":
    main()
