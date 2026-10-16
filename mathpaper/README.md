# MathPaper

**Live demo:** [mathpaper-chandan-tankala.onrender.com](https://mathpaper-chandan-tankala.onrender.com/). Evidence search is public; generated answers are disabled until an answering model is connected. Free hosting can take about a minute to wake after inactivity.

This checkout includes the author's three approved source PDFs. The derived database, vectors, model weights, and secrets are excluded from Git. The Docker build verifies the PDF fingerprints and rebuilds the corpus; publishing this source alone does not deploy a website.

A Python research chatbot project built around Chandan Tankala's three uploaded mathematics papers. It includes a browser interface, real CPU hybrid retrieval and MiniLM reranking, and a bounded agent that calls research tools through MCP.

**Ready now:** searching the papers and opening the source PDF pages. **One configuration step remains for generated answers:** connect a tool-calling chat LLM. No generative model or paid API account has been selected or used in preparing this project.

For website hosting, see [DEPLOYMENT.md](DEPLOYMENT.md). The live Render service uses its **free** plan, smaller INT8 versions of both MiniLM models, and public evidence search with uploads and generated answers disabled.

## Start with evidence search

Use Python 3.12 in a virtual environment. Open a terminal in this `mathpaper/` folder after cloning the repository and checking out `mathpaper-free-web-demo`.

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install and run:

```bash
python -m pip install -c requirements.lock -e ".[agent]"
python deployment/bootstrap.py
mathpaper models
mathpaper index
mathpaper serve
```

Open **http://127.0.0.1:8000**. Choose **Find evidence**, then try one of the suggested questions. The default full-precision model download is about 180 MB and happens once. `mathpaper models` downloads pinned official files from Hugging Face and verifies SHA-256 checksums. The hosted INT8 profile downloads about 44 MiB instead.

The approved PDFs are already at the fingerprinted paths in `data/papers.json`. `python deployment/bootstrap.py` extracts them into a local database, and `mathpaper index` rebuilds page-linked passages and vectors. If a source PDF changes, update its manifest entry and development labels before rebuilding; do not reuse the previous version's identity.

## Connect the answering model

MiniLM generates embeddings and relevance scores; it does not write answers. The agent needs a separate model that supports tool calls and JSON output.

Copy `.env.example` to `.env` in this folder. The app loads it when started; existing shell variables take precedence. Edit the values locally and restart the servers after a change.

For a local Ollama service with a suitable model already installed:

```dotenv
MATHPAPER_LLM_PROVIDER=ollama
MATHPAPER_LLM_URL=http://127.0.0.1:11434
MATHPAPER_CHAT_MODEL=YOUR_INSTALLED_TOOL_CALLING_MODEL
```

For a hosted service implementing the Chat Completions tool-calling format:

```dotenv
MATHPAPER_LLM_PROVIDER=openai-compatible
MATHPAPER_LLM_URL=https://YOUR_SERVICE/v1
MATHPAPER_CHAT_MODEL=YOUR_MODEL_NAME
MATHPAPER_API_KEY=YOUR_KEY
```

The hosted adapter expects `tools`, `response_format=json_object`, `temperature`, and `max_tokens`; a service with a different API will need a small adapter change in `agent.py`. Credentials belong in your local `.env`, which is ignored by Git. A hosted model receives the question, relevant conversation history, and retrieved passages. The local option keeps these model requests on your machine.

Keep `mathpaper serve` running. Open a second terminal, activate the same environment, change to this folder, and run:

```bash
mathpaper mcp
```

This starts the MCP service at `http://127.0.0.1:8001/mcp`. Refresh the interface and choose **Ask the agent**. A configured model name enables that mode; the first request will reveal whether your model service and MCP connection are reachable.

## Included papers

| Uploaded paper | Physical PDF pages | Version detail |
|---|---:|---|
| Beyond propagation of chaos: A stochastic algorithm for mean field optimization | 31 | Uploaded proceedings copy |
| Dense associative memory for Gaussian distributions | 33 | Uploaded ICML 2026 proceedings copy, PMLR 306 |
| Fast mixing of a randomized shift-register Markov chain | 14 | Uploaded journal copy |

The 78 pages produce **404 passages** with the pinned dependencies, each linked to its original PDF page. `data/papers.json` records file fingerprints and version notes. A paper ID is derived from its PDF bytes, so a revised PDF gets a distinct identity. All statement labels and development page labels refer to the included copies. PDF page links use physical page numbers starting at 1, rather than journal page numbers.

## Retrieval and agent behavior

| Component | Implementation |
|---|---|
| PDF extraction | PyMuPDF blocks; two-column and review-margin heuristics; exact PDF bytes retained |
| Passages | Label-aware, page-contained blocks, at most 220 model tokens; 24-token overlap when splitting long blocks |
| Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2`; normalized mean-pooled embeddings; cosine similarity |
| Keyword retrieval | BM25 over extracted passage text |
| Hybrid fusion | Reciprocal rank fusion, constant 60; up to N hits from each branch, fused down to N candidates |
| Reranking | Agreed `cross-encoder/ms-marco-MiniLM-L6-v2`, question + passage text, CPU ONNX |
| Default budgets | N=50 reranked candidates, k=5 retained, at most 512 combined pair tokens including special tokens |
| Storage | SQLite source metadata/text; NumPy vector cache; exact local cosine search |
| Conversation | SQLite per-thread memory of published claims and source references; last 12 messages |

The small dense embedding model is an initial baseline. Both models are general text models; their adequacy for mathematical notation needs measurement. A local exact vector search is sufficient for this 404-passage corpus; a vector database is a later scaling option.

The agent chooses among three MCP tools:

- `search_passages`: hybrid retrieval followed by reranking, constrained to the selected papers.
- `get_statement`: look up a labelled theorem, lemma, definition, or assumption, including continuation blocks up to the next detected boundary.
- `get_context`: expand a retrieved passage and list referenced statement labels for follow-up lookup.

The agent has a default budget of six tool calls and one repair attempt. It drafts up to six claims with source IDs. The controller rejects IDs that were not retrieved in that turn. A separate LLM call checks each remaining claim against its cited text. Only supported claims are displayed and stored; the system can return a partial answer or abstain. Source text is treated as untrusted data in the prompts, and cross-paper tool requests are restricted by the controller.

This gate reduces unsupported output but cannot certify mathematical correctness. It uses the same configured model for a separate checking call, so correlated mistakes remain possible. A citation proves provenance, not entailment. Extraction can scramble symbols, inequality signs, superscripts, and equation order; the linked PDF is authoritative. Statement lookup is a bounded layout heuristic and may include nearby prose or figures. The checker is instructed to withhold precise mathematical claims when extraction is ambiguous.

## What has been checked

- Real CPU embedding and cross-encoder inference on all 404 passages and the starter questions.
- Sixteen deterministic tests for fusion, paper filtering, stale-vector rejection, precision-specific caches, content identity, statement boundaries (including a heading merged with an equation), source links, citation validation, bounded thread history, and public-demo restrictions.
- HTTP endpoints and original-PDF byte hashes.
- Live MCP discovery and all three research tools, including the Gaussian separation assumption on PDF page 5 of the included proceedings copy.
- A scripted-model agent wiring check over the real MCP service, including selected-paper enforcement and rejection of invented citations. This fixture is only used in tests.
- JavaScript syntax and Python compilation. A live browser query returned five source passages with original-page links; mobile layout and comprehensive browser/accessibility coverage remain untested.
- The container entry point, configurable public port, readiness, real retrieval, internal MCP routing, and graceful shutdown. The first Render Docker image built successfully, and the free service passed live health, corpus, original-PDF fingerprint, and published-version statement checks. See `evaluation/cloud_validation.json`.
- The INT8 profile on the same ten development seed questions at N=20 and N=50, along with local memory measurements. These checks are recorded separately in `evaluation/free_tier_results.json` and do not establish performance on the actual free cloud instance.

No live generative model has been called, so mathematical answer quality, model tool-use reliability, end-to-end chat latency, and hallucination rates remain unmeasured.

The ten development questions in `evaluation/questions.json` are a starting smoke set, with labels updated for the included proceedings PDF. Current page-hit counts and measured timings are in [evaluation/RESULTS.md](evaluation/RESULTS.md). They are **not** a held-out benchmark or answer-accuracy result. The separation-condition question still requires a follow-up statement lookup when its definition is missing from the top five passages.

Run the deterministic tests:

```bash
python -m unittest discover -s tests -v
```

With the two normal servers stopped, run the integration check; it starts and stops its own services:

```bash
python evaluation/run_smoke.py
```

Rerun the retrieval comparison after changing a retrieval parameter or model:

```bash
python evaluation/evaluate.py
```

## Work through the project together

1. Run **Find evidence** and inspect several source pages, especially formulas and cross-page statements.
2. Choose and connect the generating model, then evaluate its first cited answers and follow-up questions.
3. Create a larger, separately labelled question set: direct facts, assumptions, theorem comparisons, proof dependencies, and questions the corpus cannot answer.
4. Tune N, k, pair input length, batch size, and inference settings against retrieval quality, complete evidence coverage, grounded answer quality, and latency. Reranker scores are relevance logits, not confidence probabilities.
5. Add passage-level relevance labels, answer support labels, request/token usage logging, and an agent-versus-single-retrieval comparison. Fine-tuning or a math-specialized model should follow measured failure patterns.

## Source map

| File | Purpose |
|---|---|
| `src/mathpaper/ingest.py` | PDF identity, layout extraction, token-bounded passages |
| `src/mathpaper/models.py` | Pinned downloads, mean pooling, CPU ONNX inference |
| `src/mathpaper/model-lock.json` | Revisions and file checksums for both models |
| `src/mathpaper/retrieval.py` | BM25, cosine search, RRF, reranking, statement/context lookup |
| `src/mathpaper/mcp_server.py` | MCP tools calling the local retrieval API |
| `src/mathpaper/agent.py` | Model adapters, tool loop, claim checks, repair, conversation memory |
| `src/mathpaper/app.py` | FastAPI routes and local PDF serving |
| `src/mathpaper/web/` | Browser interface |
| `tests/` | Deterministic safeguards |
| `evaluation/` | Seed questions, measured retrieval results, integration checks |

The local CLI binds to loopback by default; container startup exposes the API on the host's port while MCP stays on loopback. ONNX Runtime telemetry is disabled before initialization; FastMCP update checks and telemetry are off by default. Once weights are downloaded, evidence search needs no external service. The public deployment defaults to a fixed, read-only paper collection. It has no user-account authentication or paid-generation usage controls yet.
