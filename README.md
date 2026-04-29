# Applied Probability RAG

A retrieval-augmented question-answering system over a corpus of applied-probability research papers. Ask a question in natural language; get an answer grounded in the papers, with citations to specific files and pages.

Built with [LangChain](https://www.langchain.com/), [Voyage AI](https://www.voyageai.com/) embeddings, [Chroma](https://www.trychroma.com/) for vector storage, [Anthropic Claude](https://www.anthropic.com/) for generation, and [Streamlit](https://streamlit.io/) for the frontend.

## What it does

Drop a folder of PDFs in `papers/`, run `ingest.py` to build a vector index, and launch `app.py` to query the corpus through a browser. Answers are constrained to the retrieved context and include `\[paper.pdf, p. X]`-style citations so claims can be verified against the source.

## Architecture

The system is split into two pipelines that run at different times:

**Ingestion** (offline, `ingest.py`):
PDFs → page-level loading (PyPDFLoader) → math-aware chunking (RecursiveCharacterTextSplitter with theorem/lemma/proof/definition separators) → embedding (Voyage `voyage-4-large`) → persistent Chroma index on disk.

**Query** (online, `app.py` + `rag\_chain.py`):
User question → embedding (same Voyage model) → top-k similarity search in Chroma → prompt assembly (system instructions + retrieved excerpts + question) → Claude generation → answer + source list shown in Streamlit.

The two pipelines are intentionally decoupled. Ingestion is slow and runs only when the corpus changes; querying is fast and runs on every user request.

## Setup

### 1\. Clone and install

```bash
git clone https://github.com/<your-username>/applied-probability-rag.git
cd applied-probability-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2\. Get API keys

* **Anthropic** (for Claude): create a key at [console.anthropic.com](https://console.anthropic.com). Requires a small credit balance; queries cost cents.
* **Voyage AI** (for embeddings): sign up via [MongoDB Atlas](https://cloud.mongodb.com) and generate a Voyage API key from the AI Models dashboard. The free tier (200M tokens for `voyage-4-\*` models) is more than enough for a personal corpus.

### 3\. Configure the environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# then edit .env with your actual keys
```

`.env` is gitignored. Never commit it.

### 4\. Add papers and build the index

```bash
mkdir -p papers
# drop your PDFs into ./papers/
python ingest.py
```

This loads each PDF, chunks the text, embeds the chunks via Voyage, and writes a persistent Chroma database to `./chroma\_db/`. Expect a few seconds per paper for a small corpus.

### 5\. Launch the app

```bash
streamlit run app.py
```

A browser tab will open at `http://localhost:8501`. Type a question, hit **Ask**, and read the answer along with the retrieved excerpts that grounded it.

## Project structure

```
applied-probability-rag/
├── ingest.py            # build the Chroma index from papers/
├── rag\_chain.py         # retriever + Claude chain (pure logic, no UI)
├── app.py               # Streamlit frontend
├── requirements.txt
├── .env.example         # template for API keys
├── .gitignore
└── README.md
```

`papers/` and `chroma\_db/` are created locally and gitignored — the corpus and index are not part of the repository.

## Configuration

The main knobs live as constants near the top of each file:

* `ingest.py`: `EMBEDDING\_MODEL`, `chunk\_size`, `chunk\_overlap`, `MATH\_SEPARATORS`
* `rag\_chain.py`: `LLM\_MODEL`, `TOP\_K`, `SYSTEM\_PROMPT`

The embedding model in `rag\_chain.py` must match the one used in `ingest.py`. If you change either, delete `chroma\_db/` and re-run ingestion.

## Design notes

A few choices worth flagging:

**Math-aware chunk separators.** Naive character-based chunking will cut a theorem from its proof, or a definition from the symbol it introduces. The custom separator list (`Theorem`, `Lemma`, `Proof`, etc.) tells `RecursiveCharacterTextSplitter` to prefer breaking at those boundaries before falling back to paragraphs and sentences.

**Strict grounding via system prompt.** The system prompt tells Claude to answer only from the provided excerpts and to cite paper and page for every substantive claim. This makes citations meaningful — without that constraint, Claude could draw on training-data knowledge while still appearing to cite the paper.

**Temperature 0.** Generation is deterministic so identical questions produce stable answers, which matters for evaluation and debugging.

**Decoupled ingestion and query.** Putting retrieval logic in `rag\_chain.py` (no Streamlit imports) makes it possible to test the pipeline without spinning up the UI — e.g., from a notebook, a script, or a future evaluation harness.

## Limitations

* **PDF math fidelity.** PyPDFLoader extracts text but mangles a fraction of LaTeX equations and special symbols. For better fidelity on equation-heavy papers, swap in `UnstructuredPDFLoader` with `strategy="hi\_res"` or a LaTeX-aware extractor like Nougat.
* **No re-ranking.** Retrieved chunks are taken directly from vector similarity, top-k. Adding a cross-encoder re-ranker (e.g., Voyage Rerank, Cohere Rerank) typically improves answer quality at the cost of one extra API call per query.
* **No conversation memory.** Each question is answered independently. Multi-turn follow-ups ("what about the second case you mentioned?") are not supported.
* **General-purpose embeddings.** `voyage-4-large` is trained on broad web text and may not finely distinguish closely related probabilistic concepts (e.g., almost-sure vs. in-probability convergence). Domain-specialized embeddings could improve retrieval for subtle technical queries.

## Possible extensions

* Hybrid retrieval (BM25 + dense vectors with Reciprocal Rank Fusion).
* Cross-encoder re-ranking on top-20 retrieval.
* Query rewriting / HyDE before retrieval.
* A formal evaluation harness — labeled question/answer set, recall@k, MRR, faithfulness, citation accuracy.
* Containerized deployment via Docker for reproducible setup.

