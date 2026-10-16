---
title: MathPaper
emoji: 📚
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
models:
  - sentence-transformers/all-MiniLM-L6-v2
  - cross-encoder/ms-marco-MiniLM-L6-v2
---

# MathPaper

Explore three mathematics papers using BM25 + dense retrieval, reciprocal rank fusion, and MiniLM cross-encoder reranking. Each retrieved passage links to its original PDF page.

The initial public demo provides evidence search. A bounded MCP research agent is included in the source; answer generation requires a separately configured tool-calling LLM. No generative model has been evaluated yet.

See `DEPLOYMENT.md` for hosting configuration and `PROJECT_README.md` for the full setup, architecture, evaluation limits, and local development guide.
