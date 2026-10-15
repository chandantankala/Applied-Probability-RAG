# Retrieval smoke check

10 authoring-time seed questions. Page-hit checks only, not answer correctness, recall of all relevant passages, or held-out validation.

The gold labels identify physical PDF pages containing useful evidence. A hit means at least one such page appears among the top five passages. It does not establish that the complete statement or all assumptions were retrieved.

All questions search all three papers. N=50 candidates; k=5 retained; reranker pair budget=512 tokens. The CPU inference path is real ONNX inference. Timings include lazy startup on the first relevant query and are illustrative measurements from this environment.

| Method | Page hits in top 5 | Page hits among 50 candidates | Median query time |
|---|---:|---:|---:|
| bm25 | 9/10 | 10/10 | 16.4 ms |
| dense | 8/10 | 10/10 | 44.8 ms |
| hybrid | 8/10 | 10/10 | 42.4 ms |
| hybrid-rerank | 10/10 | 10/10 | 2011.5 ms |

## Queries missed by hybrid + reranking

None in this small seed set.

These questions were created during development. Do not report the results as a general RAG benchmark. Next, label additional unseen questions at passage level, include unanswerable and multi-paper questions, and separately evaluate support, citation correctness, assumptions, and latency percentiles. No live generative model was used in this retrieval check.
