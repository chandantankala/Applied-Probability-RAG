# Free-plan preparation check

The INT8 profile uses the two selected MiniLM models, with pinned official quantized ONNX exports, batch size 1, and one ONNX Runtime thread. These measurements use the currently included three PDFs and pinned Python dependencies. They were collected locally, not on Render.

Sum of individual process peak RSS: **399.2 MiB**. This sum includes the API, MCP service, and supervisor. Peaks need not coincide, and shared memory may be counted more than once. Container/page-cache overhead is not fully represented; this is not a 512 MB cgroup-limit test.

| Process | Peak RSS |
|---|---:|
| serve | 236.6 MiB |
| mcp | 87.7 MiB |
| supervisor | 74.8 MiB |

| INT8 rerank candidates | Page hits in top 5 | Median local query time |
|---:|---:|---:|
| 20 | 9/10 | 1164.5 ms |
| 50 | 9/10 | 2903.2 ms |

The ten questions are development seeds. A hit means at least one labelled physical PDF page appeared in the top five; it does not establish that an answer is correct or complete. The separation-condition question misses its definition in the top five at both candidate budgets and needs the agent's statement-lookup tool. These results do not establish held-out retrieval quality, complete assumption coverage, or hallucination rate.

The entry-point check passed public-port readiness, real retrieval, internal MCP routing, public upload/chat restrictions, candidate limits, and a long query near the pair input budget. The Docker build, actual cloud memory behavior, and Render's 0.1 CPU latency remain unverified. No live generative model was used and no paid resources were created.
