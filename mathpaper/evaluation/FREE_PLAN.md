# Free-plan preparation check

The INT8 profile uses the same two selected MiniLM models, with official pinned quantized ONNX exports, batch size 1, and one ONNX Runtime thread. These measurements were collected locally, not on Render.

Sum of individual process peak RSS: **393.4 MiB**. This sum includes the API, MCP service, and supervisor. Peaks need not coincide, and shared memory may be counted more than once. Container/page-cache overhead is not fully represented; this is not a 512 MB cgroup-limit test.

| Process | Peak RSS |
|---|---:|
| serve | 230.1 MiB |
| mcp | 88.2 MiB |
| supervisor | 75.2 MiB |

| INT8 rerank candidates | Page hits in top 5 | Median local query time |
|---:|---:|---:|
| 20 | 10/10 | 1209.0 ms |
| 50 | 10/10 | 2940.9 ms |

The ten questions are authoring-time development seeds. A hit means at least one relevant physical PDF page appeared in the top five, not that an answer is correct or complete. These results do not establish held-out retrieval quality, complete assumption coverage, or hallucination rate.

The entry-point check also passed public-port readiness, selected-paper retrieval, internal MCP routing, public upload/chat restrictions, candidate limits, and a long query approaching the pair input budget. The Docker build, actual cloud memory behavior, and Render's 0.1 CPU latency remain to be checked. No live generative model was used and no paid resources were created.
