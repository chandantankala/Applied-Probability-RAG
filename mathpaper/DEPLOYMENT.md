# Hosting MathPaper

The application is under `mathpaper/`, and the root `render.yaml` points to that Docker context on branch `mathpaper-free-web-demo`. The author's three approved PDFs are included. `deployment/bootstrap.py` verifies their fingerprints and builds the source database before indexing. The associative-memory source is the uploaded ICML 2026 proceedings copy, rather than the earlier review draft.

This setup runs the existing FastAPI interface and Python retrieval pipeline together in one container. The MCP service stays internal on loopback; the website is exposed on the platform's `PORT`. Model files and the vector index are built into the image, so normal web requests do not download weights or rebuild the corpus.

**Live on Render Free:** [mathpaper-chandan-tankala.onrender.com](https://mathpaper-chandan-tankala.onrender.com/). Render built the Docker image and started the single free service successfully. Live checks passed readiness, corpus metadata, all three original-PDF fingerprints, and published-version statement lookup; a browser query returned five reranked passages. `evaluation/cloud_validation.json` records the initial deployment checks. Generated answers remain disabled until a model is connected. No paid resources were created.

## Public demo defaults

- Evidence search works over the three included papers, with BM25, dense, hybrid, and hybrid + reranking controls.
- Uploads are disabled on both the server and interface. Updating the shared corpus is an owner operation followed by a rebuild.
- Generated answers are disabled until a model is chosen and `MATHPAPER_PUBLIC_CHAT=1` is explicitly set. No LLM API secret is included in the image.
- Both MiniLM models run on CPU. The container now defaults to the pinned INT8 exports of the same models, batch size 1, and 20 candidates; visitors can experiment with up to 50 candidates. No GPU is required for retrieval.
- Use one API worker; the supervisor exits if either the API or MCP process dies, allowing the host to restart the container.

## Try the container

On a machine with Docker installed, from the project directory:

```bash
docker build -t mathpaper .
docker run --rm -p 7860:7860 mathpaper
```

Open http://localhost:7860 and try the starter questions. `GET /healthz` must return `{"status":"ready"}`. The deployment configuration uses public-demo settings even when running this container locally.

To exercise the entry point without Docker on Linux/macOS after installing the project and models:

```bash
python evaluation/deployment_smoke.py
```

## Option A: Render Free

The included `render.yaml` now selects **`plan: free`**, with no paid web service or database. Render currently provides 512 MB RAM and 0.1 CPU on this plan. It sleeps after 15 minutes without traffic, and the next visitor may wait roughly a minute for it to restart. The workspace has 750 free instance hours per month, shared across its free services. Bandwidth and build allowances also apply. Stay within the free plan rather than upgrading automatically.

The low-memory profile uses the official pinned `model_quint8_avx2.onnx` exports for both MiniLM models. The combined model files shrink from about 173 MiB to 44 MiB. It retains BM25 + dense fusion and the agreed cross-encoder. Separate `vectors-int8.npz` and `vectors.npz` files prevent mixing quantized and full-precision embedding caches.

`evaluation/free_tier_results.json` records the local INT8 checks, per-process memory peaks, and results at N=20 and N=50. These are local measurements, not proof that the image fits Render's exact memory accounting or meets a latency target on its 0.1 CPU. Confirm that during the first free deployment.

The local sum of individual peak RSS was **399.2 MiB** for the API, MCP service, and supervisor. See [evaluation/FREE_PLAN.md](evaluation/FREE_PLAN.md) for the measurement limits and seed-question results.

1. Use repository `chandantankala/Applied-Probability-RAG`, branch **`mathpaper-free-web-demo`**. All three approved PDFs are already included at the paths in `mathpaper/data/papers.json`. The Docker build regenerates the database and vector cache; neither should be committed.
2. Connect Render to this GitHub repository. Create a Blueprint using this branch and its root `render.yaml`, verify the service says **Free**, and deploy. Alternatively, use the exact manual web-service settings below.
3. Confirm the build succeeds, health checks pass, paper links open, and a real hybrid-reranked query completes. Inspect peak memory and latency before settling on an instance size.
4. Use the returned HTTPS URL for the live demo. A custom domain is optional.

The frontend and Python backend share this one free web service. A custom domain and separate database are unnecessary. Normal requests are handled live; answers or retrieval results are not precomputed. The initial Docker build and free deployment passed; peak cloud memory and sustained-load latency remain unmeasured.

The current service was created from the public repository URL. To publish later changes, open the service in Render and choose **Manual Deploy → Deploy latest commit** for `mathpaper-free-web-demo`. Source changes in GitHub alone do not prove a new version is live.

### Manual web-service settings

| Field | Value |
|---|---|
| Repository | `https://github.com/chandantankala/Applied-Probability-RAG` |
| Branch | `mathpaper-free-web-demo` |
| Name | `mathpaper-chandan-tankala` (or another available name) |
| Language/runtime | Docker |
| Root Directory | `mathpaper` |
| Dockerfile Path | `./Dockerfile` (relative to that root) |
| Docker Build Context | `.` (relative to that root) |
| Instance/compute plan | Free |
| Health Check Path | `/healthz` |
| `MATHPAPER_ORT_THREADS` | `1` |
| `MATHPAPER_PUBLIC_DEMO` | `1` |
| `MATHPAPER_PUBLIC_CHAT` | `0` |

Leave the Docker command at its image default. The service reads Render's `PORT` automatically. Do not create a second service for MCP; the supervisor runs it privately inside the same container. The dependency versions used for local validation are pinned in `requirements.lock` and used by the Docker build.

## Run the INT8 profile locally

Set these values in your local `.env`, or as shell environment variables, before running the CLI:

```dotenv
MATHPAPER_MODEL_PRECISION=int8
MATHPAPER_INFERENCE_BATCH_SIZE=1
MATHPAPER_ORT_THREADS=1
MATHPAPER_DEFAULT_CANDIDATES=20
MATHPAPER_MAX_CANDIDATES=50
```

Then run `mathpaper models`, `mathpaper index`, and `mathpaper serve`. The download is about 44 MiB for the two quantized model files. The local CLI still defaults to FP32 unless configured; the Docker image defaults to INT8. To build an FP32 container instead, use `docker build --build-arg MODEL_PRECISION=fp32 -t mathpaper .`.

On Linux, `python evaluation/free_tier.py` exercises the full entry point, API, real inference, internal MCP, upper input length, and candidate cap, and records the sum of the individual process memory peaks. It does not emulate Render's CPU or memory quota.

## Option B: Hugging Face Docker Spaces

Current Hugging Face documentation requires a paid plan to create a Docker Space. Personal PRO is advertised at $9/month; CPU Basic has no additional hourly hardware charge, with 2 vCPU and 16 GB RAM. CPU Basic sleeps after 48 hours without traffic and wakes when visited. It is suitable for experimentation, with a cold-start tradeoff for CV visitors. Verify current terms when choosing a plan.

1. Create a Docker Space in your account and choose its intended visibility.
2. In a deployment copy of this folder, rename the main `README.md` to `PROJECT_README.md`. Copy `deployment/huggingface-README.md` to root `README.md`. Leave the full project source and deployment files in place.
3. Upload or push the project to that Space. The root Dockerfile will build and expose port 7860.
4. Confirm the app is running and test the real retrieval and original-PDF links before sharing its URL.

## Add the generating model later

Configure the LLM provider, URL, model, and API key using the hosting platform's runtime settings; keep API keys in its secret storage. The image never needs those keys at build time. The server currently supports a tool-calling Ollama service or a compatible Chat Completions service.

For a zero-cost prototype, a model with an available free API tier is a candidate. Google's Gemini API currently offers free input/output tokens for selected models, with limits. Stay on its free tier if the budget must remain zero. We have not configured or tested a live Gemini model; its tool-call metadata and structured-output behavior must be verified before enabling public chat. Free-tier service terms include use of submitted content to improve Google's products.

Before enabling open-ended generated answers for public traffic, add the chosen provider's spending cap and an application request limit. Keep the retrieval demo public while answer quality and cost are being evaluated. Simply naming a model does not establish that its endpoint, tool calling, or grounding quality works.

Runtime conversation history is currently local SQLite and may disappear on restart or redeployment without persistent storage. Benchmark questions and evaluation results belong in the repository or a persistent experiment store; do not rely on this container's writable disk as an experiment archive.

## Experiment and portfolio plan

The current interface supports changing retrieval method, reranked candidate count, and retained passage count. It returns measured retrieval latency and source passages. The existing ten-question seed set is a development smoke check, not a held-out benchmark.

Next, add a separately labelled evaluation set and a results page comparing:

- BM25, dense, hybrid, and hybrid + reranking.
- Candidate count and pair input length versus passage-level retrieval quality and latency percentiles.
- Single retrieval versus the agent's bounded follow-up lookups.
- Citation support, complete assumptions, abstention on unanswerable questions, token usage, and cost per answered question.

For a CV, link the live demo, reproducible source, and a concise technical report with measured results and failure examples. A short recorded walkthrough can remain available while an experimental backend sleeps or is redeploying.

Official references checked October 15, 2026:

- https://render.com/docs/docker
- https://render.com/docs/web-services
- https://render.com/pricing
- https://render.com/docs/free
- https://render.com/docs/compute-plans
- https://huggingface.co/docs/hub/spaces
- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/docs/hub/spaces-sdks-docker
- https://huggingface.co/docs/hub/spaces-gpus
- https://huggingface.co/pricing
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/openai
