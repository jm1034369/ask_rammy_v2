# Ask Rammy — WCU HR Chatbot

Rammy is an AI-powered HR assistant for West Chester University. It answers HR-related questions using content sourced from official WCU and PASSHE HR pages, retrieved via semantic vector search. On the `local-generation` branch, Ollama generates ordinary answers and contextual affirmative follow-ups using `llama3.2:3b`. OpenAI's GPT-4.1-mini still handles two decline paths, so the application is not fully local yet.

---

## Project Structure

```
ask_rammy_v2/
├── frontend/
│   ├── embed.html          # WCU HR page with the chatbot embedded
│   ├── analytics.html      # Analytics dashboard
│   ├── chat.js             # Chat widget logic — rendering, quick-reply chips, API calls
│   └── styling.css         # Scoped styles for the chat widget
│
├── backend/
│   ├── chatbot_api.py      # Python Flask service — all chatbot logic, PDF URL resolution
│   ├── local_orchestrator.py # Optional local search-query rewriting
│   ├── local_generator.py   # Local answer generation through Ollama
│   ├── qdrant_setup.py     # One-time script to populate the Qdrant vector database
│   └── requirements.txt    # Python dependencies
│
├── server/
│   ├── server.js           # Node.js API gateway + MinIO PDF proxy
│   └── package.json        # Node.js dependencies
│
├── docker/
│   ├── Dockerfile.python   # Python service container
│   ├── Dockerfile.node     # Node.js service container
│   ├── docker-compose.yml  # Orchestrates all five services
│   └── .env                # Local configuration and secrets (not committed)
│
└── docs/
    └── local-orchestrator-guide.md # Detailed query-planning walkthrough
```

---

## Architecture

```text
Browser → Node /api/chat → Flask /chat
                             │
                             ├─ PII checks and guided-flow routing
                             │
                             └─ Ordinary HR question
                                  ↓
                           Local orchestrator (Ollama)
                             rewrites the search query
                                  ↓
                           Embedding model → Qdrant
                             retrieves HR source chunks
                                  ↓
                           Local generator (Ollama)
                             writes answer using sources,
                             history, and original question
                                  ↓
                           Reply → Node → Browser

qdrant_setup.py → WCU/PASSHE web pages + PDFs from MinIO → Qdrant
Browser PDF links → Node /api/pdf → MinIO

No retrieved context or OUTOFSCOPE answer → OpenAI decline → Reply
```

The orchestrator and generator share the Ollama server but have separate jobs
and model settings. Disabling orchestration skips query rewriting; it does not
disable local answer generation. See the
[orchestrator guide](docs/local-orchestrator-guide.md) for the query-planning details.
That guide describes an earlier source snapshot; this README describes the current
answer-generation routing on `local-generation`.

| Response path | Current handler |
|---|---|
| Ordinary HR question with usable retrieved sources | Ollama generator |
| Recognized small talk requiring generated text | Ollama generator |
| Affirmative follow-up with usable assistant history and retrieved context | Ollama generator |
| Identity, capability, privacy warning, or intermediate eligibility question | Prepared Python response |
| No usable retrieved context | OpenAI decline |
| Ordinary generated answer contains `OUTOFSCOPE` | OpenAI replacement decline |

The planner falls back to the original search question on handled failures.
The generator has no equivalent automatic cloud fallback: its errors propagate
to the chat route. An affirmative answer containing `OUTOFSCOPE` falls through
to greeting-style handling rather than returning that answer.

### Services at a Glance

| Service | Port | Purpose |
|---|---|---|
| Node.js (Express) | 3000 | API gateway, rate limiting, PDF proxy |
| Python (Flask) | 5001 | Chatbot logic, vector search, LLM calls |
| Ollama | 11434 | Local query rewriting and answer generation |
| Qdrant | 6333 | Vector database — semantic retrieval |
| MinIO | 9000 / 9001 | PDF object storage / web console |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)
- An OpenAI API key — still required by backend startup and the remaining decline paths. Get one at [platform.openai.com](https://platform.openai.com).

---

## First-Time Setup

### 1. Clone the repository

```bash
git clone --branch local-generation https://github.com/RichDBrown/ask_rammy_v2.git
cd ask_rammy_v2
```

### 2. Create your `docker/.env` file

Create `docker/.env` with the following example values. If the file already exists,
update the relevant entries instead of replacing it.

```dotenv
OPENAI_API_KEY=replace-with-your-key
OPENAI_ORG_ID=
OPENAI_PROJECT_ID=
MINIO_USER=minioadmin
MINIO_PASS=minioadmin
MINIO_BUCKET=documents
LOCAL_ORCHESTRATOR_ENABLED=true
LOCAL_MODEL_BASE_URL=http://ollama:11434
LOCAL_MODEL=llama3.2:3b
LOCAL_MODEL_TIMEOUT=4
LOCAL_GENERATOR_MODEL=llama3.2:3b
LOCAL_GENERATOR_TIMEOUT=90
```

Use your own OpenAI API key. `.env` is ignored by Git; do not commit it.
The 90-second generation timeout is a development setting for slower local
inference; the code defaults to 20 seconds when the variable is absent.

### 3. Start Docker Desktop

Open Docker Desktop and wait for the whale icon to stop animating.

### 4. Build and start all services

```bash
cd docker
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama list
docker compose up --build -d
```

Team members do not need Ollama installed on their host. Docker Desktop runs
Ollama and the rest of the application services.

The first build downloads large dependencies and may take several minutes. Model
loading and response times depend on available hardware. If you select different
planner and generator models, download both with `ollama pull` before testing.

### 5. Populate the vector database

Fetches the configured WCU/PASSHE HR web sources and PDFs in MinIO, chunks and
embeds them, and uploads to Qdrant. Run once after the first build and again
when sources or PDFs change. The script deletes and recreates the `rammy_hr`
collection; this is a full rebuild of the index, not an incremental update.

```bash
docker compose exec python python qdrant_setup.py
```

The script reports how many vectors were indexed into `rammy_hr`; the count varies with source content.

### 6. Open the frontend

Open `frontend/embed.html` from the project root in your browser, or right-click
it in VS Code and select **Open with Live Server**. Node provides the API rather
than serving this HTML page. The widget currently calls
`http://localhost:3000/api`, so this setup assumes the browser runs on the same
computer as Docker.

---

## Day-to-Day Usage

Run either block from the project root. If already in `docker/`, omit `cd docker`.

**Start the app:**

```bash
cd docker
docker compose up -d
```

**Shut down:**

```bash
cd docker
docker compose down
```

> Qdrant data, MinIO documents, and Ollama models persist in named Docker volumes — you do **not** need to re-run `qdrant_setup.py` on every restart.

---

## Local Generation Configuration

Compose loads `docker/.env` into the Python container. Use these exact names:

| Variable | Default in Docker setup | Purpose |
|---|---|---|
| `LOCAL_ORCHESTRATOR_ENABLED` | `true` | Enable query rewriting; standalone Python class defaults to `false` |
| `LOCAL_MODEL_BASE_URL` | `http://ollama:11434` | Ollama endpoint shared by planner and generator |
| `LOCAL_MODEL` | `llama3.2:3b` | Query-rewriting model |
| `LOCAL_MODEL_TIMEOUT` | `4` | Planner request timeout in seconds |
| `LOCAL_GENERATOR_MODEL` | `llama3.2:3b` | Answer-generation model |
| `LOCAL_GENERATOR_TIMEOUT` | `20` | Generator request timeout in seconds; setup example overrides it to `90` |

`LOCAL_GENERATOR_URL` and misspelled names such as `LOCAL_GENRATOR_TIMEOUT`
are not read by the code. The Docker service hostname `ollama` is for container
communication; a host-run generator can use `http://localhost:11434` instead.

After editing `.env`, recreate Python so it receives the new environment.
After changing Python or Node source, rebuild the corresponding image.
Run these commands from `docker/`:

```bash
# Apply environment changes
docker compose up -d --force-recreate python

# Apply source changes
docker compose up -d --build python node
```

The generator waits for a complete answer (`stream: False`), with temperature
`0.3` and an output limit of `300` tokens (`num_predict`). First requests may be
slower because Ollama loads the model. The Node chat proxy has a separate
120-second timeout for the whole backend request. Increasing either timeout
allows more waiting; it does not speed up inference.

## Testing Local Generation

Run terminal commands in this section from `docker/`, with services running
and the configured model downloaded.

### 1. Test the generator independently

This calls Ollama directly through `LocalGenerator`, without Qdrant or OpenAI:

```bash
docker compose exec -T python python -u - <<'PY'
import time
from local_generator import LocalGenerator

generator = LocalGenerator()
print("Server:", generator.base_url)
print("Model:", generator.model)
print("Timeout:", generator.timeout, "seconds")

messages = [
    {
        "role": "system",
        "content": (
            "Answer only from this fictional test source: "
            "The example HR office opens at 9 AM and closes at 5 PM. "
            "If the source does not contain the answer, say you don't know."
        ),
    },
    {"role": "user", "content": "When does the example HR office open?"},
]

started = time.monotonic()
try:
    print("Answer:", generator.generate(messages))
finally:
    print("Elapsed:", round(time.monotonic() - started, 2), "seconds")
PY
```

Expect an answer stating **9 AM**. With the setup example above, the printed
timeout should be **90 seconds**. Failures display a Python traceback.

### 2. Test the full chat route

This also exercises Node, Flask, query planning, and retrieval. Qdrant must
already contain indexed documents.

```bash
curl -sS -w '\nTotal time: %{time_total}s\n' \
  http://localhost:3000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What retirement plans are available at WCU?","history":[]}'
```

Expect JSON containing a `reply`. Review its accuracy and source links; a
successful HTTP request does not establish answer quality.

### 3. Test through the browser

Open `frontend/embed.html` from the project root in your browser, or use
VS Code's **Open with Live Server** if installed. Try an HR question, a
contextual follow-up, an out-of-scope question, and a guided eligibility flow.
Use browser Developer Tools → Network → the `chat` request to inspect status,
response, and duration.

For recent server evidence:

```bash
docker compose exec ollama ollama list
docker compose logs --since=5m --tail=100 python ollama node
```

Ollama health checks (`GET /api/tags`) are not generation requests. Both the
planner and generator use `POST /api/chat`; a successful Ollama call alone does
not prove the final answer was local. Logs currently lack a shared request ID
and explicit final-answer provider, so timing can support attribution but does
not provide an exact response-text match.

### Current limitations

- Ordinary HR answers, generated small talk, and contextual affirmative
  follow-ups use the local generator. Identity replies, privacy warnings, and
  intermediate guided-flow questions use prepared text.
- Two decline paths still use OpenAI: no retrieved context, and an answer
  containing `OUTOFSCOPE`. Backend startup still requires `OPENAI_API_KEY`.
  Those paths send a prompt containing the question to OpenAI.
- A generator timeout or invalid/empty output raises an error; the chat route
  currently returns HTTP 500 rather than automatically falling back to OpenAI.
- Local generation token usage is not recorded in analytics; zero counters do
  not mean inference used no tokens.
- Guided eligibility progress is stored in a process-wide `FLOW_PROGRESS`
  dictionary, so concurrent conversations are not isolated. This still needs
  per-session state before multi-user use.
- Flask health checks do not verify that Ollama can generate an answer or that
  the chosen model is installed. Ollama's health check only runs `ollama list`.

---

## MinIO — PDF Document Storage

MinIO is an S3-compatible object storage server. HR staff can upload PDF documents (handbooks, contracts, benefit guides, collective bargaining agreements, etc.) and Rammy will include them in answers. PDF links in chat responses are clickable and open directly in the browser.

### How PDF Linking Works

When a PDF is indexed, its source is stored as `pdf:filename.pdf` in Qdrant. The Python backend resolves this to a URL (`http://localhost:3000/api/pdf/filename.pdf`) before passing context to the LLM. When a user clicks the link, Node.js fetches the file from MinIO using server-side credentials and streams it to the browser — MinIO credentials are never exposed to the client.

### Uploading PDFs

1. Open the MinIO console: [http://localhost:9001](http://localhost:9001)
2. Log in with `MINIO_USER` / `MINIO_PASS` from `.env` (default: `minioadmin` / `minioadmin`)
3. Create a bucket named exactly `documents` (first time only)
4. Click into the bucket → **Upload** → **Upload Files** → select your PDFs
5. Re-run the setup script to index them:

```bash
docker exec rammy-python python qdrant_setup.py
```

> **Note on filenames:** Spaces in filenames are fully supported (e.g. `Dental Benefits Summary.pdf`). Avoid colons (`:`) as they interfere with URL routing.

### Verifying MinIO

```bash
# Health check — expect 200 OK
curl http://localhost:9000/minio/health/live

# List all indexed PDFs
docker exec rammy-python python -c "
import os
from minio import Minio
client = Minio(
    os.getenv('MINIO_HOST', 'minio:9000'),
    access_key=os.getenv('MINIO_USER', 'minioadmin'),
    secret_key=os.getenv('MINIO_PASS', 'minioadmin'),
    secure=False,
)
for o in client.list_objects(os.getenv('MINIO_BUCKET', 'documents')):
    print(o.object_name)
"
```

---

## Refreshing HR Sources

To re-index all web sources and PDFs:

```bash
docker exec rammy-python python qdrant_setup.py
```

The chat window’s ⋮ → **Refresh HR sources** action only reconnects the backend
to Qdrant. It does not fetch changed pages, ingest new PDFs, or rebuild the index.
The same reconnect action is available through:

```bash
curl -X POST http://localhost:3000/api/refresh
```

---

## Health Check

```bash
curl http://localhost:3000/api/health
```

Expected response:

```json
{ "status": "ok", "python": "reachable" }
```

This reports backend readiness, not proof of a successful retrieval or model
response. Check the Qdrant dashboard at
[http://localhost:6333/dashboard](http://localhost:6333/dashboard) to confirm the
`rammy_hr` collection exists, then run the generation and full-chat tests above.

---

## Analytics

Open `frontend/analytics.html` in your browser to view usage metrics. Data is served from the Python backend via `GET /api/analytics`.

---

## Common Issues

| Symptom | Fix |
|---|---|
| First build taking several minutes | PyTorch and ML dependencies are large |
| Ollama model not found | From `docker/`, run `docker compose exec ollama ollama pull llama3.2:3b` (or your configured model). |
| Generator times out at 20 seconds despite `.env` changes | Check the exact spelling `LOCAL_GENERATOR_TIMEOUT`, then recreate Python. |
| Chat returns an internal error during local generation | Inspect Python/Ollama logs; connection errors, empty output, and generation timeouts currently return HTTP 500. |
| Node request times out | Its chat timeout is 120 seconds for the entire Python request, including planning, retrieval, and generation. |
| `Connection refused` on port 3000 | Make sure Docker Desktop is running, then `docker compose up -d` |
| `python unreachable` in health check | `docker logs rammy-python` to diagnose |
| Port already in use | `docker compose down` then `docker compose up -d` |
| `.env not found` error | Complete Step 2 in the setup instructions |
| Bot only gives out-of-scope replies | Check retrieval logs and collection contents; index sources if the collection is missing or empty. |
| Bot deflects after a restart | Check Qdrant availability and collection contents. Restarting Python does not erase the persistent collection. |
| `rammy_hr` collection not found | Run `docker compose exec python python qdrant_setup.py` from `docker/`. |
| MinIO console not loading | `docker logs rammy-minio` |
| PDFs not being indexed | Confirm bucket is named exactly `documents`, then re-run `qdrant_setup.py` |
| PDF link returns `Could not retrieve document` | Check `docker logs rammy-python` for MinIO errors |
| PDF filenames with colons failing | Rename the file in MinIO (remove the colon), then re-run `qdrant_setup.py` |
| Mac AirPlay conflict on port 5000 | Already handled — app uses port 5001 |
| `ModuleNotFoundError` | Always run scripts inside Docker: `docker exec rammy-python python <script>.py` |

---

## Useful Commands

```bash
# View logs for each service
docker logs rammy-python
docker logs rammy-node
docker logs rammy-qdrant
docker logs rammy-minio
docker logs rammy-ollama

# Open an interactive shell inside the Python container
docker exec -it rammy-python bash

# Rebuild after changing backend or server source (from project root)
docker compose -f docker/docker-compose.yml up -d --build python node

# Check vector count in Qdrant
curl http://localhost:6333/collections/rammy_hr

# List all PDFs in MinIO
docker exec rammy-python python -c "
import os
from minio import Minio
client = Minio(
    os.getenv('MINIO_HOST', 'minio:9000'),
    access_key=os.getenv('MINIO_USER', 'minioadmin'),
    secret_key=os.getenv('MINIO_PASS', 'minioadmin'),
    secure=False,
)
for o in client.list_objects(os.getenv('MINIO_BUCKET', 'documents')):
    print(repr(o.object_name))
"
```

---

## HR Source URLs

Rammy draws web knowledge from the following official pages. To add or remove sources, edit `SOURCE_URLS` in `backend/qdrant_setup.py` and re-run the script.

**WCU HR:** FAQs, Employee & Labor Relations, Student Employment, Professional Development, Why Work at WCU, Job Openings  
**Benefits & Leave:** Employee Benefits by Group, FMLA, PASSHE Life Events, Work-Related Injuries, Tuition Waiver  
**Retirement:** PASSHE Retirement (ARP, SERS, TSA, Deferred Compensation, Voluntary Plans), TIAA, Retirement@Work, SERS, PSERS, Empower, Fidelity  
**Other:** USCIS I-9, Payroll, Parking (Permits, Regulations, FAQs), WCU Academic Calendar  

PDF sources are managed separately via the MinIO console.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JS, HTML/CSS |
| API Gateway | Node.js + Express |
| Chatbot Backend | Python + Flask |
| Vector Search | Qdrant + sentence-transformers (`all-MiniLM-L6-v2`) |
| Local LLM | Ollama (`llama3.2:3b` by default) |
| Remaining cloud decline paths | OpenAI GPT-4.1-mini |
| PDF Storage | MinIO (S3-compatible) |
| Containerization | Docker + Docker Compose |

---

## Contributing

This project is developed as part of CSC402 at West Chester University. Local-generation migration is still in progress; see [Current limitations](#current-limitations) before removing OpenAI configuration.
