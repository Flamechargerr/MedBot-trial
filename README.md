# MedRAG – Production-Oriented Fullstack Medical RAG System

MedRAG is a fullstack retrieval-augmented generation (RAG) system for medical Q&A with a Flask backend, FAISS retrieval, LLM generation, and a browser UI.

> ⚠️ **Safety notice:** MedRAG is for engineering demonstration and decision-support workflows only. It is not a medical device and must not be used as clinical advice.

---

## What's production-ready in this version

- Versioned API (`/api/v1/*`) with backward-compatible legacy routes
- Layered backend architecture (`app.py` + service layer + security layer)
- Environment profiles (`APP_ENV=local|staging|production`) with config validation
- Security controls: auth token support, request-size checks, rate limiting, strict security headers, CORS allowlist
- Runtime resilience: persisted initialization state, retrieval cache, controlled reindexing, retry/fallback generation behavior
- CI baseline with unit tests + static security scan
- Dockerized deployment with Gunicorn WSGI server

---

## Architecture

- `app.py` – Flask app factory, API routing, health endpoints
- `src/services/med_service.py` – orchestration service (init, chat, metrics, caching, runtime state)
- `src/security.py` – request guards (auth, content-type, size, rate limiting, headers, CORS)
- `src/config.py` – strict config loading/validation and environment profiles
- `src/retrieval/langchain_faiss_store.py` – vector index/retrieval
- `src/generation/llm_generators.py` – model invocation with retry/fallback
- `src/evaluation/metrics.py` – quality metrics

---

## API contract (v1)

### POST `/api/v1/init`
Request:
```json
{ "corpus_size": 200, "force_reindex": false }
```

Response:
```json
{ "status": "success", "message": "Loaded X docs in Ys." }
```

### POST `/api/v1/chat`
Request:
```json
{ "query": "...", "reference": "optional" }
```

Response:
```json
{
  "status": "success",
  "answer": "...",
  "baseline_answer": "...",
  "sources": [{"title": "...", "text": "..."}],
  "metrics": {
    "Latency": "0.12s",
    "RAG_ROUGE_L": 0.45,
    "Baseline_ROUGE_L": 0.30,
    "Accuracy_Improvement": "15.0%"
  }
}
```

### Health checks
- `GET /api/v1/health/live`
- `GET /api/v1/health/ready`

Legacy routes `/api/init` and `/api/chat` are retained for compatibility.

---

## Environment configuration

Required in production:
- `APP_ENV=production`
- `APP_AUTH_TOKEN=<token>`
- `GROQ_API_KEY=<key>` (or secret file equivalent)
- `CORS_ORIGINS=https://your-ui-domain.com`

Optional:
- `APP_SECRET_FILE=/path/to/secrets.json`
- `RATE_LIMIT_PER_MINUTE=60`
- `MAX_REQUEST_BYTES=65536`
- `MAX_QUERY_CHARS=2000`
- `DEFAULT_CORPUS_SIZE=200`
- `MAX_CORPUS_SIZE=2000`
- `FAISS_DB_DIR=./faiss_db`
- `RUNTIME_STATE_PATH=./runtime_state.json`
- `LLM_MAX_RETRIES=2`

If using token auth in the browser UI, set:
```js
localStorage.setItem('MEDRAG_API_TOKEN', 'your-token')
```

---

## Local development

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run unit tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3) Run app
```bash
python app.py
```

Open `http://127.0.0.1:5000`.

---

## Docker deployment

Build:
```bash
docker build -t medrag:latest .
```

Run:
```bash
docker run --rm -p 5000:5000 \
  -e APP_ENV=production \
  -e APP_AUTH_TOKEN=change-me \
  -e GROQ_API_KEY=your-key \
  -e CORS_ORIGINS=http://localhost:5000 \
  medrag:latest
```

---

## CI/CD baseline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- Unit tests (`unittest`)
- Bandit security scan

---

## Operational runbook (short)

- Verify liveness/ready endpoints before routing traffic
- Rotate API/auth secrets regularly and use `APP_SECRET_FILE` for managed secret injection
- Monitor error rate, request latency, and initialization times
- Reindex intentionally using `force_reindex=true` during controlled updates
- Keep explicit safety notice in all user-facing flows

---

## Limitations and next steps

- Add persistent distributed rate limiting (Redis) for multi-instance deployments
- Add real RBAC identity provider integration
- Add tracing/metrics stack (OpenTelemetry + Prometheus/Grafana)
- Add staged deployment and rollout/rollback automation
- Add safety regression benchmark suite with clinical expert review
