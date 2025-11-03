Scriza AI Microservices Plan

Overview
- Goal: decouple real‑time voice, LLM/TTS integrations, and strategy/memory from the monolith to scale calls and reduce coupling.
- Approach: start with latency‑critical paths (Telephony Media, TTS, LLM Orchestrator), then peel off Strategy and Usage.

Service Boundaries
- API Gateway/BFF
  - AuthN/Z, request routing, rate limiting, static assets, swagger docs.
- Auth & Users
  - JWT issuance/verify, roles/credits, password reset, OAuth.
- Strategy Service
  - Strategy ingest/normalize/merge, persona assets, versioning.
- Agent Orchestrator
  - Conversation state, prompt assembly, tool resolution, model policy.
- Telephony Media
  - Twilio /voice TwiML + WebSocket /stream, per‑call VAD/ASR loop, barge‑in, reply handoff.
- ASR Service
  - Streaming transcription (faster‑whisper/whisper.cpp); language hints per call.
- TTS Service
  - ElevenLabs proxy (HTTP+WS), audio format conversion to μ‑law, disk/object‑store cache.
- LLM Orchestrator
  - OpenAI proxy, model switching (critical vs cheap), TTL cache, usage metering.
- Usage & Billing
  - Tokens/minutes accounting, credit limits, alerts, daily summaries.
- Memory/Analytics
  - Memory logs, transcripts, summaries, reporting.

Telephony Media Service (spec)
- HTTP
  - POST `/voice`: returns TwiML with `<Connect><Stream url="wss://.../stream"/>`.
  - POST `/status-callback`: call lifecycle events (queued, ringing, completed, failed).
  - GET `/health` → `{status: ok}`.
- WebSocket `/stream`
  - Receives Twilio events: `start`, `media`, `mark`, `stop` (JSON per Twilio).
  - Emits to internal services:
    - ASR: `POST /v1/transcribe` (binary or base64 μ‑law chunk) → `{text, confidence}` or WS stream equivalent.
    - LLM Orchestrator: `POST /v1/turn` → `{text, usage}` (see below).
    - TTS: `POST /v1/speak` → `{audio_base64 | url, duration, cache_hit}`; stream frames back to Twilio.
  - Per‑call state: `call_id` (Twilio SID), `conversation_id` (same), stage, last transcript, last reply, confidence.
  - Timeouts: intro 10s → play polite fallback then hangup; silence timeout → SMS fallback.
- Data writes
  - Append turns to `calls` (Mongo) with `{user, agent, stage, confidence}`.
  - Update `calls.tokens_used` incrementally; set `calls.duration_seconds` on `stop`.

LLM Orchestrator (spec)
- POST `/v1/turn`
  - Request:
    ```json
    {
      "conversation_id": "<uuid|call_sid>",
      "persona": {"name": "Scrappy", "tone": "friendly", "locale": "en-IN"},
      "goals": ["book_demo"],
      "messages": [
        {"role": "system", "content": "You are a helpful sales AI."},
        {"role": "user", "content": "I'm busy right now"}
      ],
      "tools": [],
      "policy": {"critical": true, "temperature": 0.7}
    }
    ```
  - Response:
    ```json
    {
      "text": "Totally understand—can we pick a 10‑minute slot tomorrow?",
      "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
      "model": "gpt-4o",
      "cached": false
    }
    ```
- POST `/v1/stream-turn` (optional, later)
  - SSE/WebSocket streaming of partial tokens with final usage summary.
- GET `/v1/models` → available models and policy metadata.
- GET `/health` → `{status: ok}`.
- Behavior
  - If `policy.critical=true` → use `OPENAI_MODEL_CRITICAL`, else `OPENAI_MODEL_CHEAP`.
  - TTL cache: identical (model, persona, goals, messages) reused within `LLM_CACHE_TTL_SECONDS`.
  - Returns `usage` for downstream metering.

ASR Service (spec)
- WS `/v1/stream` or POST `/v1/transcribe`
  - Input: μ‑law 8k chunks or base64 frames; optional `language` hint.
  - Output: `{text, confidence}` with timestamps.

TTS Service (spec)
- POST `/v1/speak`
  - Request:
    ```json
    {"text": "Hello!", "voice_id": "<id>", "tone": "friendly", "locale": "en-IN"}
    ```
  - Response:
    ```json
    {"audio_base64": "<mp3_b64>", "duration": 1.4, "cache_hit": true}
    ```
  - Stores/serves cached assets from object store; optional signed URLs for Twilio `<Play>`.

Usage & Billing (spec)
- Events consumed: `usage.llm`, `usage.tts`, `call.completed`.
- REST: GET `/api/v1/usage/summary?since=today` → totals + top agents.
- REST: POST `/api/v1/credit/check` → `{allowed: bool, remaining: number}`.

Data & Ownership
- Keep Mongo per service (collections scoped):
  - Telephony: `calls` (owner), writes call/turns/metrics.
  - Strategy: `strategies` (owner), stores versions and persona.
  - Memory: `memory_logs` (owner), stores conversational artifacts.
  - Usage: `usage_events`, aggregates projections for summaries.
- Object store (S3/MinIO): `voice-assets/` (prefetched prompts) and `tts-cache/`.

Communication Patterns
- Sync HTTP/gRPC: Gateway → Services; Telephony → LLM/ASR/TTS; Orchestrator → LLM/TTS.
- Async events (Kafka/NATS/SNS+SQS):
  - Topics: `call.started`, `utterance.transcribed`, `reply.generated`, `audio.synthesized`, `call.updated`, `call.completed`, `usage.updated`, `credit.threshold`.
- Idempotency keys: Twilio webhooks, LLM/TTS calls; correlation via `call_id`.

Security
- JWT on ingress; mTLS/service tokens between services.
- PII redaction in logs; secrets via vault; strict CORS at gateway.

SLOs & Observability
- KPIs: P50/P95 turn latency, ASR confidence, cache hit rate, WS disconnects, token/minute.
- Tracing: propagate `call_id` and `x-request-id`; metrics per service; structured logs.

Phased Migration Plan
1) Extract Telephony Media
   - Move `/voice` + `/stream` into a new FastAPI service.
   - Keep ASR (local lib) initially; call monolith’s existing services for LLM/TTS over HTTP.
2) Extract TTS
   - Introduce TTS service with cache; point Telephony to it; store audio in object store.
3) Extract LLM Orchestrator
   - Centralize model policy and caching; update Telephony/Agent to call it.
4) Extract Usage & Billing
   - Publish usage events; enforce credit checks on turn requests.
5) Extract Strategy Service
   - Move ingest/normalize/merge; pre-batch voice assets on ingest.

Backwards Compatibility
- Preserve current API routes at gateway; proxy to services.
- Gradually switch internal imports to HTTP clients; retire direct module calls.

Minimal Infra (start)
- MongoDB, Redis (optional), MinIO/S3, NATS or Kafka, Prometheus + Grafana, ELK/OTel.

