# Scrappy Singh Operational Guide

Scrappy Singh is the lead conversational agent for the Scriza platform. The refactored
`BaseAgent` (“the brain”) powers persona-aware conversations, real-time delegation, and
self-improvement loops. This guide explains how to configure, orchestrate, and test the
production stack.

---
## Architecture Snapshot
- FastAPI entrypoint in `main.py:42` wires authenticated routers for strategy ingest, call orchestration, voice tooling, memory access, and telephony integrations.
- Core agent logic in `apps/agents/agent_base.py:17` owns persona management, GPT prompting, streaming controls, memory logging, subagent registration, and the self-improvement loop.
- Sales-specialised behaviours in `apps/agents/sales/agent.py:34` extend the base agent with objection handling, stage tracking, fallback routing, and the `/generate_reply` API surface.
- Strategy ingestion in `apps/agents/sales/strategy_ingest.py:83` normalises business-facing payloads, captures uploads, and persists them to Mongo `COLLECTION_STRATEGIES`.
- Shared services in `apps/agents/sales/services.py:146` broker OpenAI chat completions, ElevenLabs synthesis (`apps/agents/sales/services.py:354`), Twilio call creation (`apps/agents/sales/services.py:434`), and call persistence utilities.
- Real-time telephony bridging in `apps/integrations/telephony/twilio.py:799` streams audio over Twilio, hands μ-law frames into the modular transcriber pipeline (`apps/integrations/telephony/twilio.py:996`), generates replies, and streams ElevenLabs speech back to the caller.
- ASR/translation kernel modules (`apps/agents/transcriber.py`, `apps/agents/translation.py`) modularise on-device Whisper backends, voice-activity detection, and locale-aware translation workflows.
- Tooling + observability live in `apps/agents/tools.py`, `apps/agents/subagent_manager.py`, and the ops dashboard router `apps/ops/router.py` powering `/ops` with live-call, feedback, and export panels.

## Technology Stack
- FastAPI + Pydantic power the HTTP layer (`main.py:42`, `apps/agents/sales/voice_handler.py:24`) with scoped role enforcement via `RoleChecker`.
- MongoDB is accessed asynchronously using Motor (`core/database.py:1`), backing strategies, agents, calls, and conversational memory.
- OpenAI GPT-4o drives dialogue generation in `call_openai_chat` (`apps/agents/sales/services.py:146`) while the modular ASR pipeline keeps faster-whisper/whisper.cpp on-device (`apps/agents/transcriber.py`).
- ElevenLabs text-to-speech runs over both REST and WebSocket pathways (`apps/agents/sales/services.py:354`, `apps/integrations/telephony/twilio.py:878`) with tuning flags surfaced in `core/config.py:31`.
- Twilio Programmable Voice delivers PSTN connectivity, TwiML webhooks, and media WebSockets (`apps/integrations/telephony/twilio.py:226`, `apps/integrations/telephony/twilio.py:799`) for two-way audio streaming.
- Observability outputs include Mongo call records and Excel call summaries triggered in `apps/agents/sales/call_handler.py:142` via `utils/excel_logger.py`.

## Cost Optimization & Usage
- Model switching
  - Critical (live in-call): `OPENAI_MODEL_CRITICAL` (default: `gpt-4o`).
  - Non-critical (drafts/summaries): `OPENAI_MODEL_CHEAP` (default: `gpt-4o-mini`).
- LLM caching
  - `apps/agents/llm_client.py` caches identical non-stream requests for `LLM_CACHE_TTL_SECONDS` (default 30s).
- TTS caching
  - `apps/agents/sales/services.py:synthesise_elevenlabs_voice` caches MP3 by hash of text+voice+tone+locale under `TTS_CACHE_DIR` (default `static/voice_cache`).
- Usage metrics
  - `calls.tokens_used`: incremented per LLM turn when OpenAI returns usage.
  - `calls.duration_seconds`: set on Twilio WS stop event.

### New environment variables
Add to `.env` (defaults shown):
```
OPENAI_MODEL_CRITICAL=gpt-4o
OPENAI_MODEL_CHEAP=gpt-4o-mini
LLM_CACHE_TTL_SECONDS=30
TTS_CACHE_ENABLED=true
TTS_CACHE_DIR=static/voice_cache
```

---
## Local Dev MongoDB
- If running locally without Docker Compose, set:
```
MONGODB_URL=mongodb://127.0.0.1:27017
```
- The connector (`core/database.py`) fast-fails and falls back from `mongodb://mongodb:27017` → `mongodb://localhost:27017` if the Docker hostname is unreachable.

---
## Behavior Highlights (current)
- Persona injection centralized in `llm_client`; services no longer mutate message payloads.
- Twilio WebSocket updates `calls.tokens_used` during conversation and sets `calls.duration_seconds` on stop.
- ElevenLabs streaming preferred when `ELEVENLABS_USE_WS_TTS=true`, with HTTP fallback and μ-law conversion.

---
## Roadmap (next)
- Pre-batch greeting/intro/fallback prompts during strategy ingestion and persist under voice assets.
- Add `/api/v1/usage/summary` with daily minutes, token totals, and top agents.
- Budget enforcement hooks: `users.usage_total`, `users.credit_limit` with 80% alerts.
- Silence timeout auto-hangup + SMS follow-up flow in telephony integration.

## Sovereign Kernel Enhancements
- **Streaming LLM Engine** – `apps/agents/llm_client.py` drives GPT token streams through `asyncio.Queue`s, enabling sub-token speech buffering and cancellation via `BaseAgent.cancel_active_stream()`.
- **Local-first ASR** – `apps/agents/transcriber.py` abstracts faster-whisper and whisper.cpp backends with VAD-aware μ-law ingestion. Twilio streams call `StreamingTranscriber.transcribe_chunk()` keeping audio on the box.
- **Translation middleware** – `apps/agents/translation.py` switches between passthrough, NLLB, or Bhashini for persona locales, while `BaseAgent.generate_response` auto-translates user input/output pairs.
- **Emotion + tone overrides** – Persona payloads accept `tone_override`, influencing both system prompts and ElevenLabs stability/similarity settings.
- **Tool / subagent kernel** – `apps/agents/tools.py` & `apps/agents/subagent_manager.py` expose pluggable functions (`search_docs`, `fetch_crm_data`, etc.) with TTL-managed subagents for delegation.
- **Memory summariser** – `BaseAgent.summarize_memory()` synthesises objection trends every five calls and logs `event_type="summary"` for auditing.

## Sales Calling Flow (Detailed)
1. Strategy ingestion via `/api/v1/agent/sales/ingest_strategy` stores a normalised payload in Mongo (`apps/agents/sales/strategy_ingest.py:83`), capturing persona, scripts, objections, and business context.
2. When `POST /api/v1/agent/sales/start_call` runs, the latest strategy is fetched (`apps/agents/sales/services.py:135`), a `SalesAgent` is initialised (`apps/agents/sales/call_handler.py:101`), and a Twilio call is created through `create_twilio_call` (`apps/agents/sales/services.py:434`).
   - *Tech stack*: Twilio Programmable Voice for PSTN origination, a FastAPI endpoint for call setup, and Python’s Motor client for Mongo persistence.
3. Twilio immediately invokes `/voice`, which returns a `<Connect><Stream>` directive pointing to the Scriza WebSocket endpoint (`apps/integrations/telephony/twilio.py:226`), enabling low-latency media streaming.
4. The WebSocket loop (`apps/integrations/telephony/twilio.py:799`) loads call context, plays an ElevenLabs greeting (REST or WebSocket), buffers μ-law frames, runs voice-activity detection, and batches caller audio. Custom tooling stitches together asyncio IO, websockets, and ElevenLabs TTS.
5. Completed utterances are handed to `StreamingTranscriber.transcribe_chunk()` (`apps/integrations/telephony/twilio.py:1001`); faster-whisper or whisper.cpp perform on-device ASR while translation middleware normalises language before handing transcripts to `SalesAgent.generate_sales_response` (`apps/agents/sales/agent.py:106`).
6. Responses are voiced through ElevenLabs and streamed back to Twilio (`apps/integrations/telephony/twilio.py:878`), while each turn persists to `COLLECTION_CALLS` and memory logs (`apps/integrations/telephony/twilio.py:1003`, `apps/agents/agent_base.py:194`).
   Dynamic fallback policies fire if ASR confidence dips or TTS fails, logging `fallback_triggered` events and streaming temporary silence to preserve the voice session.

> **Improvement ideas:**  
> • Swap Twilio WebSocket streaming for media gateway alternatives (e.g., SIPREC, WebRTC SFU) to reduce vendor lock-in and latency.  
> • Introduce jitter buffering and packet loss concealment to smooth noisy PSTN legs.  
> • Pre-warm the ElevenLabs WS connection per call to shave first-token latency; explore local neural TTS for full sovereignty.  
> • Layer in sentiment/emotion classifiers on the ASR output to dynamically adapt tone overrides mid-call.
7. `start_call` collates metadata, writes Excel summaries, and hands conversation data to the daily self-improvement loop (`apps/agents/sales/call_handler.py:142`, `apps/agents/agent_base.py:90`).

## Ops Monitoring & Observability
- `/ops` serves a lightweight React console (`apps/ops/router.py`) summarising live calls, feedback timelines, persona state, and downloadable Excel exports from `logs/calls/`.
- `GET /api/v1/ops/live_calls` streams current speaker, transcript, and stage via the in-memory `LiveCallRegistry` fed by the Twilio media loop.
- `GET /api/v1/ops/feedback_timeline` collates `feedback`, `train_loop`, `fallback_triggered`, and `summary` memory events for QA.
- `GET /api/v1/ops/excel_logs` enumerates xlsx exports with a companion download endpoint for audit trails.
- All ops routes are admin-gated through `RoleChecker` and reuse the same FastAPI authentication middleware as the main APIs.

## Business Agent Wizard
- The dashboard exposes a **Business Agents** panel that walks through industry → use-case → details.
- Reference metadata comes from `GET /api/v1/agent/sales/agents/metadata` (lists shown in the design screenshots).
- On submission the front-end posts to `POST /api/v1/agent/sales/agents/business` with:
  - Selected industry & use case IDs
  - Agent name, primary goal, optional website URL/persona hints
  - `ai_generate=true` triggers website scraping + GPT strategy draft (`draft_strategy_from_context`).
- The backend scrapes/summarises website copy (BeautifulSoup + GPT), generates JSON strategy, normalizes it, and calls `SalesAgent.train()` so the agent is immediately configured.
- The Strategy Studio form is automatically pre-filled with the generated payload so users can review/tweak before ingestion.

## 1. Train the Agent

### 1.1 REST Ingestion Endpoint
- **Route:** `POST /api/v1/agent/sales/ingest_strategy`
- **Auth:** Bearer token with `user` or `admin` role
- **Body:** `StrategyPayload` (see below)
- **Effect:** Persists strategy → calls `SalesAgent.train()` → seeds `BaseAgent` with
  persona, goals, strategy, subagent specs.

### 1.2 Strategy Payload Skeleton
```json
{
  "title": "Enterprise SaaS Sales",
  "description": "Playbook for scheduling pilot demos",
  "goals": ["book_demo", "capture_feedback"],
  "persona": {
    "name": "Scrappy Singh",
    "tone": "friendly, confident, empathetic",
    "tone_override": "assertive",
    "description": "An energetic Indian sales specialist in their late 20s",
    "voice_id": "0YniYnwPKdgbGysKiJSN",
    "locale": "en-IN"
  },
  "scripts": {
    "greeting": "Namaste! This is Scrappy from Scriza.",
    "pitch": "We help your SDRs close more revenue with AI copilots.",
    "faqs": {
      "pricing": "We start at $499/mo with volume discounts.",
      "security": "SOC2 compliant with granular PII controls."
    },
    "objections": {
      "busy": "Totally hear you—how about a 10-minute slot tomorrow?",
      "budget": "Let me walk you through ROI numbers from similar teams."
    }
  },
  "closing_techniques": [
    "Let’s lock in a quick demo—tomorrow morning or afternoon?"
  ],
  "fallback_scenarios": {
    "compliance": "I’ll have our compliance officer follow up within the hour."
  },
  "fallback_policies": [
    {
      "trigger": "confidence_below_0.5",
      "threshold": 0.45,
      "action": "prompt_for_clarification"
    }
  ],
  "subagents": [
    {
      "name": "followup_bot",
      "class": "apps.agents.sales.agent.FallbackAgent",
      "init": {
        "agent_id": "scrappy-followup",
        "user_id": "<user-id>",
        "name": "Scrappy Follow-up",
        "fallback_map": {
          "email": "I’ll send you a recap with pricing in a few minutes."
        }
      }
    }
  ]
}
```

> **Tip:** Any subagent listed under `subagents` must resolve to a `BaseAgent` subclass
> via dotted path. On training, the base agent imports, instantiates, and registers the
> subagent automatically.

---
## 2. Real-Time Webhooks & ASR

`BaseAgent.ingest_webhook_payload(payload)` handles live inputs from telephony, chat, or
ASR services. Wire your public endpoint to forward payloads into it.

### 2.1 Sample FastAPI hook
```python
@router.post("/api/v1/hooks/twilio")
async def twilio_webhook(body: dict, agent: BaseAgent = Depends(load_agent)):
    """Raw Twilio transcription → Scrappy Singh"""
    updates = await agent.ingest_webhook_payload({
        "adapter_key": "twilio_stream",
        "transcript": body.get("SpeechResult"),
        "auto_reply": True,
        "intent": body.get("DetectedIntent"),
        "stream": False
    })
    return {"status": "ok", "updates": updates}
```

### 2.2 Payload Fields
| Field | Type | Purpose |
|-------|------|---------|
| `persona` | object | Patch persona traits on the fly (tone shift, etc.) |
| `goals` | array | Update current goals, e.g., `["collect_feedback"]` |
| `strategy` | object | Merge into active strategy document |
| `adapter_key` | string | Routes to a registered adapter (e.g., `twilio_stream`) |
| `transcript`/`text` | string | Lead utterance → auto reply via `generate_response()` |
| `intent` | object | Contains `delegate_to` for subagent handoff |
| `spawn_subagent` | object | One-off subagent bootstrap (same structure as in strategy) |
| `auto_reply` | bool | Toggle automatic replies for this payload |
| `stream` | bool | Forward tokens through the streaming callback |

`BaseAgent` automatically logs turns (`log_turn`) and webhook events into
`memory_logs`, keeping the conversational audit trail.

### 2.3 Registering an Adapter
```python
async def twilio_adapter(payload: dict) -> dict:
    transcript = payload.get("transcript", "")
    sentiment = detect_sentiment(transcript)
    await agent.log_feedback(
        feedback=f"Live sentiment: {sentiment}",
        source="twilio_adapter",
        importance=2,
    )
    return {"sentiment": sentiment}

agent.register_webhook_adapter("twilio_stream", twilio_adapter)
```

---
## 3. Streaming GPT-4o + ElevenLabs

1. Call `await agent.generate_response(text, stream=True, on_token=callback)` to receive
   incremental GPT-4o tokens.
2. Use `callback` to forward tokens into your preferred speech buffer or UI stream.
3. After the full turn, call `await agent.speak_response(reply_text)` to generate the
   synced ElevenLabs voice asset. The helper already respects `persona.voice_id`.
4. For Twilio media streams, convert the MP3 bytes to μ-law frames with
   `_mp3_to_mulaw_chunks` (already provided in `apps/integrations/telephony/twilio.py`) and
   push them over the websocket.

---
## 4. Daily Self-Evolution Loop

Schedule `await agent.train_loop()` (e.g., via Celery or APScheduler) once per day:
```python
results = await agent.train_loop()
if results["status"] == "updated":
    logger.info("Scrappy refined itself: %s", results["improvement"])
```
The base agent:
- Summarises buffered feedback.
- Calls GPT-4o for structured improvements.
- Applies persona/goals/strategy deltas when present.
- Logs the cycle to `memory_logs` → `event_type = train_loop`.

Feed feedback with `await agent.log_feedback("Lead preferred SMS follow-up", source="sales_call")`
throughout the day to keep the loop meaningful.

---
## 5. Production Calling Checklist

1. **Environment**: Ensure `.env` (or deployment secrets) contains:
   - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`
   - `ELEVENLABS_API_KEY`, `ELEVENLABS_DEFAULT_VOICE_ID`
   - `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-4o`
   - `TWILIO_PUBLIC_BASE_URL` or explicit `TWILIO_CALL_WEBHOOK_URL`
   - `ASR_BACKEND` (`faster-whisper`|`whispercpp`|`stub`), `ASR_MODEL_PATH`, `ASR_MIN_RMS`, `ASR_MIN_UTTERANCE_MS`, `ASR_MAX_SILENCE_MS`
   - `TRANSLATION_BACKEND` (`passthrough`|`nllb`|`bhashini`), `NLLB_MODEL`, `BHASHINI_API_KEY`
2. **Start Call**: `POST /api/v1/agent/sales/start_call` with `lead_phone`, `lead_name`.
   The handler spins up a `SalesAgent`, calls Twilio, orchestrates 4-stage dialogue, and
   uploads ElevenLabs audio snippets to your configured `AUDIO_UPLOAD_URL`.
3. **Media Stream**: Twilio hits `/api/v1/integrations/telephony/twilio/voice` → websocket
   stream. Extend `media` event handling to forward real-time audio/ASR payloads into
   `agent.ingest_webhook_payload` for continuous dialogue.
4. **Verification**: Check Mongo `memory_logs` and Excel exports in `logs/calls/` for
   post-call audits.
5. **Fallback**: If ElevenLabs auth fails, the agent logs the error and continues the call
   with textual responses. Monitor logs for `Voice synthesis failed` messages.

---
## 6. Testing Workflow

1. **Dry Run**
   ```bash
   uvicorn main:app --reload
   http POST :8000/api/v1/agent/sales/ingest_strategy "@strategy.json"
   http POST :8000/api/v1/agent/sales/start_call lead_phone=+15555551234 lead_name=Arjun
   ```
2. **Webhook Simulation**
   ```bash
   http POST :8000/api/v1/hooks/twilio \
        adapter_key=twilio_stream \
        transcript="Hey, what's pricing?" \
        intent:="{\"delegate_to\": \"fallback_bot\"}"
   ```
3. **Train Loop**
   ```python
   await agent.log_feedback("Lead loved the rapport opener", source="qa_team", importance=2)
   await agent.train_loop()
   ```
4. **Remote Prod Check**
   ```bash
   docker compose -f docker/docker-compose.prod.yml exec scriza-api \
       python -m scripts.create_admin  # ensure env + dependencies are intact
   ```

---
## Pending Integration & Gaps
- Incoming-call handling still returns a mock payload and should be wired to Twilio status callbacks with CRM logging (`apps/agents/sales/call_handler.py:188`).
- The first-turn greeting is hard-coded and marked TODO for persona-aware or multilingual templates (`apps/integrations/telephony/twilio.py:818`).
- Environment toggles for call simulation and downstream CRMs are declared but not yet plumbed into runtime services (`core/config.py:57`, `core/config.py:78`).
- Twilio call creation currently requires `API_BASE_URL`; add fallbacks to `TWILIO_CALL_WEBHOOK_URL` or explicit overrides for private deployments (`apps/agents/sales/services.py:434`).
- `start_call` responds before the WebSocket loop backfills Mongo, so clients need polling or streaming guidance to show live turns (`apps/agents/sales/call_handler.py:118`).

---
## 7. Quick Reference

| Feature | Entry Point |
|---------|-------------|
| Persona/strategy training | `SalesAgent.train()` via `/ingest_strategy` |
| Streaming GPT-4o | `generate_response(..., stream=True, on_token=callback)` |
| ElevenLabs voice sync | `speak_response()` & Twilio stream helpers |
| Webhook ingestion | `BaseAgent.ingest_webhook_payload()` |
| Delegation | `agent.delegate("subagent_name", payload)` |
| Daily evolution | `BaseAgent.train_loop()` |
| Memory trail | Mongo collection `memory_logs` |
| Memory summariser | `BaseAgent.summarize_memory()` |
| Modular ASR | `StreamingTranscriber` via `apps/integrations/telephony/twilio.py` |
| Tool registry | `BaseAgent.tool_registry` |
| Ops dashboard | `/ops` (`apps/ops/router.py`) |

Scrappy Singh is now ready to act as an autonomous, production-ready sales rep across
voice and chat channels. Wire in real call transcripts, schedule the train loop, and let
the AI evolve continuously.
