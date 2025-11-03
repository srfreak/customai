# Scrappy Singh Sovereign Kernel

Scrappy Singh powers voice-first, persona-driven sales automation for the Scriza platform. This upgrade elevates the agent from an MVP caller into a modular, sovereign AI kernel with local ASR, streaming LLM control, tone-aware synthesis, and an integrated ops console.

## Highlights
- **Local-first transcription** – `StreamingTranscriber` (`apps/agents/transcriber.py`) wraps faster-whisper / whisper.cpp backends with μ-law VAD buffering so Twilio audio never leaves the box.
- **Streaming GPT orchestration** – `apps/agents/llm_client.py` streams GPT-4o tokens into `asyncio.Queue`s so speech buffers can start speaking before the turn completes and cancel mid-stream on barge-in.
- **Multi-language middleware** – `apps/agents/translation.py` toggles between passthrough, NLLB, or Bhashini translation paths keyed off `persona.locale` so user input/output are auto-localised.
- **Persona emotion tuning** – Payloads and webhooks can set `tone_override` (friendly/assertive/empathetic/humorous) driving both GPT prompts and ElevenLabs stability/similarity settings.
- **Tool + subagent kernel** – `ToolRegistry` exposes async functions (`search_docs`, `fetch_crm_data`, `send_followup_email`, `read_product_faq`) with a TTL-managed `SubagentManager` for sovereign workflows.
- **Ops dashboard** – `/ops` renders a React console (via CDN) with live-call telemetry, fallback/error timelines, persona snapshots, and downloadable Excel exports.

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# FastAPI entrypoint
uvicorn main:app --reload

# Seed a strategy and start a simulated call
http POST :8000/api/v1/agent/sales/ingest_strategy @apps/agents/sales/prompts/strategy.json
http POST :8000/api/v1/agent/sales/start_call lead_phone=+15555551234 lead_name=Arjun
```

Environment variables (`core/config.py`) now include:
- `ASR_BACKEND`, `ASR_MODEL_PATH`, `ASR_MIN_RMS`, `ASR_MIN_UTTERANCE_MS`, `ASR_MAX_SILENCE_MS`
- `TRANSLATION_BACKEND`, `NLLB_MODEL`, `BHASHINI_API_KEY`, `DEFAULT_PERSONA_LOCALE`
- Existing Twilio / ElevenLabs / OpenAI keys remain required.

### Ops Console
- Visit `https://<host>/ops` (admin role only) to monitor live calls, inspect fallback triggers, or grab the latest Excel exports from `logs/calls/`.
- Websocket updates derive from `LiveCallRegistry` (`apps/ops/live_calls.py`) which is hydrated inside the Twilio media stream.

### Web UI (New)
- `GET /login` — user login helper; stores JWT locally and redirects to the unified dashboard.
- `GET /admin/login` — admin login with the same workflow; requires the `admin` role to access elevated modules.
- `GET /dashboard` — role-aware console with panels for strategy ingest, call orchestration, ElevenLabs synthesis, memory management, user directory (admin), and quick access to the Ops console.
- Frontend utilities live in `static/js/app.js` and `static/css/style.css`; everything is served directly by FastAPI (no additional build step required).

## Agent Kernel Internals
- `BaseAgent.generate_response()` performs translation → message build → streaming GPT with optional tool-calls → translation back to persona locale → tone-aware TTS.
- `BaseAgent.cancel_active_stream()` aborts an in-flight stream and is invoked on Twilio barge-in to keep replies tight.
- Daily refinements: `BaseAgent.summarize_memory()` and `BaseAgent.train_loop()` mine `COLLECTION_CALLS` + `memory_logs` for objections, intent drift, and persona deltas.

## Dynamic Fallbacks
- Strategies can define rule-based policies:
  ```json
  "fallback_policies": [
    {"trigger": "confidence_below_0.5", "threshold": 0.45, "action": "prompt_for_clarification"},
    {"trigger": "tts_error", "action": "maintain_voice_stream"}
  ]
  ```
- Policies are evaluated inside `twilio.media_stream_endpoint` and log structured events into both `COLLECTION_CALLS` and `memory_logs` with `event_type="fallback_triggered"`. Voice continuity is preserved with `_stream_silence` even during ElevenLabs outages.

## Testing Checklist
- `uvicorn main:app --reload`
- `http POST :8000/api/v1/agent/sales/ingest_strategy …`
- `http POST :8000/api/v1/agent/sales/start_call …`
- `/ops` dashboard for observability.

## Pending Integrations
- Wire CRM backends (Zoho, HubSpot) into the tool registry for richer context.
- Harden the translation layer with on-device models per locale and caching.
- Extend the ops console with train-loop triggers and full change streams once Mongo change streams are available in deployment.

Welcome to Scrappy Singh v2 — autonomous, sovereign, and production ready.
- Optional extras (only if you want on-device NLLB or faster-whisper):
  ```bash
  pip install 'faster-whisper>=0.10.0' 'transformers>=4.38.0' 'whispercpp>=0.0.15'
  ```
