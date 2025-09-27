# Scrappy Singh Operational Guide

Scrappy Singh is the lead conversational agent for the Scriza platform. The refactored
`BaseAgent` (“the brain”) powers persona-aware conversations, real-time delegation, and
self-improvement loops. This guide explains how to configure, orchestrate, and test the
production stack.

---
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

Scrappy Singh is now ready to act as an autonomous, production-ready sales rep across
voice and chat channels. Wire in real call transcripts, schedule the train loop, and let
the AI evolve continuously.
