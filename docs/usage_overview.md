Usage Metrics and Cost Optimization Overview

Summary of usage-related metrics and model selection across the Scriza AI Voice Platform.

Metrics captured
- calls.tokens_used: Incremented per LLM turn during live calls when usage data is available (total_tokens).
- calls.duration_seconds: Recorded at the end of a Twilio WebSocket call (start→stop event delta).

Model selection
- Critical (in-call, live response): settings.OPENAI_MODEL_CRITICAL (default: gpt-4o).
- Non-critical (summaries, drafts, offline prompts): settings.OPENAI_MODEL_CHEAP (default: gpt-4o-mini).

Caching
- LLM response caching: TTL-based (settings.LLM_CACHE_TTL_SECONDS, default 30s) in apps/agents/llm_client.complete_chat(). Identical prompts within TTL are reused.
- TTS audio caching: Disk cache under settings.TTS_CACHE_DIR (default static/voice_cache). Keyed by text+voice+model+tone+locale in apps/agents/sales/services.synthesise_elevenlabs_voice().

Implementation notes
- Persona injection is centralized in apps/agents/llm_client (services no longer performs pre-injection to avoid duplication).
- Token usage is attached to BaseAgent.generate_response metadata when using non-stream completions and persisted to the calls collection by the Twilio media stream handler.
- Calls collection now initializes with tokens_used=0 and duration_seconds=0.0 in services.save_call_record().

Future opportunities
- Add Redis-backed distributed caching for LLM and TTS to work across multiple workers.
- Capture and aggregate usage by user/agent to enforce credit limits and alert thresholds.
- Add /api/v1/usage/summary endpoints and periodic usage alerts.
- Pre-batch greeting/intro/fallback audio during strategy ingestion and store in voice assets for reuse.
