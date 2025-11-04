from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import uuid
from abc import ABC
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Type

from apps.agents.llm_client import complete_chat, stream_chat_completion
from apps.agents.llm_stream import LLMStreamState
from apps.agents.subagent_manager import SubagentManager
from apps.agents.tools import ToolRegistry, bootstrap_default_tools
from apps.agents.translation import TranslationMiddleware, build_translation_middleware
from core.database import get_collection
from core.config import settings
from shared.constants import COLLECTION_CALLS, COLLECTION_MEMORY_LOGS

TokenCallback = Optional[Callable[[str], Awaitable[None]]]


class BaseAgent(ABC):
    """Root Scrappy Singh agent base with self-training, delegation, and memory."""

    def __init__(
        self,
        agent_id: str,
        user_id: str,
        name: str,
        persona: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.user_id = user_id
        self.name = name
        self.persona: Dict[str, Any] = {}
        self.voice_id: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.strategy: Optional[Dict[str, Any]] = None
        self.strategy_context: Optional[str] = None
        self.short_term_context: List[Dict[str, Any]] = []
        self.goals: List[str] = []
        self.active_subagents: Dict[str, "BaseAgent"] = {}
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.training_history: List[Dict[str, Any]] = []
        self.webhook_adapters: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        self.is_trained: bool = False
        self.tool_registry = ToolRegistry()
        bootstrap_default_tools(self.tool_registry)
        self.subagent_manager = SubagentManager()
        self.translation: TranslationMiddleware = build_translation_middleware(
            settings.TRANSLATION_BACKEND,
            model_name=settings.NLLB_MODEL,
            api_key=settings.BHASHINI_API_KEY,
        )
        self._active_stream_state: Optional[LLMStreamState] = None
        self._active_stream_task: Optional[asyncio.Task] = None
        if persona:
            self.configure_persona(persona)

    # ------------------------------------------------------------------
    # Training / setup
    # ------------------------------------------------------------------
    async def train(self, strategy_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Boot the agent with persona/strategy context and dynamically spawn subagents."""
        if not strategy_payload:
            raise ValueError("Strategy payload is required for training")

        persona = strategy_payload.get("persona") if isinstance(strategy_payload, dict) else None
        if isinstance(persona, dict):
            self.configure_persona({**self.persona, **persona})

        goals = strategy_payload.get("goals") or []
        if isinstance(goals, Sequence) and not isinstance(goals, (str, bytes)):
            self.set_goals(list(goals))

        self.attach_strategy(strategy_payload)
        await self._bootstrap_subagents_from_payload(strategy_payload)
        await self._log_agent_event(
            event_type="training_completed",
            data={
                "strategy_reference": strategy_payload.get("strategy_id") or strategy_payload.get("title"),
                "goals": self.goals,
                "persona": self.persona,
            },
        )

        self.training_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "strategy_snapshot": strategy_payload,
            }
        )
        self.is_trained = True
        return {
            "status": "trained",
            "agent_id": self.agent_id,
            "persona": self.persona,
            "goals": self.goals,
            "subagents": list(self.active_subagents.keys()),
        }

    async def train_loop(self) -> Dict[str, Any]:
        """Digest accumulated feedback and propose strategy/persona refinements."""
        if not self.feedback_buffer:
            return {"status": "noop", "reason": "no_feedback_available"}

        feedback_lines = [
            f"[{item['timestamp']}] ({item['source']}) importance={item['importance']}: {item['feedback']}"
            for item in self.feedback_buffer
        ]
        feedback_digest = "\n".join(feedback_lines)
        strategy_snapshot = json.dumps(self.strategy or {}, default=str)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Scrappy Singh's self-improvement module."
                    " Produce JSON with optional 'persona', 'goals', and 'strategy_updates' keys"
                    " derived from the provided feedback."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current strategy: {strategy_snapshot}\n"
                    f"Recent feedback:\n{feedback_digest}"
                ),
            },
        ]

        improvement_text = await self._call_llm(messages, streaming=False)

        try:
            improvement_payload = json.loads(improvement_text)
        except json.JSONDecodeError:
            improvement_payload = {"summary": improvement_text}

        persona_update = improvement_payload.get("persona") if isinstance(improvement_payload, dict) else None
        if isinstance(persona_update, dict):
            self.configure_persona({**self.persona, **persona_update})

        goals_update = improvement_payload.get("goals") if isinstance(improvement_payload, dict) else None
        if isinstance(goals_update, Sequence) and not isinstance(goals_update, (str, bytes)):
            self.set_goals(list(goals_update))

        strategy_updates = improvement_payload.get("strategy_updates") if isinstance(improvement_payload, dict) else None
        if isinstance(strategy_updates, dict):
            merged_strategy = {**(self.strategy or {}), **strategy_updates}
            self.attach_strategy(merged_strategy)

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "feedback_batch": feedback_lines,
            "improvement": improvement_payload,
        }
        self.training_history.append(record)
        await self._log_agent_event("train_loop", record)

        self.feedback_buffer = []
        return {"status": "updated", "improvement": improvement_payload}

    def configure_persona(self, persona: Dict[str, Any]) -> None:
        self.persona = persona or {}
        default_locale = getattr(settings, "DEFAULT_PERSONA_LOCALE", "en")
        self.persona.setdefault("locale", persona.get("locale") or default_locale)
        if persona.get("tone_override"):
            self.persona["tone_override"] = persona["tone_override"]
        self.voice_id = self.persona.get("voice_id")
        self.updated_at = datetime.utcnow()

    def set_goals(self, goals: Optional[List[str]]) -> None:
        self.goals = goals or []
        self.updated_at = datetime.utcnow()

    def attach_strategy(self, strategy: Dict[str, Any]) -> None:
        self.strategy = strategy
        self.strategy_context = self._summarize_strategy(strategy)
        self.updated_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------
    async def generate_response(self, user_input: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate a GPT reply with streaming + translation + tool integration."""
        stream = bool(kwargs.get("stream", False))
        on_token: TokenCallback = kwargs.get("on_token")
        metadata: Dict[str, Any] = dict(kwargs.get("metadata") or {})
        # Allow the caller to select a specific model
        override_model = kwargs.get("model")
        persona_locale = self.persona.get("locale")

        translated_input = await self.translation.ensure_english(user_input, persona_locale)
        if translated_input.text != user_input:
            metadata["translated_user_input"] = translated_input.text
            metadata["user_locale"] = persona_locale
        messages = await self._build_messages(
            user_input=translated_input.text,
            stage=kwargs.get("stage"),
            lead_name=kwargs.get("lead_name"),
            strategy_context=kwargs.get("strategy_context"),
            metadata=metadata or None,
        )
        tools = self.tool_registry.tool_specs()
        tool_calls: List[Dict[str, Any]] = []

        if stream:
            stream_state = await stream_chat_completion(
                messages,
                model=override_model,
                persona=self.persona,
                goals=self.goals,
                tools=tools if tools else None,
            )
            self._active_stream_state = stream_state

            async def _forward_tokens() -> None:
                while True:
                    token = await stream_state.queue.get()
                    if token is None:
                        break
                    if on_token:
                        try:
                            await on_token(token)
                        except Exception:  # pragma: no cover - callback safety
                            continue

            if on_token:
                self._active_stream_task = asyncio.create_task(_forward_tokens())
            await stream_state.finished_event.wait()
            if self._active_stream_task:
                with contextlib.suppress(asyncio.CancelledError):
                    await self._active_stream_task
            if stream_state.error:
                raise RuntimeError(f"Stream failed: {stream_state.error}")
            assistant_internal = stream_state.text()
            tool_calls = stream_state.tool_calls or []
            metadata["token_queue"] = stream_state.queue
            self._active_stream_state = None
            self._active_stream_task = None
        else:
            response = await complete_chat(
                messages,
                model=override_model,
                persona=self.persona,
                goals=self.goals,
                tools=tools if tools else None,
            )
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            assistant_internal = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            usage = response.get("usage")
            if usage:
                metadata["usage"] = usage

        tool_results: List[Dict[str, Any]] = []
        if tool_calls:
            for call in tool_calls:
                fn = (call.get("function") or {}).get("name")
                arguments = (call.get("function") or {}).get("arguments") or "{}"
                if not fn:
                    continue
                try:
                    params = json.loads(arguments)
                except json.JSONDecodeError:
                    params = {}
                try:
                    result = await self.tool_registry.invoke(fn, **params)
                except Exception as exc:  # pragma: no cover - tooling resilience
                    result = {"error": str(exc)}
                tool_results.append({"name": fn, "result": result})

            try:
                follow_up_messages: List[Dict[str, Any]] = list(messages)
                follow_up_messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                for call, result in zip(tool_calls, tool_results):
                    follow_up_messages.append(
                        {
                            "role": "tool",
                            "name": call.get("function", {}).get("name", "tool"),
                            "tool_call_id": call.get("id") or call.get("function", {}).get("name"),
                            "content": json.dumps(result["result"]),
                        }
                    )
                follow_up_response = await complete_chat(
                    follow_up_messages,
                    persona=self.persona,
                    goals=self.goals,
                )
                choice = (follow_up_response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                follow_up_text = message.get("content") or ""
                if follow_up_text:
                    assistant_internal = follow_up_text
            except Exception as exc:  # pragma: no cover - defensive fallback
                await self._log_agent_event("tool_followup_failed", {"error": str(exc)})

        translated_output = await self.translation.to_persona_lang(assistant_internal, persona_locale)
        assistant_display = translated_output.text

        self._append_context("user", translated_input.text)
        self._append_context("assistant", assistant_internal)
        self.updated_at = datetime.utcnow()

        metadata.update(
            {
                "messages": messages,
                "streamed": stream,
                "persona": self.persona,
                "goals": self.goals,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            }
        )
        if translated_output.text != assistant_internal:
            metadata["translated_output"] = assistant_display
            metadata["assistant_locale"] = persona_locale
        return {"text": assistant_display, "metadata": metadata}

    async def speak_response(self, text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
        """Render the agent's reply into ElevenLabs audio, preserving persona voice preferences."""

        from apps.agents.sales import services

        voice_choice = voice_id or self.voice_id
        tone_hint = self.persona.get("tone_override") or self.persona.get("tone")
        payload = await services.synthesise_elevenlabs_voice(
            text=text,
            voice_id=voice_choice,
            tone_hint=tone_hint,
            locale=self.persona.get("locale"),
        )
        return payload

    async def cancel_active_stream(self, reason: Optional[str] = None) -> None:
        if self._active_stream_state:
            await self._active_stream_state.cancel(reason)
            self._active_stream_state = None
        if self._active_stream_task:
            self._active_stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._active_stream_task
            self._active_stream_task = None

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        *,
        description: str,
        schema: Dict[str, Any],
    ) -> None:
        self.tool_registry.register(name, handler, description=description, schema=schema)

    def unregister_tool(self, name: str) -> None:
        self.tool_registry.unregister(name)

    async def log_turn(
        self,
        agent_text: str,
        user_text: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a single conversational exchange into Mongo memory logs."""
        memory_collection = get_collection(COLLECTION_MEMORY_LOGS)
        entry = {
            "memory_id": str(uuid.uuid4()),
            "conversation_id": self.conversation_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "agent_text": agent_text,
            "user_text": user_text,
            "extra": extra or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        await memory_collection.insert_one(entry)
        return entry["memory_id"]

    async def log_feedback(
        self,
        feedback: str,
        source: str = "system",
        importance: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Buffer feedback for future self-training cycles and log the event."""
        entry = {
            "feedback": feedback,
            "source": source,
            "importance": importance,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.feedback_buffer.append(entry)
        if len(self.feedback_buffer) > 100:
            self.feedback_buffer = self.feedback_buffer[-100:]
        await self._log_agent_event("feedback", entry)

    async def ingest_webhook_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply realtime webhook payloads, optionally triggering responses or delegation."""
        if not isinstance(payload, dict):
            raise ValueError("Webhook payload must be a dict")

        updates: Dict[str, Any] = {}
        persona_update = payload.get("persona")
        if isinstance(persona_update, dict):
            self.configure_persona({**self.persona, **persona_update})
            updates["persona"] = self.persona

        goals_update = payload.get("goals")
        if isinstance(goals_update, Sequence) and not isinstance(goals_update, (str, bytes)):
            self.set_goals(list(goals_update))
            updates["goals"] = self.goals

        tone_override = payload.get("tone_override")
        if isinstance(tone_override, str):
            self.persona["tone_override"] = tone_override
            updates["tone_override"] = tone_override

        strategy_update = payload.get("strategy")
        if isinstance(strategy_update, dict):
            base_strategy = self.strategy or {}
            merged = {**base_strategy, **strategy_update}
            self.attach_strategy(merged)
            updates["strategy"] = merged

        adapter_key = payload.get("adapter_key")
        if adapter_key and adapter_key in self.webhook_adapters:
            updates["adapter_response"] = await self.webhook_adapters[adapter_key](payload)

        agent_response: Optional[str] = None
        if payload.get("auto_reply", True):
            user_utterance = payload.get("transcript") or payload.get("text") or payload.get("message")
            if user_utterance:
                response = await self.generate_response(
                    user_utterance,
                    metadata={"source": "webhook", "adapter": adapter_key},
                    stream=payload.get("stream", False),
                )
                agent_response = response["text"]
                updates["agent_response"] = agent_response
                await self.log_turn(
                    agent_text=agent_response,
                    user_text=user_utterance,
                    extra={"source": "webhook", "adapter": adapter_key},
                )

        intent = payload.get("intent")
        if isinstance(intent, dict):
            delegate_target = intent.get("delegate_to")
            if delegate_target:
                updates["delegation"] = await self.delegate(delegate_target, intent)

        spawn_spec = payload.get("spawn_subagent")
        if isinstance(spawn_spec, dict):
            try:
                class_path = spawn_spec["class"]
                name = spawn_spec.get("name", f"dynamic-{len(self.active_subagents)+1}")
                agent_cls = self._import_agent_class(class_path)
                subagent = agent_cls(**spawn_spec.get("init", {}))
                if isinstance(subagent, BaseAgent):
                    ttl_seconds = int(spawn_spec.get("ttl_seconds", 900))
                    self.register_subagent(name, subagent, ttl_seconds=ttl_seconds)
                    updates["spawned_subagent"] = name
            except Exception as exc:  # pragma: no cover - defensive
                updates["spawn_error"] = str(exc)

        await self._log_agent_event("webhook_ingest", {"payload": payload, "updates": updates})
        return updates

    async def clear_memory(self, memory_id: Optional[str] = None) -> bool:
        """Delete either a single memory entry or the full agent history."""
        collection = get_collection(COLLECTION_MEMORY_LOGS)
        if memory_id:
            result = await collection.delete_one({"memory_id": memory_id, "agent_id": self.agent_id})
            return bool(result.deleted_count)
        await collection.delete_many({"agent_id": self.agent_id})
        return True

    async def get_memory(self, memory_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch ordered memory records for analytics or replay."""
        collection = get_collection(COLLECTION_MEMORY_LOGS)
        if memory_id:
            doc = await collection.find_one({"memory_id": memory_id, "agent_id": self.agent_id})
            return [doc] if doc else []
        cursor = collection.find({"agent_id": self.agent_id, "conversation_id": self.conversation_id}).sort("created_at", 1)
        return await cursor.to_list(length=200)

    async def summarize_memory(self) -> Dict[str, Any]:
        """Summarise recent calls to surface objections, intent drift, and sentiment."""
        calls_collection = get_collection(COLLECTION_CALLS)
        cursor = (
            calls_collection.find({"agent_id": self.agent_id})
            .sort("created_at", -1)
            .limit(5)
        )
        recent_calls = await cursor.to_list(length=5)
        if not recent_calls:
            return {"status": "noop", "reason": "no_calls"}

        notes: List[str] = []
        for call in recent_calls:
            call_id = call.get("call_id")
            sentiment = call.get("sentiment")
            key_phrases = call.get("key_phrases") or []
            failure_reason = call.get("failure_reason")
            notes.append(
                json.dumps(
                    {
                        "call_id": call_id,
                        "sentiment": sentiment,
                        "key_phrases": key_phrases,
                        "failure_reason": failure_reason,
                    }
                )
            )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are Scrappy Singh's memory summariser. Analyse recent calls and identify recurring objections, "
                    "intent drift, and sentiment trends. Respond with JSON containing keys "
                    "'recurring_objections', 'intent_drift', 'sentiment_patterns', and 'recommendations'."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(notes),
            },
        ]

        summary_text = await self._call_llm(prompt, streaming=False)
        try:
            summary_payload = json.loads(summary_text)
        except json.JSONDecodeError:
            summary_payload = {"raw": summary_text}

        await self._log_agent_event(
            "summary",
            {
                "calls_considered": [call.get("call_id") for call in recent_calls],
                "summary": summary_payload,
            },
        )
        return {"status": "summarised", "summary": summary_payload}

    # ------------------------------------------------------------------
    # Delegation hooks (multi-agent future)
    # ------------------------------------------------------------------
    def register_subagent(self, name: str, agent: "BaseAgent", *, ttl_seconds: int = 900) -> None:
        """Register a fully initialised subagent for delegated tasks."""
        self.active_subagents[name] = agent
        self.subagent_manager.spawn(name, agent, ttl_seconds=ttl_seconds)

    async def delegate(self, name: str, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Invoke a named subagent if available, returning its task result."""
        agent = self.subagent_manager.route(name) or self.active_subagents.get(name)
        if not agent:
            return None
        result = agent.handle_task(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def handle_task(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:  # pragma: no cover
        """Placeholder task handler for child agents to override."""
        return None

    def spawn_subagent(
        self,
        name: str,
        agent_cls: Type["BaseAgent"],
        *args: Any,
        ttl_seconds: int = 900,
        **kwargs: Any,
    ) -> "BaseAgent":
        """Instantiate and register a subagent from a class reference."""
        if not inspect.isclass(agent_cls) or not issubclass(agent_cls, BaseAgent):
            raise TypeError("agent_cls must be a subclass of BaseAgent")
        subagent = agent_cls(*args, **kwargs)
        self.register_subagent(name, subagent, ttl_seconds=ttl_seconds)
        return subagent

    def register_webhook_adapter(
        self,
        key: str,
        adapter: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> None:
        """Attach an async adapter used to pre-process webhook payloads."""
        self.webhook_adapters[key] = adapter

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        streaming: bool = False,
        on_token: TokenCallback = None,
    ) -> str:
        """Proxy GPT chat calls with persona injection and streaming control."""
        tools = self.tool_registry.tool_specs()
        if streaming:
            stream_state = await stream_chat_completion(
                messages,
                persona=self.persona,
                goals=self.goals,
                tools=tools if tools else None,
            )

            async def _drain() -> None:
                while True:
                    token = await stream_state.queue.get()
                    if token is None:
                        break
                    if on_token:
                        try:
                            await on_token(token)
                        except Exception:  # pragma: no cover
                            continue

            if on_token:
                await _drain()
            await stream_state.finished_event.wait()
            if stream_state.error:
                raise RuntimeError(stream_state.error)
            return stream_state.text()

        response = await complete_chat(
            messages,
            persona=self.persona,
            goals=self.goals,
            tools=tools if tools else None,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return message.get("content") or ""

    def _append_context(self, role: str, content: str) -> None:
        """Store an exchange in rolling short-term context."""
        self.short_term_context.append({"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()})
        if len(self.short_term_context) > 40:
            self.short_term_context = self.short_term_context[-40:]

    async def _build_messages(
        self,
        user_input: str,
        stage: Optional[str] = None,
        lead_name: Optional[str] = None,
        strategy_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """Compose the message stack combining persona prompt and rolling context."""
        system_prompt = self._build_system_prompt(
            stage=stage,
            lead_name=lead_name,
            strategy_context=strategy_context,
            metadata=metadata,
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in self.short_term_context[-20:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_system_prompt(
        self,
        stage: Optional[str] = None,
        lead_name: Optional[str] = None,
        strategy_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a rich system prompt encoding persona, goals, stage, and context."""
        persona_parts = [f"Agent Name: {self.persona.get('name') or self.name}"]
        tone_override = self.persona.get("tone_override")
        tone = self.persona.get("tone")
        if tone_override:
            persona_parts.append(f"Tone Override: {tone_override}")
        elif tone:
            persona_parts.append(f"Tone: {tone}")
        description = self.persona.get("description")
        if description:
            persona_parts.append(description)
        if self.goals:
            persona_parts.append("Goals: " + "; ".join(self.goals))
        context = strategy_context or self.strategy_context
        if context:
            persona_parts.append("Strategy Context: " + context)
        if stage:
            persona_parts.append(f"Current Stage: {stage}")
        if lead_name:
            persona_parts.append(f"Speaking with: {lead_name}")
        if metadata:
            persona_parts.append("Metadata: " + "; ".join(f"{k}: {v}" for k, v in metadata.items()))
        persona_parts.append(
            "Always act with empathy, emotional intelligence, ethical persuasion, and adaptive real-time learning."
        )
        return "\n".join(persona_parts)

    def _summarize_strategy(self, strategy: Optional[Dict[str, Any]]) -> Optional[str]:
        from shared.strategy_utils import summarize_strategy_payload
        return summarize_strategy_payload(strategy)

    async def _bootstrap_subagents_from_payload(self, strategy_payload: Dict[str, Any]) -> None:
        """Automatically instantiate subagents declared in the strategy."""
        specs = strategy_payload.get("subagents") if isinstance(strategy_payload, dict) else None
        if not specs:
            return
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            class_path = spec.get("class")
            init_kwargs = spec.get("init", {})
            if not (name and class_path):
                continue
            try:
                agent_cls = self._import_agent_class(class_path)
            except (ImportError, AttributeError, ValueError) as exc:
                await self._log_agent_event(
                    "subagent_bootstrap_failed",
                    {"spec": spec, "reason": str(exc)},
                )
                continue
            try:
                subagent = agent_cls(**init_kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                await self._log_agent_event(
                    "subagent_bootstrap_failed",
                    {"spec": spec, "reason": str(exc)},
                )
                continue
            if isinstance(subagent, BaseAgent):
                self.register_subagent(name, subagent)
                await self._log_agent_event("subagent_registered", {"name": name})

    def _import_agent_class(self, dotted_path: str) -> Type["BaseAgent"]:
        """Import a BaseAgent subclass by dotted path for dynamic registration."""
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        agent_cls = getattr(module, class_name)
        if not inspect.isclass(agent_cls) or not issubclass(agent_cls, BaseAgent):
            raise TypeError(f"{dotted_path} is not a BaseAgent subclass")
        return agent_cls

    async def _log_agent_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write arbitrary lifecycle events into the memory log collection."""
        collection = get_collection(COLLECTION_MEMORY_LOGS)
        entry = {
            "memory_id": str(uuid.uuid4()),
            "conversation_id": self.conversation_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "event_type": event_type,
            "data": data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        await collection.insert_one(entry)
