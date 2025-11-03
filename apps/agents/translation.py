from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    text: str
    source_language: str
    target_language: str


class TranslationBackend:
    async def translate(self, text: str, source_language: str, target_language: str) -> TranslationResult:
        raise NotImplementedError


class NLLBBackend(TranslationBackend):
    """Thin wrapper around facebook NLLB pipeline via transformers."""

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M") -> None:
        self.model_name = model_name
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline:
            return
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning("transformers not installed; translation will be a no-op (%s)", exc)
            self._pipeline = None
            return
        try:
            self._pipeline = pipeline("translation", model=self.model_name)
        except Exception as exc:  # pragma: no cover - heavy init
            logger.error("Failed to load NLLB model %s: %s", self.model_name, exc)
            self._pipeline = None

    async def translate(self, text: str, source_language: str, target_language: str) -> TranslationResult:
        self._ensure_pipeline()
        if not self._pipeline:
            return TranslationResult(text=text, source_language=source_language, target_language=target_language)

        async def _run() -> str:
            result = self._pipeline(text, src_lang=source_language, tgt_lang=target_language)  # type: ignore[attr-defined]
            if not result:
                return text
            return result[0]["translation_text"]

        translated = await asyncio.to_thread(_run)
        return TranslationResult(text=translated, source_language=source_language, target_language=target_language)


class BhashiniBackend(TranslationBackend):
    """Adapter for India's Bhashini API (placeholder implementation)."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://bhashini.gov.in") -> None:
        self.api_key = api_key
        self.base_url = base_url

    async def translate(self, text: str, source_language: str, target_language: str) -> TranslationResult:
        # Placeholder: integrate real Bhashini client when credentials are wired.
        logger.debug("Bhashini translation requested %s→%s; returning original text (stub).", source_language, target_language)
        return TranslationResult(text=text, source_language=source_language, target_language=target_language)


class PassthroughBackend(TranslationBackend):
    def __init__(self, **_: Any) -> None:
        pass

    async def translate(self, text: str, source_language: str, target_language: str) -> TranslationResult:
        return TranslationResult(text=text, source_language=source_language, target_language=target_language)


class TranslationMiddleware:
    """Convenience wrapper to bridge persona locale and GPT/TTS languages."""

    def __init__(self, backend: TranslationBackend):
        self.backend = backend

    async def ensure_english(self, text: str, persona_locale: Optional[str]) -> TranslationResult:
        if not text or not persona_locale or persona_locale.lower().startswith("en"):
            return TranslationResult(text=text, source_language=persona_locale or "auto", target_language="en")
        return await self.backend.translate(text, source_language=persona_locale, target_language="en")

    async def to_persona_lang(self, text: str, persona_locale: Optional[str]) -> TranslationResult:
        if not text or not persona_locale or persona_locale.lower().startswith("en"):
            return TranslationResult(text=text, source_language="en", target_language=persona_locale or "en")
        return await self.backend.translate(text, source_language="en", target_language=persona_locale)


def build_translation_middleware(backend_name: str, **kwargs) -> TranslationMiddleware:
    lookup = {
        "nllb": NLLBBackend,
        "bhashini": BhashiniBackend,
        "passthrough": PassthroughBackend,
    }
    backend_cls = lookup.get(backend_name.lower(), PassthroughBackend)
    backend = backend_cls(**kwargs)
    return TranslationMiddleware(backend)
