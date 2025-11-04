from __future__ import annotations

import asyncio
import audioop
import logging
from dataclasses import dataclass, field
import numpy as np  # type: ignore
from typing import AsyncIterator, Callable, List, Optional, Tuple
import io
import wave

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SOURCE_SAMPLE_RATE = 8000


@dataclass
class TranscriptionChunk:
    """Container for a completed transcription segment."""

    text: str
    language: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None


class TranscriptionBackend:
    """Abstract interface for pluggable transcription backends."""

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionChunk:
        raise NotImplementedError


class FasterWhisperBackend(TranscriptionBackend):
    """faster-whisper wrapper. Uses CPU by default for portability."""

    def __init__(self, model_size: str = "base", device: str = "cpu") -> None:
        self.model_size = model_size
        self.device = device
        self._model = None
        self._ensure_model()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning("faster-whisper not installed: %s", exc)
            self._model = None
            return
        try:
            self._model = WhisperModel(self.model_size, device=self.device)
        except Exception as exc:  # pragma: no cover - defensive load
            logger.error("Failed to initialise faster-whisper model: %s", exc)
            self._model = None

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionChunk:
        if not self._model:
            raise RuntimeError("faster-whisper backend unavailable")

        def _run() -> Tuple[str, Optional[float]]:
            # Convert raw PCM16 (little-endian) to float32 numpy array in [-1.0, 1.0]
            try:
                audio_np = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                # As a defensive fallback, pass through and let backend handle
                audio_np = pcm16  # type: ignore[assignment]

            segments, info = self._model.transcribe(  # type: ignore[attr-defined]
                audio=audio_np,
                language=language,
                beam_size=1,
                temperature=0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
            )
            text_parts: List[str] = []
            confidence = None
            for segment in segments:
                text_parts.append(getattr(segment, "text", ""))
                # Not all builds expose avg_log_prob; ignore if absent
                try:
                    confidence = getattr(segment, "avg_log_prob", None)
                except Exception:
                    confidence = None
            return " ".join(tp for tp in text_parts if tp).strip(), confidence

        text, confidence = await asyncio.to_thread(_run)
        return TranscriptionChunk(text=text, language=language, confidence=confidence)


class WhisperCppBackend(TranscriptionBackend):
    """whisper.cpp python bindings wrapper."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self._ctx = None
        self._ensure_context()

    def _ensure_context(self) -> None:
        if self._ctx is not None:
            return
        try:
            from whispercpp import Whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning("whispercpp not installed: %s", exc)
            self._ctx = None
            return
        path = self.model_path
        if not path:
            path = "models/ggml-base.en.bin"
        try:
            self._ctx = Whisper(path)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load whisper.cpp model: %s", exc)
            self._ctx = None

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionChunk:
        if not self._ctx:
            raise RuntimeError("whisper.cpp backend unavailable")

        async def _run() -> str:
            return self._ctx.transcribe(  # type: ignore[attr-defined]
                pcm16,
                sampling_rate=sample_rate,
                language=language or "en",
            )

        text = await asyncio.to_thread(_run)
        return TranscriptionChunk(text=text.strip(), language=language)


class StubBackend(TranscriptionBackend):
    """Fallback backend when no native inference runtime is available."""

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionChunk:
        # Last resort: return empty transcript to keep pipeline alive.
        logger.debug("Stub backend invoked for transcription; returning empty text.")
        return TranscriptionChunk(text="", language=language)


class OpenAIWhisperBackend(TranscriptionBackend):
    """OpenAI Whisper/Transcribe API backend.

    Sends per‑utterance PCM to OpenAI's `/v1/audio/transcriptions` endpoint.
    """

    def __init__(self, model_name: Optional[str] = None, timeout_ms: Optional[int] = None) -> None:
        from core.config import settings
        from openai import OpenAI  # lazy import

        self.model_name = model_name or getattr(settings, "ASR_OPENAI_MODEL", "whisper-large-v3")
        self.timeout_s = float((timeout_ms or getattr(settings, "ASR_OPENAI_TIMEOUT_MS", 15000)) / 1000.0)
        # Create client once; the SDK manages sessions internally
        self._client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", ""))

    def _pcm16_to_wav_bytes(self, pcm16: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16)
        return buf.getvalue()

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionChunk:
        wav_bytes = self._pcm16_to_wav_bytes(pcm16, sample_rate)

        def _run() -> str:
            # The Python SDK accepts (filename, bytes, mime) tuple
            file_tuple = ("audio.wav", wav_bytes, "audio/wav")
            resp = self._client.audio.transcriptions.create(  # type: ignore[attr-defined]
                model=self.model_name,
                file=file_tuple,
                language=language,
                timeout=self.timeout_s,
            )
            # SDK returns object with `text`
            return getattr(resp, "text", "") or ""

        try:
            text = await asyncio.to_thread(_run)
        except Exception as exc:  # pragma: no cover - resilience
            logger.exception("OpenAI ASR failure: %s", exc)
            return TranscriptionChunk(text="", language=language)
        return TranscriptionChunk(text=text.strip(), language=language)


@dataclass
class TranscriberConfig:
    backend: str = "faster-whisper"
    language: Optional[str] = None
    min_rms: int = 350
    min_utterance_ms: int = 1200
    max_silence_ms: int = 600
    sample_rate: int = DEFAULT_SAMPLE_RATE
    source_sample_rate: int = DEFAULT_SOURCE_SAMPLE_RATE


class StreamingTranscriber:
    """Voice activity detector + transcription orchestrator for μ-law streams."""

    def __init__(self, config: TranscriberConfig) -> None:
        self.config = config
        self._buffer = bytearray()
        self._trailing_silence_ms = 0
        self._in_speech = False
        self._backend = self._select_backend(config.backend)
        self._lock = asyncio.Lock()

    def _select_backend(self, backend: str) -> TranscriptionBackend:
        lookup = {
            "faster-whisper": FasterWhisperBackend,
            "whispercpp": WhisperCppBackend,
            "openai-whisper": OpenAIWhisperBackend,
            "stub": StubBackend,
        }
        factory = lookup.get(backend.lower())
        if not factory:
            logger.warning("Unknown ASR backend '%s'; defaulting to stub.", backend)
            factory = StubBackend
        try:
            return factory()
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to instantiate ASR backend %s: %s", backend, exc)
            return StubBackend()

    async def add_chunk(self, mulaw_chunk: bytes) -> Optional[TranscriptionChunk]:
        """Feed a raw μ-law chunk. Returns transcription when utterance closes."""
        pcm8k = audioop.ulaw2lin(mulaw_chunk, 2)
        energy = audioop.rms(pcm8k, 2)
        frame_ms = int(len(mulaw_chunk) / (self.config.source_sample_rate / 1000))

        if energy >= self.config.min_rms:
            self._buffer.extend(pcm8k)
            self._trailing_silence_ms = 0
            self._in_speech = True
            return None

        if self._in_speech:
            self._trailing_silence_ms += frame_ms
            if self._trailing_silence_ms <= self.config.max_silence_ms:
                self._buffer.extend(pcm8k)
                return None

            min_pcm_len = int(
                self.config.min_utterance_ms
                * self.config.source_sample_rate
                * 2
                / 1000
            )
            if len(self._buffer) >= min_pcm_len:
                payload = bytes(self._buffer)
                awaitable = self._transcribe(payload)
                self._reset_state()
                return await awaitable

            self._reset_state()

        return None

    async def _transcribe(self, pcm16: bytes) -> TranscriptionChunk:
        async with self._lock:
            try:
                if self.config.source_sample_rate != self.config.sample_rate:
                    pcm16, _ = audioop.ratecv(
                        pcm16,
                        2,
                        1,
                        self.config.source_sample_rate,
                        self.config.sample_rate,
                        None,
                    )
                result = await self._backend.transcribe(
                    pcm16=pcm16,
                    sample_rate=self.config.sample_rate,
                    language=self.config.language,
                )
            except Exception as exc:  # pragma: no cover - resilience
                logger.exception("ASR backend failure: %s", exc)
                return TranscriptionChunk(text="", language=self.config.language)
        return result

    def _reset_state(self) -> None:
        self._buffer.clear()
        self._trailing_silence_ms = 0
        self._in_speech = False

    def reset(self) -> None:
        self._reset_state()

    async def transcribe_chunk(self, mulaw_bytes: bytes) -> TranscriptionChunk:
        """Utility for batch transcription of a pre-segmented μ-law buffer."""
        pcm8k = audioop.ulaw2lin(mulaw_bytes, 2)
        if self.config.source_sample_rate != self.config.sample_rate:
            pcm16, _ = audioop.ratecv(
                pcm8k,
                2,
                1,
                self.config.source_sample_rate,
                self.config.sample_rate,
                None,
            )
        else:
            pcm16 = pcm8k
        return await self._transcribe(pcm16)
