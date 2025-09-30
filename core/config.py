import os
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Settings:
    # Project settings
    PROJECT_NAME: str = "Scriza AI Platform"
    PROJECT_VERSION: str = "0.1.0"
    PROJECT_DESCRIPTION: str = "Next-gen platform for creating, training, and deploying human-like AI agents"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database settings
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "scriza_db")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # ElevenLabs settings
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_BASE_URL: str = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")
    ELEVENLABS_DEFAULT_VOICE_ID: str = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "")
    ELEVENLABS_MODEL: str = os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1")
    ELEVENLABS_USE_WS_TTS: bool = os.getenv("ELEVENLABS_USE_WS_TTS", "false").lower() == "true"
    ELEVENLABS_WS_URL: str = os.getenv(
        "ELEVENLABS_WS_URL",
        "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input",
    )
    ELEVENLABS_WS_OUTPUT_FORMAT: str = os.getenv("ELEVENLABS_WS_OUTPUT_FORMAT", "ulaw_8000")
    ELEVENLABS_WS_OPTIMIZE_LATENCY: int = int(os.getenv("ELEVENLABS_WS_OPTIMIZE_LATENCY", "3"))
    ELEVENLABS_WS_STABILITY: float = float(os.getenv("ELEVENLABS_WS_STABILITY", "0.4"))
    ELEVENLABS_WS_SIMILARITY: float = float(os.getenv("ELEVENLABS_WS_SIMILARITY", "0.8"))
    
    # Twilio settings
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    TWILIO_CALL_WEBHOOK_URL: str = os.getenv(
        "TWILIO_CALL_WEBHOOK_URL",
        "http://demo.twilio.com/docs/voice.xml",
    )
    TWILIO_PUBLIC_BASE_URL: str = os.getenv("TWILIO_PUBLIC_BASE_URL", "")
    AUDIO_UPLOAD_URL: str = os.getenv("AUDIO_UPLOAD_URL", "")
    TWILIO_STREAM_URL: str = os.getenv("TWILIO_STREAM_URL", "")
    TWILIO_STREAM_TRACK: str = os.getenv("TWILIO_STREAM_TRACK", "")  # e.g. inbound_track, both_tracks
    SIMULATE_CALL_FLOW: bool = os.getenv("SIMULATE_CALL_FLOW", "false").lower() == "true"

    # Public base for building absolute URLs
    API_BASE_URL: str = os.getenv("API_BASE_URL", "")

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Logging settings
    CALL_LOG_DIR: str = os.getenv("CALL_LOG_DIR", "logs/calls")
    # Live call / media stream tuning
    CALL_DEBUG_VERBOSE: bool = os.getenv("CALL_DEBUG_VERBOSE", "false").lower() == "true"
    CALL_FIRST_TURN_GREETING: bool = os.getenv("CALL_FIRST_TURN_GREETING", "true").lower() == "true"
    CALL_VAD_MIN_RMS: int = int(os.getenv("CALL_VAD_MIN_RMS", "300"))
    CALL_SILENCE_THRESHOLD_SEC: float = float(os.getenv("CALL_SILENCE_THRESHOLD_SEC", "0.7"))
    CALL_MIN_BUFFER_MS: int = int(os.getenv("CALL_MIN_BUFFER_MS", "1500"))
    CALL_NUDGE_AFTER_SEC: float = float(os.getenv("CALL_NUDGE_AFTER_SEC", "0"))  # 0 disables
    
    # Zoho CRM settings
    ZOHO_CLIENT_ID: str = os.getenv("ZOHO_CLIENT_ID", "")
    ZOHO_CLIENT_SECRET: str = os.getenv("ZOHO_CLIENT_SECRET", "")
    
    # Telegram settings
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

settings = Settings()
