"""Standalone Whisper transcription via Groq API for Telegram voice messages."""

import io
import httpx


class VoiceTranscriber:
    """Transcribes Telegram voice messages using Groq Whisper API."""

    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB Groq limit

    def __init__(self, config: dict):
        self.enabled: bool = config.get("whisper_enabled", True)
        self.api_key: str = config.get("whisper_api_key", "")
        self.model: str = config.get("whisper_model", "whisper-large-v3-turbo")
        self.endpoint: str = config.get(
            "whisper_endpoint",
            "https://api.groq.com/openai/v1/audio/transcriptions",
        )

    @property
    def is_available(self) -> bool:
        return self.enabled and bool(self.api_key)

    async def transcribe(self, audio_bytes: bytes, filename: str = "voice.ogg") -> str:
        """Transcribe audio bytes to text via Groq Whisper API.

        Args:
            audio_bytes: Raw audio data (OGG/Opus from Telegram).
            filename: Filename hint for the API.

        Returns:
            Transcribed text.

        Raises:
            ValueError: If file exceeds size limit.
            RuntimeError: If transcription fails.
        """
        if not self.is_available:
            raise RuntimeError("Whisper transcription is not configured (missing API key).")

        if len(audio_bytes) > self.MAX_FILE_SIZE:
            raise ValueError(
                f"Voice message too large ({len(audio_bytes) / 1024 / 1024:.1f} MB). "
                f"Max is {self.MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
            )

        files = {"file": (filename, io.BytesIO(audio_bytes), "audio/ogg")}
        data = {
            "model": self.model,
            "language": "de",
            "response_format": "json",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.endpoint,
                headers=headers,
                files=files,
                data=data,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Whisper API error {response.status_code}: {response.text[:200]}"
            )

        result = response.json()
        text = result.get("text", "").strip()
        if not text:
            raise RuntimeError("Whisper returned empty transcription.")
        return text
