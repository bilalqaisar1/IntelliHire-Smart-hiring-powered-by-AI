from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Tool:
    name: str
    description: str
    run: callable


def tool_speak():
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        # Lazy import to avoid loading TTS engine unless needed
        from ...voice_interview import speak as _speak
        text = payload.get("text", "")
        _speak(text)
        return {"ok": True}

    return Tool(
        name="speak",
        description="Text-to-speech output using local TTS",
        run=_run,
    )


def tool_record_audio():
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        from ...voice_interview import record_audio
        filename = payload.get("filename", "recordings/answer.wav")
        record_audio(filename)
        return {"filename": filename}

    return Tool(
        name="record_audio",
        description="Record microphone input until silence and save WAV",
        run=_run,
    )


def tool_transcribe_audio():
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        from ...voice_interview import transcribe_audio
        filename = payload.get("filename", "recordings/answer.wav")
        text = transcribe_audio(filename)
        return {"text": text}

    return Tool(
        name="transcribe_audio",
        description="Transcribe WAV to text using Faster-Whisper",
        run=_run,
    )


