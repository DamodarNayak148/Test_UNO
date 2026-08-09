"""
voice_result.py — Structured Speech-to-Text Voice Perception Result for WALLE.

Produced by VoiceEngine and consumed by the UI and future AI Brain.
"""

from dataclasses import dataclass, field
import time


@dataclass
class VoiceResult:
    """
    Structured snapshot of a voice transcription event.
    """
    text: str = ""
    confidence: float = 0.0       # 0.0 - 1.0 (1.0 = high confidence / valid transcript)
    is_final: bool = True
    is_listening: bool = False
    is_transcribing: bool = False
    timestamp: float = field(default_factory=time.time)
    error: str = ""               # Error message if microphone / STT fails

    @staticmethod
    def listening() -> "VoiceResult":
        return VoiceResult(is_listening=True, is_transcribing=False, text="Listening...")

    @staticmethod
    def transcribing() -> "VoiceResult":
        return VoiceResult(is_listening=False, is_transcribing=True, text="Transcribing...")

    @staticmethod
    def empty(error: str = "") -> "VoiceResult":
        return VoiceResult(is_listening=False, is_transcribing=False, error=error)
