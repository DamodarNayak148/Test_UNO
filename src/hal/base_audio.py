from abc import ABC, abstractmethod
from typing import Optional, Callable

class BaseAudio(ABC):
    """Abstract interface for Game Master audio output (TTS/SFX) and input (Microphone)."""

    @abstractmethod
    def speak(self, text: str, on_complete: Optional[Callable[[], None]] = None) -> None:
        """Speak the provided text using Text-to-Speech synthesis."""
        pass

    @abstractmethod
    def stop_speaking(self) -> None:
        """Interrupt and stop current speech playback."""
        pass

    @abstractmethod
    def play_sfx(self, sfx_name: str) -> None:
        """Play a sound effect by name (e.g. 'fanfare', 'buzzer', 'drumroll')."""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """Return True if TTS audio is currently playing."""
        pass
