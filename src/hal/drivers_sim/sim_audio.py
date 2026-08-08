import threading
import queue
import time
from typing import Optional, Callable
import pyttsx3
from src.hal.base_audio import BaseAudio

class SimAudio(BaseAudio):
    """Simulated Audio Driver handling TTS speech synthesis & audio SFX."""

    def __init__(self, voice_rate: int = 170):
        self.voice_rate = voice_rate
        self.speech_queue = queue.Queue()
        self._is_speaking = False
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()
        self._on_speech_state_change: Optional[Callable[[bool, str], None]] = None

    def set_state_change_callback(self, callback: Callable[[bool, str], None]) -> None:
        """Register callback for UI audio state notifications (is_speaking, current_text)."""
        self._on_speech_state_change = callback

    def speak(self, text: str, on_complete: Optional[Callable[[], None]] = None) -> None:
        """Queue text to be spoken via TTS engine."""
        print(f"[SimAudio TTS]: '{text}'")
        self.speech_queue.put((text, on_complete))

    def stop_speaking(self) -> None:
        """Clear queued speech items and stop current speech."""
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()
        self._is_speaking = False
        if self._on_speech_state_change:
            self._on_speech_state_change(False, "")

    def play_sfx(self, sfx_name: str) -> None:
        print(f"[SimAudio SFX]: Playing sound effect '{sfx_name}'")

    def is_speaking(self) -> bool:
        return self._is_speaking

    def _speech_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self.speech_queue.get(timeout=0.2)
                if item is None:
                    continue
                
                text, on_complete = item
                self._is_speaking = True
                if self._on_speech_state_change:
                    self._on_speech_state_change(True, text)

                engine = None
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', self.voice_rate)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[SimAudio Engine Error]: {e}")
                    time.sleep(len(text) * 0.05)
                finally:
                    if engine:
                        try:
                            del engine
                        except Exception:
                            pass

                self._is_speaking = False
                if self._on_speech_state_change:
                    self._on_speech_state_change(False, "")

                if on_complete:
                    try:
                        on_complete()
                    except Exception as e:
                        print(f"[SimAudio Callback Error]: {e}")

                self.speech_queue.task_done()
            except queue.Empty:
                continue
