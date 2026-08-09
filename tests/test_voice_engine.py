"""
tests/test_voice_engine.py — Unit tests for WALLE Local Speech-to-Text Voice Subsystem.

Tests cover:
  1. VoiceResult structure and default fields
  2. WALLE Proper Noun Normalization ("hello volley" -> "Hello WALLE")
  3. VoiceEngine initialization with faster-whisper (tiny.en) & Vosk fallback
  4. Push-to-Talk state transitions (is_listening, is_transcribing, get_result)
  5. Empty / short audio recording error handling

Run:
    python -m pytest tests/test_voice_engine.py -v
"""

import os
import time
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.voice.voice_result import VoiceResult
from src.voice.voice_engine import VoiceEngine, normalize_walle_transcript


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestVoiceResultStructure:

    def test_voice_result_default_fields(self):
        vr = VoiceResult()
        assert hasattr(vr, "text")
        assert hasattr(vr, "confidence")
        assert hasattr(vr, "is_final")
        assert hasattr(vr, "is_listening")
        assert hasattr(vr, "is_transcribing")
        assert hasattr(vr, "timestamp")
        assert hasattr(vr, "error")

        assert vr.text == ""
        assert vr.confidence == 0.0
        assert vr.is_final is True
        assert vr.is_listening is False
        assert vr.is_transcribing is False
        assert vr.error == ""

    def test_voice_result_factory_methods(self):
        vr_listen = VoiceResult.listening()
        assert vr_listen.is_listening is True
        assert vr_listen.text == "Listening..."

        vr_trans = VoiceResult.transcribing()
        assert vr_trans.is_transcribing is True
        assert vr_trans.text == "Transcribing..."

        vr_err = VoiceResult.empty(error="Microphone unavailable")
        assert vr_err.error == "Microphone unavailable"
        assert vr_err.is_listening is False


class TestTextNormalization:

    def test_normalize_walle_phoneme_corrections(self):
        assert normalize_walle_transcript("hello volley") == "Hello WALLE"
        assert normalize_walle_transcript("hello wall e") == "Hello WALLE"
        assert normalize_walle_transcript("is wally ready") == "Is WALLE ready"
        assert normalize_walle_transcript("hello wallie") == "Hello WALLE"
        assert normalize_walle_transcript("hello walle") == "Hello WALLE"
        assert normalize_walle_transcript("") == ""


class TestVoiceEngine:

    @pytest.fixture(scope="class")
    def engine(self):
        return VoiceEngine(backend="whisper", model_size="tiny.en")

    def test_voice_engine_initialization(self, engine):
        assert engine is not None
        assert hasattr(engine, "start_recording")
        assert hasattr(engine, "stop_recording_and_transcribe")
        assert hasattr(engine, "is_listening")
        assert hasattr(engine, "is_transcribing")
        assert hasattr(engine, "get_result")

    def test_voice_engine_initial_state(self, engine):
        assert engine.is_listening() is False
        assert engine.is_transcribing() is False
        res = engine.get_result()
        assert res.is_listening is False
        assert res.is_transcribing is False

    def test_empty_audio_worker_handling(self, qapp):
        from src.voice.voice_engine import _SttWorker

        received_res = []

        def _on_finished(res):
            received_res.append(res)

        worker = _SttWorker(
            backend="whisper",
            whisper_model=None,
            vosk_model=None,
            kaldi_rec_cls=None,
            audio_pcm=b"",
            audio_float=np.array([]),
            duration_sec=0.0,
        )
        worker.finished_signal.connect(_on_finished)
        worker.run()

        assert len(received_res) == 1
        res = received_res[0]
        assert isinstance(res, VoiceResult)
        assert res.error == "Recording too short"
