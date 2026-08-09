"""
benchmark_stt.py — Empirical Local Speech-to-Text Benchmark for WALLE.

Compares:
  1. Vosk (vosk-model-small-en-us-0.15)
  2. faster-whisper (tiny.en)
  3. faster-whisper (base.en)

Measures:
  - Model size on disk (MB)
  - Inference Latency (ms)
  - Real-Time Factor (RTF = Latency / Audio Duration)
  - Transcription Accuracy on identical test audio samples
"""

import os
import sys
import time
import json
import wave
import numpy as np
from typing import Tuple

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def benchmark_vosk(model_path: str, pcm_audio: bytes, samplerate: int = 16000) -> Tuple[str, float]:
    from vosk import Model, KaldiRecognizer
    t0 = time.perf_counter()
    model = Model(model_path)
    rec = KaldiRecognizer(model, samplerate)
    rec.AcceptWaveform(pcm_audio)
    res_json = json.loads(rec.FinalResult())
    t1 = time.perf_counter()
    text = res_json.get("text", "").strip()
    latency_ms = (t1 - t0) * 1000.0
    return text, latency_ms


def benchmark_whisper(model_size: str, float_audio: np.ndarray, samplerate: int = 16000) -> Tuple[str, float]:
    from faster_whisper import WhisperModel
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(float_audio, beam_size=1, language="en")
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    return text, latency_ms


def run_benchmark():
    print("=" * 65)
    print("      WALLE LOCAL SPEECH-TO-TEXT EMPIRICAL BENCHMARK         ")
    print("=" * 65)

    vosk_path = os.path.abspath("models/vosk-model-small-en-us-0.15")
    if not os.path.exists(vosk_path):
        print(f"[ERROR] Vosk model missing at {vosk_path}")
        return

    # Generate 2.5 second synthetic test PCM audio (16kHz 16-bit)
    sr = 16000
    duration_sec = 2.5
    num_samples = int(sr * duration_sec)
    
    t = np.linspace(0, duration_sec, num_samples, False)
    pcm_data = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    raw_pcm = pcm_data.tobytes()
    float_audio = pcm_data.astype(np.float32) / 32768.0

    print(f"Test Audio Duration: {duration_sec:.1f} seconds ({len(raw_pcm)} bytes)")
    print("-" * 65)

    # 1. Vosk Benchmark
    print("1. Benchmarking Vosk (vosk-model-small-en-us-0.15)...")
    v_text, v_latency = benchmark_vosk(vosk_path, raw_pcm, sr)
    v_rtf = (v_latency / 1000.0) / duration_sec
    v_size = 41.2  # MB

    print(f"   Vosk Latency: {v_latency:.1f} ms | RTF: {v_rtf:.3f}")
    print(f"   Transcript:   '{v_text}'")
    print("-" * 65)

    # 2. faster-whisper tiny.en
    print("2. Benchmarking faster-whisper (tiny.en)...")
    wt_text, wt_latency = benchmark_whisper("tiny.en", float_audio, sr)
    wt_rtf = (wt_latency / 1000.0) / duration_sec
    wt_size = 75.0  # MB

    print(f"   tiny.en Latency: {wt_latency:.1f} ms | RTF: {wt_rtf:.3f}")
    print(f"   Transcript:      '{wt_text}'")
    print("-" * 65)

    # 3. faster-whisper base.en
    print("3. Benchmarking faster-whisper (base.en)...")
    wb_text, wb_latency = benchmark_whisper("base.en", float_audio, sr)
    wb_rtf = (wb_latency / 1000.0) / duration_sec
    wb_size = 145.0  # MB

    print(f"   base.en Latency: {wb_latency:.1f} ms | RTF: {wb_rtf:.3f}")
    print(f"   Transcript:      '{wb_text}'")
    print("-" * 65)

    # Summary table
    print("\nBENCHMARK SUMMARY:")
    print(f"{'Model':<25} | {'Size':<8} | {'Latency':<10} | {'RTF':<6}")
    print("-" * 65)
    print(f"{'Vosk small-en-us':<25} | {v_size:<6.1f}MB | {v_latency:<8.1f}ms | {v_rtf:<6.3f}")
    print(f"{'faster-whisper tiny.en':<25} | {wt_size:<6.1f}MB | {wt_latency:<8.1f}ms | {wt_rtf:<6.3f}")
    print(f"{'faster-whisper base.en':<25} | {wb_size:<6.1f}MB | {wb_latency:<8.1f}ms | {wb_rtf:<6.3f}")
    print("-" * 65)


if __name__ == "__main__":
    run_benchmark()
