"""Synthetic audio fixtures for fast, deterministic audio-module unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 22050


def _write_wav(tmp_path: Path, name: str, y: np.ndarray, sr: int = SR) -> Path:
    path = tmp_path / name
    sf.write(str(path), y, sr)
    return path


@pytest.fixture
def clicktrack_wav(tmp_path: Path) -> Path:
    """A steady 120 BPM click track: short percussive clicks on every beat."""
    bpm = 120.0
    duration = 8.0
    beat_period = 60.0 / bpm
    y = np.zeros(int(SR * duration), dtype=np.float32)
    t = 0.0
    click_len = int(SR * 0.02)
    click = (np.random.default_rng(0).standard_normal(click_len) * 0.8).astype(np.float32)
    envelope = np.linspace(1.0, 0.0, click_len, dtype=np.float32)
    click *= envelope
    while t < duration:
        start = int(t * SR)
        end = min(start + click_len, len(y))
        y[start:end] += click[: end - start]
        t += beat_period
    return _write_wav(tmp_path, "clicktrack.wav", y)


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    """A continuous 440Hz tone (loud, tonal, non-speech-band-adjacent)."""
    duration = 3.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return _write_wav(tmp_path, "tone.wav", y)


@pytest.fixture
def speech_like_wav(tmp_path: Path) -> Path:
    """A synthetic 'speech-like' signal: modulated formant-ish tones in vocal range."""
    duration = 4.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    carrier = 0.4 * np.sin(2 * np.pi * 250.0 * t) + 0.2 * np.sin(2 * np.pi * 1200.0 * t)
    # Amplitude-modulate to mimic syllabic energy bursts (~4 Hz, typical speech rate).
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t))
    y = (carrier * envelope).astype(np.float32)
    return _write_wav(tmp_path, "speech_like.wav", y)


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    duration = 3.0
    y = np.zeros(int(SR * duration), dtype=np.float32)
    return _write_wav(tmp_path, "silent.wav", y)


@pytest.fixture
def loud_then_silent_wav(tmp_path: Path) -> Path:
    """2s of tone, then 2s of true silence, then 2s of tone again."""
    seg = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    tone = (0.6 * np.sin(2 * np.pi * 440.0 * seg)).astype(np.float32)
    silence = np.zeros_like(tone)
    y = np.concatenate([tone, silence, tone])
    return _write_wav(tmp_path, "loud_silent_loud.wav", y)
