from __future__ import annotations

from pathlib import Path

import numpy as np

from cloney.core.audio import (
    assemble,
    duration_seconds,
    measure_lufs,
    normalize_lufs,
    read_wav,
    silence,
    trim_silence,
    write_wav,
)

SR = 24000


def _tone(seconds: float, amplitude: float = 0.2, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_wav_roundtrip(tmp_path: Path) -> None:
    audio = _tone(1.0)
    path = tmp_path / "t.wav"
    write_wav(path, audio, SR)
    back, rate = read_wav(path)
    assert rate == SR
    assert np.max(np.abs(back - audio)) < 1e-4


def test_stille_wird_getrimmt() -> None:
    padded = np.concatenate([silence(0.5, SR), _tone(1.0), silence(0.5, SR)])
    trimmed = trim_silence(padded, SR)
    assert 1.0 <= duration_seconds(trimmed, SR) < 1.3


def test_lautheit_wird_angeglichen() -> None:
    quiet = normalize_lufs(_tone(2.0, amplitude=0.02), SR, -16.0)
    assert abs(measure_lufs(quiet, SR) - (-16.0)) < 1.0


def test_zu_kurzes_audio_bleibt_unberuehrt() -> None:
    """Unter 400 ms kann die Lautheit nicht gemessen werden -- dann lieber nichts tun."""
    short = _tone(0.1)
    assert np.array_equal(normalize_lufs(short, SR, -16.0), short)


def test_pausen_richten_sich_nach_der_absatzgrenze() -> None:
    segment = _tone(1.0)
    track = assemble(
        [(segment, False), (segment, True), (segment, False)],
        SR,
        pause_sentence_ms=300,
        pause_paragraph_ms=900,
    )
    # Drei Chunks (getrimmt, mit Rand) plus eine Satz- und eine Absatzpause.
    expected_pauses = (0.3 + 0.9) * SR
    assert len(track) > 3 * SR + expected_pauses * 0.9


def test_assemble_ohne_segmente() -> None:
    assert assemble([], SR).size == 0


def test_ausgabe_uebersteuert_nicht() -> None:
    loud = _tone(2.0, amplitude=0.9)
    track = assemble([(loud, False), (loud, True)], SR, target_lufs=-16.0)
    assert np.max(np.abs(track)) <= 1.0
