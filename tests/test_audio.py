from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cloney.core.audio import (
    PEAK_CEILING_DBFS,
    Segment,
    assemble,
    duration_seconds,
    measure_lufs,
    normalize_lufs,
    peak_dbfs,
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
        [Segment(segment), Segment(segment, ends_paragraph=True), Segment(segment)],
        SR,
        pause_sentence_ms=300,
        pause_paragraph_ms=900,
    )
    # Drei Chunks (getrimmt, mit Rand) plus eine Satz- und eine Absatzpause.
    expected_pauses = (0.3 + 0.9) * SR
    assert len(track) > 3 * SR + expected_pauses * 0.9


def test_assemble_ohne_segmente() -> None:
    assert assemble([], SR).size == 0


def test_nach_einer_ueberschrift_steht_die_laengste_pause() -> None:
    """Im Hörbuch trennt genau diese Pause den Titel vom Text. Ohne sie klingt
    das Kapitel, als hätte jemand vergessen abzusetzen."""
    segment = _tone(1.0)
    mit_titel = assemble(
        [Segment(segment, ends_paragraph=True, is_heading=True), Segment(segment)],
        SR,
        pause_paragraph_ms=800,
        pause_heading_ms=1600,
    )
    ohne_titel = assemble(
        [Segment(segment, ends_paragraph=True), Segment(segment)],
        SR,
        pause_paragraph_ms=800,
        pause_heading_ms=1600,
    )

    assert len(mit_titel) - len(ohne_titel) == pytest.approx(0.8 * SR, abs=SR // 100)


def test_ausgabe_uebersteuert_nicht() -> None:
    loud = _tone(2.0, amplitude=0.9)
    track = assemble([Segment(loud), Segment(loud, ends_paragraph=True)], SR, target_lufs=-16.0)
    assert np.max(np.abs(track)) <= 1.0


# -- Nichts verschenken, was das Modell hergibt ------------------------------


def test_erzeugter_ton_wird_mit_24_bit_geschrieben(tmp_path) -> None:  # noqa: ANN001
    """Ein Chunk wird geschrieben, für den Zusammenbau gelesen, angeglichen und
    erneut geschrieben. Bei 16 Bit quantisiert das zweimal -- vermeidbar."""
    import soundfile as sf

    pfad = tmp_path / "chunk.wav"
    write_wav(pfad, _tone(1.0), SR)
    assert sf.info(pfad).subtype == "PCM_24"


def test_lautheitsangleichung_schneidet_keine_spitzen_ab() -> None:
    """Sprache hat einen hohen Scheitelfaktor. Ohne Grenze könnte die Anhebung
    auf die Ziel-Lautheit die lautesten Stellen abschneiden -- und ein
    abgeschnittener Spitzenwert ist hörbare Verzerrung, kein Rundungsfehler."""
    # Kurze, kräftige Spitzen auf leisem Grund: laut gemessen, aber sehr spitz.
    t = np.arange(3 * SR, dtype=np.float32) / SR
    spitz = (np.abs(np.sin(2 * np.pi * 4.0 * t)) ** 8 * np.sin(2 * np.pi * 200 * t)).astype(
        np.float32
    )

    laut = normalize_lufs(spitz, SR, target_lufs=-6.0)

    assert peak_dbfs(laut) <= PEAK_CEILING_DBFS + 0.01
    # Kein plattgedrückter Spitzenwert: die Form bleibt, nur der Pegel sinkt.
    verhaeltnis = float(np.max(np.abs(laut))) / float(np.max(np.abs(spitz)))
    assert np.allclose(laut, spitz * verhaeltnis, atol=1e-5)


def test_leise_aufnahme_erreicht_die_ziellautheit() -> None:
    """Die Grenze greift nur, wenn sie muss -- sonst bliebe alles zu leise."""
    leise = _tone(3.0) * 0.05
    angeglichen = normalize_lufs(leise, SR, target_lufs=-16.0)
    assert measure_lufs(angeglichen, SR) == pytest.approx(-16.0, abs=0.5)
