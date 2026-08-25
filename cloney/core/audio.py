"""Audio-Ein-/Ausgabe, Lautheit und Zusammenbau der Chunks zur fertigen Spur."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf

#: pyloudnorm braucht mindestens einen vollen 400-ms-Block für die Messung.
_MIN_LUFS_SECONDS = 0.45


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono(audio), int(sample_rate)


def write_wav(path: Path | str, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0).astype(np.float32), sample_rate, subtype="PCM_16")


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    return len(audio) / float(sample_rate)


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(seconds * sample_rate)), dtype=np.float32)


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20.0 * np.log10(peak) if peak > 0 else -np.inf


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -45.0,
    margin_ms: int = 40,
) -> np.ndarray:
    """Schneidet Stille an Anfang und Ende weg, lässt aber einen Rand stehen.

    Ohne Rand klingt der Einsatz abgehackt; mit zu viel Rand summieren sich über
    hundert Chunks mehrere Sekunden ungewollter Pause an.
    """
    if audio.size == 0:
        return audio

    frame = max(1, sample_rate // 100)
    frames = audio[: len(audio) - len(audio) % frame].reshape(-1, frame)
    if frames.size == 0:
        return audio

    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    with np.errstate(divide="ignore"):
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    loud = np.flatnonzero(rms_db > threshold_db)
    if loud.size == 0:
        return audio[:0]

    margin = int(sample_rate * margin_ms / 1000)
    start = max(0, loud[0] * frame - margin)
    end = min(len(audio), (loud[-1] + 1) * frame + margin)
    return audio[start:end]


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float | None:
    if duration_seconds(audio, sample_rate) < _MIN_LUFS_SECONDS:
        return None
    meter = pyloudnorm.Meter(sample_rate)
    loudness = float(meter.integrated_loudness(audio.astype(np.float64)))
    return None if np.isinf(loudness) or np.isnan(loudness) else loudness


def normalize_lufs(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    """Hebt/senkt auf die Ziel-Lautheit. Zu kurze oder stille Chunks bleiben unberührt."""
    loudness = measure_lufs(audio, sample_rate)
    if loudness is None:
        return audio
    gain = 10.0 ** ((target_lufs - loudness) / 20.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def apply_edge_fade(audio: np.ndarray, sample_rate: int, fade_ms: int = 12) -> np.ndarray:
    """Kurze Ein-/Ausblende gegen Knackser an den Schnittstellen."""
    n = int(sample_rate * fade_ms / 1000)
    if n <= 0 or audio.size < 2 * n:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def assemble(
    segments: list[tuple[np.ndarray, bool]],
    sample_rate: int,
    target_lufs: float = -16.0,
    pause_sentence_ms: int = 350,
    pause_paragraph_ms: int = 800,
    edge_fade_ms: int = 12,
    trim_threshold_db: float = -45.0,
) -> np.ndarray:
    """Fügt Chunks zur fertigen Spur zusammen.

    ``segments`` ist eine Liste aus (Audio, endet_am_Absatz). Jeder Chunk wird
    einzeln getrimmt und auf die Ziel-Lautheit gebracht -- das ist der Grund,
    warum die Stimme über ein ganzes Kapitel gleich laut bleibt, auch wenn
    einzelne Chunks neu gerendert wurden.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)

    parts: list[np.ndarray] = []
    for index, (audio, ends_paragraph) in enumerate(segments):
        cleaned = trim_silence(audio, sample_rate, trim_threshold_db)
        cleaned = normalize_lufs(cleaned, sample_rate, target_lufs)
        cleaned = apply_edge_fade(cleaned, sample_rate, edge_fade_ms)
        parts.append(cleaned)
        if index < len(segments) - 1:
            pause_ms = pause_paragraph_ms if ends_paragraph else pause_sentence_ms
            parts.append(silence(pause_ms / 1000.0, sample_rate))

    return np.concatenate(parts).astype(np.float32)
