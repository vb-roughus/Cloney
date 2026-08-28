"""Audio-Ein-/Ausgabe, Lautheit und Zusammenbau der Chunks zur fertigen Spur."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf

#: pyloudnorm braucht mindestens einen vollen 400-ms-Block für die Messung.
_MIN_LUFS_SECONDS = 0.45

#: Auflösung, in der erzeugter Ton geschrieben wird.
#:
#: Die Modelle liefern Gleitkomma. Ein Chunk wird geschrieben, für den
#: Zusammenbau wieder gelesen, angeglichen und erneut geschrieben -- bei
#: 16 Bit quantisiert das zweimal. Hörbar ist das kaum, vermeidbar aber
#: umsonst, und "so gut wie das Modell es hergibt" ist die richtige Vorgabe
#: für eine Zwischendatei.
DEFAULT_SUBTYPE = "PCM_24"

#: Wieviel Luft bis zur Vollaussteuerung nach der Lautheitsangleichung bleibt.
#: Sprache hat einen hohen Scheitelfaktor; ohne diese Grenze könnte die
#: Anhebung auf die Ziel-Lautheit die Spitzen abschneiden -- und ein
#: abgeschnittener Spitzenwert ist hörbare Verzerrung, kein Rundungsfehler.
PEAK_CEILING_DBFS = -1.0


@dataclass(frozen=True)
class AudioInfo:
    """Was in einer Tondatei steckt, ohne sie zu laden."""

    sample_rate: int
    channels: int
    subtype: str
    format: str
    duration_s: float

    def beschreibung(self) -> str:
        kanaele = {1: "Mono", 2: "Stereo"}.get(self.channels, f"{self.channels} Kanäle")
        return f"{self.sample_rate} Hz, {kanaele}, {self.subtype}"


def describe_audio(path: Path | str) -> AudioInfo:
    info = sf.info(str(path))
    return AudioInfo(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=str(info.subtype),
        format=str(info.format),
        duration_s=float(info.duration),
    )


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono(audio), int(sample_rate)


def write_wav(
    path: Path | str, audio: np.ndarray, sample_rate: int, subtype: str = DEFAULT_SUBTYPE
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0).astype(np.float32), sample_rate, subtype=subtype)


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


def normalize_lufs(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    peak_ceiling_db: float = PEAK_CEILING_DBFS,
) -> np.ndarray:
    """Hebt/senkt auf die Ziel-Lautheit. Zu kurze oder stille Chunks bleiben unberührt.

    Reicht die Aussteuerung für die Ziel-Lautheit nicht, gewinnt die Spitze:
    der Chunk wird so weit zurückgenommen, dass er unter die Grenze passt, und
    bleibt damit etwas leiser als gewollt. Vorher wurde stattdessen bei
    Vollaussteuerung abgeschnitten -- das hätte die lautesten Stellen verzerrt,
    und ein um ein Dezibel zu leiser Satz ist das kleinere Übel.
    """
    loudness = measure_lufs(audio, sample_rate)
    if loudness is None:
        return audio
    gain = 10.0 ** ((target_lufs - loudness) / 20.0)
    verstaerkt = audio * gain

    grenze = 10.0 ** (peak_ceiling_db / 20.0)
    spitze = float(np.max(np.abs(verstaerkt))) if verstaerkt.size else 0.0
    if spitze > grenze:
        verstaerkt *= grenze / spitze
    return verstaerkt.astype(np.float32)


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
