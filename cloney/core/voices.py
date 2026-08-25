"""Verwaltung der Referenzstimmen samt Eingangsprüfung.

Eine schlechte Referenzaufnahme ist die häufigste Ursache für einen schlechten
Klon. Die Prüfung läuft deshalb beim Anlegen der Stimme -- nicht erst, wenn ein
ganzes Kapitel gerendert ist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cloney.core.audio import duration_seconds, peak_dbfs, read_wav, trim_silence, write_wav
from cloney.engines.base import VoiceRef

_SLUG = re.compile(r"[^a-z0-9]+")
_META = "voice.json"
_REFERENCE = "reference.wav"


@dataclass(frozen=True)
class VoiceCheck:
    """Ergebnis der Eingangsprüfung. ``ok`` heißt: brauchbar, nicht perfekt."""

    ok: bool
    warnings: list[str]
    duration_s: float
    sample_rate: int
    peak_dbfs: float
    speech_ratio: float


def inspect_reference(
    audio: np.ndarray,
    sample_rate: int,
    min_seconds: float = 5.0,
    max_seconds: float = 20.0,
) -> VoiceCheck:
    duration = duration_seconds(audio, sample_rate)
    peak = peak_dbfs(audio)
    speech = trim_silence(audio, sample_rate)
    ratio = (len(speech) / len(audio)) if len(audio) else 0.0

    warnings: list[str] = []
    if duration < min_seconds:
        warnings.append(
            f"Nur {duration:.1f}s Referenz. Unter {min_seconds:.0f}s wird der Klon instabil."
        )
    if duration > max_seconds:
        warnings.append(
            f"{duration:.1f}s Referenz. Über {max_seconds:.0f}s bringt keinen Gewinn "
            "und kostet bei jedem Chunk Rechenzeit."
        )
    if peak > -0.5:
        warnings.append("Aufnahme übersteuert (Peak nahe 0 dBFS). Verzerrungen werden mitgeklont.")
    if peak < -30.0:
        warnings.append(f"Sehr leise Aufnahme (Peak {peak:.0f} dBFS). Rauschen wird mitgeklont.")
    if sample_rate < 16000:
        warnings.append(f"Samplerate nur {sample_rate} Hz. Hohe Frequenzen fehlen dauerhaft.")
    if ratio < 0.5:
        warnings.append(
            f"Nur {ratio:.0%} der Aufnahme ist Sprache. Lange Pausen schwächen die Referenz."
        )

    blocking = duration < 1.0 or ratio < 0.2
    return VoiceCheck(
        ok=not blocking,
        warnings=warnings,
        duration_s=duration,
        sample_rate=sample_rate,
        peak_dbfs=peak,
        speech_ratio=ratio,
    )


class VoiceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, name: str) -> Path:
        return self.root / _slug(name)

    def exists(self, name: str) -> bool:
        return (self.path(name) / _META).exists()

    def add(
        self,
        name: str,
        audio_path: Path,
        transcript: str = "",
        min_seconds: float = 5.0,
        max_seconds: float = 20.0,
    ) -> tuple[VoiceRef, VoiceCheck]:
        audio, sample_rate = read_wav(audio_path)
        check = inspect_reference(audio, sample_rate, min_seconds, max_seconds)

        directory = self.path(name)
        directory.mkdir(parents=True, exist_ok=True)
        write_wav(directory / _REFERENCE, audio, sample_rate)
        (directory / _META).write_text(
            json.dumps(
                {
                    "name": name,
                    "transcript": transcript,
                    "sample_rate": sample_rate,
                    "duration_s": round(check.duration_s, 2),
                    "warnings": check.warnings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.get(name), check

    def set_transcript(self, name: str, transcript: str) -> None:
        meta_path = self.path(name) / _META
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["transcript"] = transcript
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, name: str) -> VoiceRef:
        directory = self.path(name)
        meta = json.loads((directory / _META).read_text(encoding="utf-8"))
        return VoiceRef(
            name=meta["name"],
            audio_path=directory / _REFERENCE,
            transcript=meta.get("transcript", ""),
        )

    def list_all(self) -> list[VoiceRef]:
        if not self.root.exists():
            return []
        return [self.get(d.name) for d in sorted(self.root.iterdir()) if (d / _META).exists()]


def _slug(name: str) -> str:
    return _SLUG.sub("-", name.lower()).strip("-") or "stimme"
