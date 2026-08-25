"""Schnittstelle der Spracherkennung (Referenz-Transkription und QC-Rückschrift)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TranscribedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcript:
    """Rückschrift, wahlweise mit Zeitangaben je Wort.

    Die Zeiten braucht die Erkennung des Referenz-Vorspanns: nur mit ihnen lässt
    sich sagen, ab welcher Sekunde der eigentlich gewünschte Text beginnt.
    """

    text: str
    words: tuple[TranscribedWord, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        return self.text

    def __bool__(self) -> bool:
        return bool(self.text)


@runtime_checkable
class ASREngine(Protocol):
    def transcribe(
        self, audio: np.ndarray, sample_rate: int, language: str = "de"
    ) -> Transcript: ...

    def close(self) -> None: ...
