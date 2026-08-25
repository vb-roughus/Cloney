"""Schnittstelle der Spracherkennung (Referenz-Transkription und QC-Rückschrift)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASREngine(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "de") -> str: ...

    def close(self) -> None: ...
