"""Gegenstück zu DummyEngine: liest den Text zurück, der das Audio erzeugt hat."""

from __future__ import annotations

import numpy as np

from cloney.engines.dummy import lookup


class DummyASR:
    """Perfekte Rückschrift -- oder gezielt fehlerhaft, um die Retry-Schleife zu prüfen.

    ``corrupt_seeds`` verfälscht die Rückschrift genau dann, wenn der Chunk mit
    einem dieser Seeds erzeugt wurde. Ein erneuter Versuch würfelt einen anderen
    Seed und liefert dann ein sauberes Ergebnis -- genau das Verhalten, das die
    Retry-Schleife abfangen soll.
    """

    def __init__(self, corrupt_seeds: set[int] | None = None) -> None:
        self.corrupt_seeds = corrupt_seeds or set()

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "de") -> str:
        found = lookup(audio)
        if found is None:
            return ""
        text, seed = found
        if seed in self.corrupt_seeds:
            return "völlig anderer inhalt als erwartet"
        return text

    def close(self) -> None:
        return None
