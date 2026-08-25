"""Gegenstück zu DummyEngine: liest den Text zurück, der das Audio erzeugt hat."""

from __future__ import annotations

import numpy as np

from cloney.engines.dummy import lookup

_GARBAGE = "völlig anderer inhalt als erwartet"


class DummyASR:
    """Perfekte Rückschrift -- oder gezielt fehlerhaft, um die Retry-Schleife zu prüfen.

    ``corrupt_seeds`` verfälscht die Rückschrift genau dann, wenn der Chunk mit
    einem dieser Seeds erzeugt wurde. Ein erneuter Versuch würfelt einen anderen
    Seed und liefert dann ein sauberes Ergebnis -- genau das Verhalten, das die
    Retry-Schleife abfangen soll. ``always_corrupt`` bildet den Fall ab, in dem
    auch kein Neuversuch mehr hilft.
    """

    def __init__(
        self,
        corrupt_seeds: set[int] | None = None,
        always_corrupt: bool = False,
    ) -> None:
        self.corrupt_seeds = corrupt_seeds or set()
        self.always_corrupt = always_corrupt

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "de") -> str:
        found = lookup(audio)
        if found is None:
            return ""
        text, seed = found
        if self.always_corrupt or seed in self.corrupt_seeds:
            return _GARBAGE
        return text

    def close(self) -> None:
        return None
