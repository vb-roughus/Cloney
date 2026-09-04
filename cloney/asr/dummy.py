"""Gegenstück zu DummyEngine: liest den Text zurück, der das Audio erzeugt hat."""

from __future__ import annotations

import numpy as np

from cloney.asr.base import TranscribedWord, Transcript
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
        bleed_words: str = "",
        stummer_vorspann: float = 0.0,
    ) -> None:
        self.corrupt_seeds = corrupt_seeds or set()
        self.always_corrupt = always_corrupt
        #: Wörter, die der Rückschrift vorangestellt werden -- bildet den
        #: Referenz-Vorspann nach, den F5-TTS am Anfang stehen lassen kann.
        self.bleed_words = bleed_words
        #: Sekunden Vorspann, den die Erkennung *nicht* aufschreibt. Der andere
        #: und häufigere Fall: eine angerissene Silbe ist zu hören, aber kein
        #: Wort, das sich zuordnen ließe. Verraten wird sie nur dadurch, dass
        #: das erste Wort erst spät beginnt.
        self.stummer_vorspann = stummer_vorspann

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "de") -> Transcript:
        found = lookup(audio)
        if found is None:
            return Transcript("")
        text, seed = found
        if self.always_corrupt or seed in self.corrupt_seeds:
            text = _GARBAGE

        vorspann = self.bleed_words.split()
        alle = vorspann + text.split()
        # Gleichmäßig über die Dauer verteilte Zeiten genügen: geprüft wird die
        # Zuordnung von Wörtern zu Zeiten, nicht die Genauigkeit von Whisper.
        dauer = len(audio) / sample_rate if len(audio) else 0.0
        rest = max(0.0, dauer - self.stummer_vorspann)
        schritt = rest / len(alle) if alle else 0.0
        woerter = tuple(
            TranscribedWord(
                wort,
                self.stummer_vorspann + i * schritt,
                self.stummer_vorspann + (i + 1) * schritt,
            )
            for i, wort in enumerate(alle)
        )
        return Transcript(text=" ".join(alle), words=woerter)

    def close(self) -> None:
        return None
