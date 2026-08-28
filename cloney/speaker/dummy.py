"""Modellfreie Stimmeinbettung für Tests und CI.

Genommen wird das gemittelte Betragsspektrum. Das ist keine Sprechererkennung --
aber es verhält sich wie eine: gleiches Audio ergibt denselben Vektor, ähnliches
einen ähnlichen, deutlich anderes einen entfernten. Für die Prüfung der
Verkettung und der Kennzahl genügt das, und es braucht kein Modell.
"""

from __future__ import annotations

import numpy as np

_BAENDER = 64


class DummySpeakerEmbedder:
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.size == 0:
            return np.zeros(_BAENDER, dtype=np.float32)

        spektrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
        # Auf eine feste Zahl Bänder zusammenfassen, damit unterschiedlich lange
        # Aufnahmen vergleichbare Vektoren ergeben.
        kanten = np.linspace(0, len(spektrum), _BAENDER + 1).astype(int)
        paare = zip(kanten[:-1], kanten[1:], strict=True)
        baender = np.array([spektrum[a:b].mean() if b > a else 0.0 for a, b in paare])
        return np.log1p(baender).astype(np.float32)

    def close(self) -> None:
        return None
