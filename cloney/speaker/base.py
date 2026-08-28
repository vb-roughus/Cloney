"""Schnittstelle für Sprecher-Einbettungen.

Die Fehlerrate prüft, ob die richtigen Wörter herauskommen. Ob es dabei noch
nach der Referenzstimme klingt, sagt sie nicht -- ein Chunk kann fehlerfrei
sein und nach jemand anderem klingen. Diese Lücke schließt ein Vergleich der
Stimmeinbettungen von Referenz und Ergebnis.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SpeakerEmbedder(Protocol):
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Ein Vektor, der die Stimme beschreibt -- nicht den Inhalt."""
        ...

    def close(self) -> None: ...
