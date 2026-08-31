"""Sätze aussuchen: nach Zustand und nach Wortlaut.

Ein Kapitel hat schnell hundert Sätze. Wer nach einem Renderlauf die auffälligen
durchhören will, sucht sie sonst von Hand aus einer Liste heraus, in der
neunzig davon in Ordnung sind.

Gefiltert wird auf dem Server, nicht im Browser: die Satztabelle lädt sich
während eines Laufs alle zwei Sekunden neu, und ein Filter, der nur im DOM
steht, wäre nach dem ersten Austausch weg.
"""

from __future__ import annotations

from dataclasses import dataclass

from cloney.core.project import Chunk, ChunkStatus

#: Zustandsgruppen, wie sie in der Oberfläche stehen. Zusammengefasst nach dem,
#: was der Mensch damit vorhat -- 'auffällig' und 'fehlgeschlagen' sind beides
#: Fälle für dieselbe Runde Durchhören.
GRUPPEN: dict[str, tuple[ChunkStatus, ...]] = {
    "pruefen": (ChunkStatus.NEEDS_REVIEW, ChunkStatus.FAILED),
    "offen": (ChunkStatus.PENDING, ChunkStatus.SYNTHESIZED),
    "fertig": (ChunkStatus.OK,),
}

#: Beschriftung je Gruppe, in der Reihenfolge der Anzeige.
LABELS: tuple[tuple[str, str], ...] = (
    ("alle", "Alle"),
    ("pruefen", "Zu prüfen"),
    ("offen", "Offen"),
    ("fertig", "Fertig"),
)


@dataclass(frozen=True)
class Auswahl:
    """Was von einem Kapitel gerade zu sehen ist."""

    chunks: list[Chunk]
    #: Wie viele es insgesamt gibt -- ohne diese Zahl wüsste niemand, dass die
    #: Liste gekürzt ist.
    gesamt: int
    status: str = "alle"
    query: str = ""

    @property
    def gefiltert(self) -> bool:
        return len(self.chunks) != self.gesamt

    @property
    def leer(self) -> bool:
        return not self.chunks


def select(chunks: list[Chunk], status: str = "alle", query: str = "") -> Auswahl:
    """Sätze nach Zustand und Wortlaut aussuchen.

    Ein unbekannter Zustand zeigt alles, statt nichts: der Wert kommt aus einer
    URL, und eine leere Tabelle sähe aus wie ein Fehler im Projekt.
    """
    gewaehlt = list(chunks)
    zustaende = GRUPPEN.get(status)
    if zustaende is not None:
        gewaehlt = [c for c in gewaehlt if c.status in zustaende]

    begriff = query.strip().lower()
    if begriff:
        gewaehlt = [c for c in gewaehlt if _trifft(c, begriff)]

    return Auswahl(
        chunks=gewaehlt,
        gesamt=len(chunks),
        status=status if status in GRUPPEN else "alle",
        query=query.strip(),
    )


def _trifft(chunk: Chunk, begriff: str) -> bool:
    """Gesucht wird in allen drei Fassungen eines Satzes.

    Im Rohtext steht, was geschrieben wurde; in der Sprechfassung, was das
    Modell bekam; in der Rückschrift, was die Erkennung gehört hat. Wer eine
    Stelle sucht, weiß nicht immer, in welcher der drei sie steht -- und wer
    einer schlechten Aussprache nachgeht, sucht gerade in der Rückschrift.
    """
    return any(
        begriff in (feld or "").lower()
        for feld in (chunk.raw_text, chunk.normalized_text, chunk.asr_text)
    )
