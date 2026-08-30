"""Was auf der Startseite steht.

Eine Übersicht ist schnell zu viel: jede Zahl, die sich zählen lässt, findet
sonst ihre Kachel. Hier stehen deshalb nur Zahlen, aus denen ein nächster
Schritt folgt -- was läuft, was auf Durchsicht wartet, wie weit die Arbeit ist --
und daneben der Bestand, der sagt, womit überhaupt gearbeitet werden kann.

Eigenes Modul, weil sich das so ohne HTTP prüfen lässt: die Zählung ist der
Teil, der falsch sein kann, nicht das Ausliefern der Seite.
"""

from __future__ import annotations

from dataclasses import dataclass

from cloney.core.project import ChunkStatus, Project


@dataclass(frozen=True)
class Overview:
    """Kennzahlen der Startseite."""

    projects: int
    complete: int
    #: Projekte, die gerade rendern.
    running: int
    chunks_done: int
    chunks_total: int
    #: Sätze, die auf einen Menschen warten -- die einzige Zahl hier, aus der
    #: unmittelbar Arbeit folgt.
    review: int
    voices: int
    comparisons: int
    models: int

    @property
    def in_arbeit(self) -> int:
        return self.projects - self.complete

    @property
    def leer(self) -> bool:
        """Nichts angelegt -- dann trägt die Seite eine Anleitung statt Kacheln."""
        return not (self.projects or self.voices or self.comparisons or self.models)


def summarize(
    projects: list[Project],
    *,
    running: int = 0,
    voices: int = 0,
    comparisons: int = 0,
    models: int = 0,
) -> Overview:
    fertig = sum(1 for p in projects if p.is_complete)
    erledigt = 0
    gesamt = 0
    durchsicht = 0
    for project in projects:
        done, total = project.progress
        erledigt += done
        gesamt += total
        durchsicht += sum(
            1 for c in project.chunks if c.status in (ChunkStatus.NEEDS_REVIEW, ChunkStatus.FAILED)
        )
    return Overview(
        projects=len(projects),
        complete=fertig,
        running=running,
        chunks_done=erledigt,
        chunks_total=gesamt,
        review=durchsicht,
        voices=voices,
        comparisons=comparisons,
        models=models,
    )
