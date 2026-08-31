"""Sätze aussuchen -- die Auswahl, nicht die Seite.

Ein Kapitel hat schnell hundert Sätze. Was hier festgehalten wird, ist das,
worauf man sich beim Durchhören verlässt: dass "zu prüfen" wirklich alle
auffälligen zeigt und die Suche auch dort greift, wo man sie braucht.
"""

from __future__ import annotations

import pytest

from cloney.core.project import Chunk, ChunkStatus
from cloney.web.filters import select


def _chunk(index: int, status: ChunkStatus, raw: str = "", asr: str | None = None) -> Chunk:
    return Chunk(
        index=index,
        raw_text=raw or f"Satz {index}",
        normalized_text=raw or f"Satz {index}",
        seed=1,
        status=status,
        asr_text=asr,
    )


@pytest.fixture
def kapitel() -> list[Chunk]:
    return [
        _chunk(0, ChunkStatus.OK, "Am dritten Mai begann es."),
        _chunk(1, ChunkStatus.NEEDS_REVIEW, "Die SWIFT-Nachricht kam."),
        _chunk(2, ChunkStatus.PENDING, "Noch nicht erzeugt."),
        _chunk(3, ChunkStatus.FAILED, "Hier ging etwas schief."),
        _chunk(4, ChunkStatus.OK, "Zum Schluss noch dies."),
        _chunk(5, ChunkStatus.SYNTHESIZED, "Erzeugt, aber ungeprüft."),
    ]


def test_ohne_filter_bleibt_alles(kapitel: list[Chunk]) -> None:
    auswahl = select(kapitel)

    assert len(auswahl.chunks) == 6
    assert not auswahl.gefiltert


def test_zu_pruefen_zeigt_auffaellige_und_gescheiterte(kapitel: list[Chunk]) -> None:
    """Beides ist derselbe Fall: nachhören und entscheiden."""
    auswahl = select(kapitel, status="pruefen")

    assert [c.index for c in auswahl.chunks] == [1, 3]
    assert auswahl.gefiltert
    assert auswahl.gesamt == 6


def test_offen_zeigt_auch_das_ungeprueft_erzeugte(kapitel: list[Chunk]) -> None:
    """Ein Satz ohne Messung ist nicht fertig -- er wartet noch auf die
    Qualitätskontrolle."""
    assert [c.index for c in select(kapitel, status="offen").chunks] == [2, 5]


def test_fertig_zeigt_nur_geprueftes(kapitel: list[Chunk]) -> None:
    assert [c.index for c in select(kapitel, status="fertig").chunks] == [0, 4]


def test_suche_findet_im_rohtext(kapitel: list[Chunk]) -> None:
    assert [c.index for c in select(kapitel, query="swift").chunks] == [1]


def test_suche_achtet_nicht_auf_gross_und_kleinschreibung(kapitel: list[Chunk]) -> None:
    assert select(kapitel, query="SCHLUSS").chunks == select(kapitel, query="schluss").chunks


def test_suche_greift_auch_in_der_rueckschrift() -> None:
    """Wer einer schlechten Aussprache nachgeht, sucht genau dort: im Rohtext
    steht das richtige Wort, gehört wurde ein anderes."""
    chunks = [_chunk(0, ChunkStatus.OK, "Die Atomarität zählt.", asr="Die Atom Arität zählt.")]

    assert select(chunks, query="atom arität").chunks == chunks


def test_zustand_und_suche_greifen_zusammen(kapitel: list[Chunk]) -> None:
    assert select(kapitel, status="pruefen", query="schief").chunks == [kapitel[3]]
    assert select(kapitel, status="fertig", query="schief").leer


def test_unbekannter_zustand_zeigt_alles(kapitel: list[Chunk]) -> None:
    """Der Wert kommt aus einer URL. Eine leere Tabelle sähe aus wie ein Fehler
    im Projekt, nicht wie ein Tippfehler in der Adresse."""
    auswahl = select(kapitel, status="gibtesnicht")

    assert len(auswahl.chunks) == 6
    assert auswahl.status == "alle"


def test_leerraum_im_suchbegriff_zaehlt_nicht(kapitel: list[Chunk]) -> None:
    assert select(kapitel, query="  ").chunks == kapitel
    assert select(kapitel, query="  swift  ").query == "swift"
