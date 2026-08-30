"""Buchstabierte Abkürzungen und die Suche nach Kandidaten.

Geprüft wird nur, was eine Regel hergibt. Wie ``SWIFT`` klingen soll, steht
nicht hier -- das entscheidet ein Mensch und trägt es ein.
"""

from __future__ import annotations

import pytest

from cloney.core.pronounce import acronyms, spell_out


@pytest.mark.parametrize(
    ("wort", "erwartet"),
    [
        ("USB", "U-Es-Be"),
        ("GmbH", "Ge-Em-Be-Ha"),
        ("EU", "E-U"),
        ("ÖV", "Ö-Vau"),
        ("MP3", "Em-Pe-Drei"),
        ("xy", "Iks-Ypsilon"),
    ],
)
def test_buchstabieren(wort: str, erwartet: str) -> None:
    assert spell_out(wort) == erwartet


def test_bindestrich_haelt_die_buchstaben_zusammen() -> None:
    """Mit Leerzeichen liest eine Engine drei einzelne Wörter, jedes mit
    eigener Betonung."""
    assert spell_out("USB", trenner=" ") == "U Es Be"


def test_kandidaten_sind_ketten_aus_grossbuchstaben() -> None:
    text = "Die SWIFT-Nachricht war ACID-konform, sagte Dr. Meier am 3. Mai."

    assert acronyms(text) == ["SWIFT", "ACID"]


def test_ein_einzelner_grossbuchstabe_ist_kein_kandidat() -> None:
    """Sonst wäre jeder Satzanfang und jede Initiale ein Fund."""
    assert acronyms("Er kam. Sie ging. A. Meier blieb.") == []


def test_jeder_kandidat_steht_nur_einmal() -> None:
    assert acronyms("ACID und ACID und nochmal ACID") == ["ACID"]


def test_gemischte_schreibweise_ist_kein_kandidat() -> None:
    """'iPhone' und 'GmbH' werden nicht buchstabiert -- sie sind Wörter."""
    assert acronyms("Das iPhone der Meier GmbH") == []
