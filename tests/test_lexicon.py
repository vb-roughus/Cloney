"""Das Aussprache-Wörterbuch.

Was hier geprüft wird, ist die Mechanik: finden, ersetzen, speichern. Welche
Sprechweise richtig ist, steht nirgends im Code -- das entscheidet der Mensch,
der den Text kennt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cloney.core.lexicon import Lexicon
from cloney.core.normalize import normalize_german


def test_leeres_woerterbuch_laesst_den_text_in_ruhe() -> None:
    assert Lexicon().apply("Die SWIFT-Nachricht kam.") == "Die SWIFT-Nachricht kam."


def test_eintrag_wird_ersetzt() -> None:
    lexikon = Lexicon(entries={"SWIFT": "Ssuift"})
    assert lexikon.apply("Die SWIFT-Nachricht kam.") == "Die Ssuift-Nachricht kam."


def test_gross_und_kleinschreibung_finden_dasselbe_wort() -> None:
    """Im Text steht mal 'SWIFT', mal 'Swift'. Gemeint ist beide Male dasselbe."""
    lexikon = Lexicon(entries={"SWIFT": "Ssuift"})

    assert lexikon.apply("swift und Swift und SWIFT") == "Ssuift und Ssuift und Ssuift"


def test_nur_ganze_woerter() -> None:
    """Sonst träfe ein Eintrag mitten in einem anderen Wort."""
    lexikon = Lexicon(entries={"ACID": "Ässid"})

    assert lexikon.apply("PLACIDE bleibt PLACIDE") == "PLACIDE bleibt PLACIDE"


def test_laengster_eintrag_gewinnt() -> None:
    """Sonst schlüge der kürzere zuerst zu und der längere käme nie zum Zug."""
    lexikon = Lexicon(entries={"SWIFT": "Ssuift", "SWIFT Code": "Ssuift Kohd"})

    assert lexikon.apply("Der SWIFT Code lautet") == "Der Ssuift Kohd lautet"


def test_sprechweise_wird_selbst_normalisiert() -> None:
    """Wer 'Em-Pe-3' einträgt, will nicht 'Em-Pe-3' hören."""
    lexikon = Lexicon(entries={"MP3": "Em-Pe-3"})

    assert normalize_german("Als MP3 gespeichert.", lexikon) == "Als Em-Pe-drei gespeichert."


def test_ohne_woerterbuch_bleibt_die_normalisierung_wie_sie_war() -> None:
    assert normalize_german("Am 3. Mai 2024.") == normalize_german("Am 3. Mai 2024.", Lexicon())


def test_speichern_und_laden(tmp_path: Path) -> None:
    lexikon = Lexicon()
    lexikon.set("Journal", "Schurnahl")
    lexikon.save(tmp_path)

    assert Lexicon.load(tmp_path).entries == {"Journal": "Schurnahl"}


def test_fehlendes_woerterbuch_ist_kein_fehler(tmp_path: Path) -> None:
    assert Lexicon.load(tmp_path / "gibt-es-nicht").entries == {}


def test_eintrag_ohne_sprechweise_wird_abgelehnt() -> None:
    """Ein leerer Eintrag ließe das Wort verschwinden -- und niemand merkte es,
    bis die Stelle im Hörbuch fehlt."""
    with pytest.raises(ValueError, match="Sprechweise"):
        Lexicon().set("SWIFT", "   ")


def test_leeres_wort_wird_abgelehnt() -> None:
    with pytest.raises(ValueError, match="nicht leer"):
        Lexicon().set("  ", "Ssuift")


def test_entfernen_meldet_ob_etwas_da_war() -> None:
    lexikon = Lexicon(entries={"SWIFT": "Ssuift"})

    assert lexikon.remove("SWIFT") is True
    assert lexikon.remove("SWIFT") is False
