from __future__ import annotations

from cloney.core.metrics import cer, normalize_for_comparison, wer


def test_interpunktion_und_grossschreibung_zaehlen_nicht() -> None:
    """Die ASR-Schreibkonvention darf die Kennzahl nicht verfälschen."""
    assert cer("Das ist ein Test.", "das ist ein test") == 0.0
    assert wer("Das ist ein Test.", "das ist ein test") == 0.0


def test_umlaute_werden_vereinheitlicht() -> None:
    assert cer("Grüße aus München", "Gruesse aus Muenchen") == 0.0


def test_abweichung_wird_gemessen() -> None:
    assert 0.0 < cer("Das ist ein Test", "Das ist kein Fest") < 0.5
    assert wer("Das ist ein Test", "Das ist kein Fest") == 0.5


def test_leere_rueckschrift_ist_totalausfall() -> None:
    assert cer("Ein Satz", "") == 1.0


def test_leere_referenz() -> None:
    assert cer("", "") == 0.0
    assert cer("", "etwas") == 1.0


def test_vergleichsform() -> None:
    assert normalize_for_comparison("  Hallo,   Welt! ") == "hallo welt"
