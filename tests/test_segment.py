from __future__ import annotations

import pytest

from cloney.core.segment import build_chunks, heading_text, split_sentences


def test_abkuerzungen_trennen_keinen_satz() -> None:
    """Der naive Split am Punkt ist im Deutschen der klassische Fehler."""
    sentences = split_sentences("Dr. Meier sagte z.B., dass es u.a. daran liegt. Ende.")
    assert [s.raw for s in sentences] == [
        "Dr. Meier sagte z.B., dass es u.a. daran liegt.",
        "Ende.",
    ]


def test_ordinalzahl_trennt_keinen_satz() -> None:
    sentences = split_sentences("Am 3. Mai war es soweit. Danach nicht mehr.")
    assert len(sentences) == 2
    assert sentences[0].raw == "Am 3. Mai war es soweit."


def test_grosse_zahl_vor_punkt_ist_satzende() -> None:
    """'50.' ist keine plausible Ordinalzahl mehr -- hier endet der Satz."""
    sentences = split_sentences("Es waren 50. Danach ging es weiter.")
    assert len(sentences) == 2


def test_initialen_trennen_keinen_satz() -> None:
    assert len(split_sentences("Das schrieb J. W. Goethe damals.")) == 1


def test_absatzgrenzen_bleiben_erhalten() -> None:
    chunks = build_chunks("Erster Absatz hier.\n\nZweiter Absatz hier.", target_seconds=60)
    assert len(chunks) == 2
    assert all(c.ends_paragraph for c in chunks)


def test_chunks_bleiben_unter_der_zielgroesse() -> None:
    text = " ".join(f"Dies ist der Satz Nummer {i} in diesem Text." for i in range(40))
    chunks = build_chunks(text, chars_per_second=14.0, target_seconds=10.0)
    assert len(chunks) > 1
    # Die Zielgröße darf höchstens um einen Satz überschritten werden.
    assert all(c.estimated_seconds(14.0) < 10.0 + 6.0 for c in chunks)


def test_ueberlanger_satz_wird_an_teilsatzgrenzen_getrennt() -> None:
    long_sentence = ", ".join(f"Teilsatz Nummer {i} mit etwas Inhalt" for i in range(30)) + "."
    chunks = build_chunks(long_sentence, chars_per_second=14.0, target_seconds=5.0, max_seconds=8.0)
    assert len(chunks) > 1


def test_chunks_normalisieren_den_text() -> None:
    chunks = build_chunks("Am 3. Mai 2024 kostete es 1.250,50 €.")
    assert "dritten Mai" in chunks[0].normalized_text
    assert "3." not in chunks[0].normalized_text
    assert chunks[0].raw_text == "Am 3. Mai 2024 kostete es 1.250,50 €."


# -- Titel und Kapitelüberschriften -----------------------------------------


@pytest.mark.parametrize(
    ("zeile", "folgt", "erwartet"),
    [
        ("Kapitel 3", "", "Kapitel 3"),
        ("Der lange Weg", "", "Der lange Weg"),
        ("# Nachwort", "", "Nachwort"),
        ("### Zweiter Teil ###", "", "Zweiter Teil"),
        ("Prolog", "Es war einmal.", "Prolog"),
        # Kein Titel: endet wie ein Satz.
        ("Es war einmal ein Text.", "", None),
        ("Und dann?", "", None),
        ("Erstens:", "", None),
        # Kein Titel: zu lang, um eine Überschrift zu sein.
        ("Ein Satz ohne Punkt am Ende der schon deutlich zu lang ist um Titel zu sein", "", None),
        # Kein Titel: die Fortsetzung zeigt, dass hier ein Satz umbrochen wurde.
        ("Er ging über die Straße und", "dachte an nichts.", None),
        ("Sie sagte, es sei gut", ", aber niemand hörte zu.", None),
        ("kleingeschrieben weiter", "", None),
        ("", "", None),
    ],
)
def test_ueberschrift_erkennen(zeile: str, folgt: str, erwartet: str | None) -> None:
    assert heading_text(zeile, folgt) == erwartet


def test_ueberschrift_bekommt_einen_punkt_in_der_sprechfassung() -> None:
    """Der Kern des Ganzen: ohne Satzzeichen setzt die Engine nicht ab und
    hetzt die Zeile herunter. Im Rohtext steht der Punkt nicht."""
    titel = split_sentences("Kapitel 3\n\nEs war einmal.")[0]

    assert titel.is_heading
    assert titel.raw == "Kapitel 3"
    assert titel.normalized == "Kapitel drei."


def test_ueberschrift_ohne_leerzeile_wird_nicht_eingeschmolzen() -> None:
    """Der Normalfall in echten Texten: der Titel steht direkt über dem Absatz.
    Zusammengezogen läse die Engine ihn in den ersten Satz hinein."""
    saetze = split_sentences("Der lange Weg\nEs war einmal ein Text.")

    assert [s.normalized for s in saetze] == ["Der lange Weg.", "Es war einmal ein Text."]
    assert [s.is_heading for s in saetze] == [True, False]


def test_hart_umbrochener_absatz_ist_keine_ueberschrift() -> None:
    """Die Falle: ein auf 72 Zeichen umbrochener Text besteht aus lauter Zeilen
    ohne Satzzeichen. Als Überschriften gelesen zerfiele er in Bruchstücke."""
    text = "Er ging über die Straße und\ndachte an nichts Bestimmtes.\nDann kam der Regen."
    saetze = split_sentences(text)

    assert not any(s.is_heading for s in saetze)


def test_ueberschrift_bleibt_ein_eigener_chunk() -> None:
    text = "Kapitel 3\n\nEin kurzer Satz. Noch einer."
    chunks = build_chunks(text, chars_per_second=14.0, target_seconds=30.0)

    assert chunks[0].is_heading
    assert chunks[0].normalized_text == "Kapitel drei."
    assert not chunks[1].is_heading
