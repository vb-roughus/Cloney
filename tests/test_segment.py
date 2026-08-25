from __future__ import annotations

from cloney.core.segment import build_chunks, split_sentences


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
