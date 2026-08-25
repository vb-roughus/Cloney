"""Tabellentests für die deutsche Normalisierung."""

from __future__ import annotations

import pytest

from cloney.core.normalize import cardinal, normalize_german, ordinal, year


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Ordinalzahlen werden nach dem vorangehenden Wort dekliniert.
        ("am 3. Mai", "am dritten Mai"),
        ("der 3. Platz", "der dritte Platz"),
        ("die 21. Auflage", "die einundzwanzigste Auflage"),
        ("im 19. Jahrhundert", "im neunzehnten Jahrhundert"),
        # Ein Punkt nach einer Zahl ist mehrdeutig: hier ist es ein Satzende.
        ("Es waren 50. Danach kam mehr.", "Es waren fünfzig. Danach kam mehr."),
        # Datum und Uhrzeit
        ("am 03.05.2024", "am dritten Mai zweitausendvierundzwanzig"),
        ("um 14:30 Uhr", "um vierzehn Uhr dreißig"),
        ("um 9:00 Uhr", "um neun Uhr"),
        # Währung
        ("1.250,50 €", "eintausendzweihundertfünfzig Euro fünfzig"),
        ("1 €", "ein Euro"),
        ("2.500 CHF", "zweitausendfünfhundert Franken"),
        ("20,00 €", "zwanzig Euro"),
        # Einheiten, Symbole, Prozent
        ("42,195 km", "zweiundvierzig Komma eins neun fünf Kilometer"),
        ("21 °C", "einundzwanzig Grad Celsius"),
        ("50 %", "fünfzig Prozent"),
        ("3 h", "drei Stunden"),
        ("1 Minute später", "eine Minute später"),
        ("1 min", "eine Minute"),
        ("1 kg", "ein Kilogramm"),
        # Abkürzungen
        ("z.B. heute", "zum Beispiel heute"),
        ("Dr. Meier bzw. Prof. Schmidt", "Doktor Meier beziehungsweise Professor Schmidt"),
        ("Siehe S. 12, Kap. 4.", "Siehe Seite zwölf, Kapitel vier."),
        ("3 Mio. Euro", "drei Millionen Euro"),
        # Jahreszahlen nur mit Signalwort in Hunderter-Lesung
        ("Er wurde 1984 geboren", "Er wurde neunzehnhundertvierundachtzig geboren"),
        ("im Jahr 1848", "im Jahr achtzehnhundertachtundvierzig"),
        ("1500 Meter", "eintausendfünfhundert Meter"),
        # Lange Ziffernfolgen einzeln lesen
        ("0791234567", "null sieben neun eins zwei drei vier fünf sechs sieben"),
    ],
)
def test_normalize(source: str, expected: str) -> None:
    assert normalize_german(source) == expected


def test_satzendpunkt_bleibt_erhalten() -> None:
    """Die Einheitenregel darf den Satzpunkt nicht verschlucken -- sonst
    verliert die Segmentierung die Satzgrenze."""
    assert normalize_german("Er lief in ca. 3 h.").endswith("Stunden.")


def test_mehrstellige_zahl_wird_nicht_zerlegt() -> None:
    """Regression: fehlende Gruppierung im Zahlenmuster las 1500 als 150 + 0."""
    assert normalize_german("1500 Meter") == "eintausendfünfhundert Meter"
    assert normalize_german("2030") == "zweitausenddreißig"


@pytest.mark.parametrize(
    ("value", "oblique", "expected"),
    [
        (1, False, "erste"),
        (1, True, "ersten"),
        (3, True, "dritten"),
        (21, False, "einundzwanzigste"),
    ],
)
def test_ordinal_deklination(value: int, oblique: bool, expected: str) -> None:
    assert ordinal(value, oblique) == expected


def test_jahreszahl_hunderterlesung() -> None:
    assert year(1984) == "neunzehnhundertvierundachtzig"
    assert year(1900) == "neunzehnhundert"
    assert year(2024) == cardinal(2024) == "zweitausendvierundzwanzig"
