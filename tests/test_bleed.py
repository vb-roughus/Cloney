"""Der Referenz-Vorspann: ihn finden, und ihn an der richtigen Stelle abtrennen.

F5-TTS erzeugt Referenz und neuen Text in einem Stück und trennt sie an einer
aus der Länge der Aufnahme berechneten Stelle. Passt die Aufnahme nicht genau zu
ihrem Wortlaut, dehnt das Modell den Referenzteil, und ein Rest landet hinter
der Schnittstelle -- am Anfang des Ergebnisses zu hören.

Zwei Fälle sind zu unterscheiden, und lange war nur der erste behandelt:

1. Die Erkennung schreibt den Vorspann als Wörter auf. Dann steht in der
   Rückschrift vor dem gewünschten Text etwas, das nicht dazugehört.
2. Sie schreibt ihn **nicht** auf -- eine angerissene Silbe ist kein Wort. Dann
   passt die Rückschrift Wort für Wort, und trotzdem ist der Vorspann da.
   Verraten wird er allein dadurch, dass das erste Wort erst spät beginnt.

Und wo genau geschnitten wird, ist eine eigene Frage: Whispers Wortzeiten sind
geschätzt, und ein paar Hundertstel zu spät kosten dem ersten Wort seinen
Anlaut.
"""

from __future__ import annotations

import numpy as np

from cloney.asr.base import TranscribedWord
from cloney.core.bleed import cut_point, find_content_start

RATE = 24000


def _worte(*paare: tuple[str, float]) -> tuple[TranscribedWord, ...]:
    """Wörter mit Startzeit; das Ende interessiert hier nicht."""
    return tuple(TranscribedWord(text, start, start + 0.2) for text, start in paare)


# -- Wo der gewünschte Text beginnt ----------------------------------------


def test_aufgeschriebener_vorspann_wird_gefunden() -> None:
    worte = _worte(("Rest", 0.0), ("davor", 0.2), ("Hier", 0.5), ("beginnt", 0.8), ("es", 1.1))

    start, woerter = find_content_start(worte, "Hier beginnt es.")

    assert start == 0.5
    assert woerter == 2


def test_ein_spaeter_beginn_verraet_den_stummen_vorspann() -> None:
    """Der eigentliche Fall. Die Rückschrift passt Wort für Wort -- und trotzdem
    steht eine halbe Sekunde Referenz davor, die kein Wort ergab."""
    worte = _worte(("Hier", 0.5), ("beginnt", 0.8), ("es", 1.1))

    start, woerter = find_content_start(worte, "Hier beginnt es.")

    assert start == 0.5
    # Kein Wort gehört zum Vorspann: die Rückschrift bleibt vollständig.
    assert woerter == 0


def test_ohne_vorspann_beginnt_es_bei_null() -> None:
    """Die Schwelle zieht der Aufrufer -- hier zählt nur die Zahl."""
    worte = _worte(("Hier", 0.0), ("beginnt", 0.3), ("es", 0.6))

    assert find_content_start(worte, "Hier beginnt es.") == (0.0, 0)


def test_ein_einzelnes_wort_genuegt_als_beleg_nicht() -> None:
    """'der' kommt im Vorspann so gut vor wie im Satz. Ohne drei Wörter in Folge
    wäre der Schnitt ein Ratespiel."""
    worte = _worte(("Der", 0.0), ("Rest", 0.2), ("Der", 0.5), ("Hund", 0.8), ("bellt", 1.1))

    start, _ = find_content_start(worte, "Der Kater schläft.")

    assert start is None


def test_ohne_rueckschrift_wird_nicht_geraten() -> None:
    assert find_content_start((), "Ein Satz.") == (None, 0)
    assert find_content_start(_worte(("Hier", 0.0)), "") == (None, 0)


# -- Wo genau geschnitten wird ---------------------------------------------


def _tonspur(*abschnitte: tuple[float, float]) -> np.ndarray:
    """Abschnitte aus (Sekunden, Amplitude) hintereinander."""
    teile = []
    for dauer, pegel in abschnitte:
        n = int(dauer * RATE)
        t = np.arange(n, dtype=np.float32) / RATE
        teile.append((pegel * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
    return np.concatenate(teile)


def test_der_schnitt_faellt_in_die_pause_dazwischen() -> None:
    """0,3 s Vorspann, 0,1 s Ruhe, dann der Satz. Whisper meldet den Beginn
    fünf Hundertstel zu spät -- geschnitten wird trotzdem in der Ruhe."""
    audio = _tonspur((0.30, 0.4), (0.10, 0.0), (0.60, 0.4))

    schnitt = cut_point(audio, RATE, 0.45)

    assert 0.30 <= schnitt <= 0.40


def test_eine_senke_im_vorspann_zieht_den_schnitt_nicht_an() -> None:
    """Der gemessene Fehler. Sprache hat Senken: eine Verschlusslaut-Pause
    mitten im Vorspann ist leiser als die Lücke danach. Wer die leiseste Stelle
    nimmt, trifft sie -- und schneidet weit vor der Grenze. Zu hören war das als
    eine Silbe, die blieb, nur kürzer.

    Gesucht wird deshalb die *letzte* ruhige Stelle, nicht die leiseste.
    """
    audio = _tonspur(
        (0.30, 0.4),  # Vorspann
        (0.06, 0.0),  # Senke darin -- vollkommen still
        (0.10, 0.4),  # Vorspann geht weiter
        (0.08, 0.02),  # die eigentliche Grenze: leise, aber nicht still
        (0.60, 0.4),  # der Satz
    )

    schnitt = cut_point(audio, RATE, 0.56)

    assert 0.46 <= schnitt <= 0.55


def test_ohne_ruhe_bleibt_es_beim_kandidaten() -> None:
    """Läuft der Vorspann ohne Absetzen in den Satz, gibt es keine Grenze zu
    finden. Eine Senke zu suchen, die es nicht gibt, hieße raten."""
    audio = _tonspur((0.40, 0.4), (0.05, 0.1), (0.40, 0.4))

    assert cut_point(audio, RATE, 0.42) == 0.42


def test_das_fenster_reicht_nur_so_weit_wie_die_ungenauigkeit() -> None:
    """Es ist nach Whispers Zeiten bemessen, nicht nach der Länge des
    Vorspanns. Ein weites Fenster fände Ruhe mitten im Vorspann -- oder, bei
    einem zu spät gemeldeten Anfang, im Satz, und nähme dem ersten Wort seinen
    Anlaut."""
    audio = _tonspur((0.30, 0.4), (0.10, 0.0), (0.60, 0.4))

    # Um zwei Zehntel zu spät gemeldet: so weit greift die Berichtigung nicht
    # mehr zurück, und der Kandidat bleibt stehen.
    assert cut_point(audio, RATE, 0.60) == 0.60


def test_zu_kurzes_audio_bleibt_beim_kandidaten() -> None:
    assert cut_point(np.zeros(10, dtype=np.float32), RATE, 0.5) == 0.5
    assert cut_point(np.zeros(0, dtype=np.float32), RATE, 0.5) == 0.5
