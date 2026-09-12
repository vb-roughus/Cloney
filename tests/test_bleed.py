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
from cloney.core.bleed import (
    beurteile,
    cut_point,
    find_content_start,
    first_word,
    leading_fragment,
    pause_erwartet,
)

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


# -- Der Fetzen, den die Rückschrift nicht sieht ----------------------------


def _gemessener_satz(erster_teil: float = 0.10) -> np.ndarray:
    """Ein Satz, wie er aus einem echten Lauf kam.

    Die Zahlen stammen aus 'cloney vorspann' gegen eine Stimme, deren Referenz
    auf "Washington." endet: Vorlauf 0,10 s, Fetzen 0,14 s, Pause 0,36 s, dann
    der Satz. Genau an dieser Spur sind die Schwellen bemessen.
    """
    return _tonspur((erster_teil, 0.0), (0.14, 0.15), (0.36, 0.0), (2.0, 0.4))


def test_der_gemessene_fetzen_wird_gefunden() -> None:
    """Der Fall aus der Praxis. Drei Schwellen lagen daneben: der Fetzen beginnt
    nicht bei null (F5 gibt der Referenz 50 ms Stille mit), er ist länger als
    angenommen, und das Suchfenster endete genau dort, wo der Satz anfängt --
    dann sah es aus, als käme nach der Pause nichts mehr."""
    schnitt = leading_fragment(_gemessener_satz(), RATE, "Diese Zusammenfassung gilt.")

    assert schnitt is not None
    # Vorlauf, Fetzen und Pause zusammen: rund sechs Zehntel.
    assert 0.55 <= schnitt <= 0.61


def test_ohne_pause_kein_fetzen() -> None:
    """Satz 1 derselben Messung: Beginn 0,26 s, Dauer 0,13 s, Pause 0,01 s.
    Das ist der Satz selbst, und die fehlende Pause sagt es."""
    audio = _tonspur((0.26, 0.4), (0.13, 0.4), (0.01, 0.0), (2.0, 0.4))

    urteil = beurteile(audio, RATE, "Zusammenfassung der Lage.")

    assert urteil.schnitt is None


def test_ein_langer_anfang_ist_kein_fetzen_sondern_sprache() -> None:
    audio = _tonspur((0.30, 0.4), (0.30, 0.0), (1.0, 0.4))

    urteil = beurteile(audio, RATE, "Erster Satz hier.")

    assert urteil.schnitt is None
    assert "Sprache" in urteil.grund


def test_faengt_es_erst_spaeter_an_ist_es_der_satz() -> None:
    """F5 trennt Referenz und Text an einer berechneten Stelle. Was übersteht,
    liegt am Anfang und nirgendwo sonst."""
    audio = _tonspur((0.30, 0.0), (0.10, 0.4), (0.40, 0.0), (1.0, 0.4))

    urteil = beurteile(audio, RATE, "Erster Satz hier.")

    assert urteil.schnitt is None
    assert "beginnt erst" in urteil.grund


def test_eine_kommapause_ist_keine_satzpause() -> None:
    """Ein Verschlusslaut ist drei bis acht Hundertstel still, eine Kommapause
    anderthalb bis zwei Zehntel. Die Pause hinter einem Fetzen ist ein Satzende
    und liegt darüber."""
    audio = _tonspur((0.05, 0.0), (0.12, 0.3), (0.18, 0.0), (1.0, 0.4))

    urteil = beurteile(audio, RATE, "Nun geht es los.")

    assert urteil.schnitt is None
    assert "Pause" in urteil.grund


def test_ein_satzzeichen_hinter_dem_ersten_wort_schuetzt_den_satz() -> None:
    """'Ja,' sieht aus wie ein Fetzen mit Pause dahinter -- und ist keiner.
    Dieselbe Tonspur, zwei Texte, zwei Antworten: das ist der Punkt."""
    audio = _gemessener_satz()

    assert beurteile(audio, RATE, "Ja, so war es.").schnitt is None
    assert beurteile(audio, RATE, "Sie kam trotzdem.").schnitt is not None


def test_das_laengenmass_haette_einen_echten_vorspann_verworfen() -> None:
    """Die frühere Regel rechnete den Fetzen gegen die Länge des ersten Wortes.
    In der Messung stand vor "sie" ein Fetzen von vierzehn Hundertstel -- länger
    als die halbe erwartete Dauer von drei Zeichen, und damit verworfen. Er war
    trotzdem einer."""
    assert leading_fragment(_gemessener_satz(), RATE, "Sie kam trotzdem.") is not None


def test_ohne_sprache_dahinter_wird_nicht_geschnitten() -> None:
    """Kommt hinter der Pause nichts mehr, war der 'Fetzen' vielleicht alles,
    was der Satz hat."""
    audio = _tonspur((0.04, 0.15), (1.0, 0.0))

    urteil = beurteile(audio, RATE, "Kurz.")

    assert urteil.schnitt is None
    assert "kommt nichts mehr" in urteil.grund


def test_ohne_ton_gibt_es_nichts_zu_beurteilen() -> None:
    urteil = beurteile(np.zeros(RATE, dtype=np.float32), RATE, "Ein Satz.")

    assert urteil.schnitt is None
    assert urteil.befund is None


def test_pause_erwartet_liest_das_satzzeichen() -> None:
    assert pause_erwartet("Ja, so war es.")
    assert pause_erwartet("Nun: es geht los.")
    assert pause_erwartet('"Ja," sagte er.')
    assert not pause_erwartet("Sie kam trotzdem.")
    assert not pause_erwartet("")


def test_first_word_nimmt_den_wortlaut_ohne_satzzeichen() -> None:
    assert first_word("Der erste Satz.") == "der"
    assert first_word("") == ""


# -- Der Befehl, der die Zahlen zeigt ---------------------------------------


def test_vorspann_zeigt_die_zahlen(settings, voice_store, monkeypatch) -> None:  # noqa: ANN001
    """Der Befehl ist da, damit die nächste Runde auf Zahlen beruht und nicht
    auf Adjektiven: 'Dauer' sagt, wie lang der Fetzen ist, 'Pause', wie deutlich
    er absetzt. Ohne GPU, ohne Modell, ohne Netz."""
    from typer.testing import CliRunner

    from cloney.asr.dummy import DummyASR
    from cloney.cli import app
    from cloney.core.project import Project
    from cloney.engines.dummy import DummyEngine
    from cloney.pipeline import run_project

    project = Project.create(
        name="Kapitel",
        text="Erster Satz hier. Zweiter Satz da.",
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
        target_seconds=1.5,
    )
    run_project(project, settings, voice_store, DummyEngine, DummyASR)
    monkeypatch.setattr("cloney.config._settings", settings)

    ergebnis = CliRunner().invoke(app, ["vorspann", project.id])

    assert ergebnis.exit_code == 0, ergebnis.output
    assert "Beginn" in ergebnis.output
    assert "Dauer" in ergebnis.output
    # Das Testsignal trägt keinen Fetzen: es geht unmittelbar in Sprache über.
    assert "ja " not in ergebnis.output


def test_vorspann_ohne_projekt_sagt_es(settings, monkeypatch) -> None:  # noqa: ANN001
    from typer.testing import CliRunner

    from cloney.cli import app

    monkeypatch.setattr("cloney.config._settings", settings)
    ergebnis = CliRunner().invoke(app, ["vorspann", "gibt-es-nicht"])

    assert ergebnis.exit_code == 1
    assert "gibt es nicht" in ergebnis.output
