"""Eingangsprüfung der Referenzaufnahme."""

from __future__ import annotations

import numpy as np
import pytest

from cloney.core.voices import inspect_reference

SR = 24000


def _speech(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (amplitude * envelope * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def test_passendes_transkript_ohne_beanstandung() -> None:
    # 10 Sekunden, rund 140 Zeichen -- deutsches Sprechtempo.
    check = inspect_reference(_speech(10.0), SR, transcript="x" * 140)
    assert check.chars_per_second == pytest.approx(14.0)
    assert not any("Referenztext" in w for w in check.warnings)


def test_beschriftung_statt_wortlaut_wird_bemaengelt() -> None:
    """Der häufigste Grund für unverständliche Ausgabe: statt des Wortlauts
    steht eine kurze Beschriftung im Feld. F5-TTS hält den Sprecher dann für
    sehr langsam und presst den neuen Text in zu wenig Zeit."""
    check = inspect_reference(_speech(10.0), SR, transcript="meine Stimme")
    assert any("Beschriftung" in w for w in check.warnings)
    assert check.chars_per_second is not None
    assert check.chars_per_second < 7.0


def test_zu_langes_transkript_wird_bemaengelt() -> None:
    check = inspect_reference(_speech(5.0), SR, transcript="x" * 400)
    assert any("sehr lang" in w for w in check.warnings)


def test_ohne_transkript_keine_temporuege() -> None:
    check = inspect_reference(_speech(10.0), SR)
    assert check.chars_per_second is None
    assert not any("Referenztext" in w for w in check.warnings)


def test_uebersteuerte_aufnahme_wird_bemaengelt() -> None:
    check = inspect_reference(_speech(8.0, amplitude=0.999), SR)
    assert any("übersteuert" in w for w in check.warnings)


def test_zu_kurze_aufnahme_wird_bemaengelt() -> None:
    assert any("instabil" in w for w in inspect_reference(_speech(2.0), SR).warnings)


# -- Sprechtempo und Reglervorschlag ---------------------------------------


def test_tempo_wird_auf_dem_sprachanteil_gerechnet() -> None:
    """F5-TTS schneidet lange Stille aus der Referenz, bevor es das Tempo
    ableitet. Auf der Dateilänge gerechnet wirkte eine Aufnahme mit Vorlauf
    langsamer als sie ist -- und der Vorschlag daneben."""
    import numpy as np

    from cloney.core.audio import silence

    sprache = _speech(6.0)
    mit_vorlauf = np.concatenate([silence(4.0, SR), sprache, silence(4.0, SR)])

    ohne = inspect_reference(sprache, SR, transcript="x" * 100)
    mit = inspect_reference(mit_vorlauf, SR, transcript="x" * 100)

    assert mit.duration_s > ohne.duration_s + 7
    # Trotz doppelter Dateilänge nahezu dasselbe Tempo.
    assert mit.chars_per_second == pytest.approx(ohne.chars_per_second, rel=0.15)


@pytest.mark.parametrize(
    ("rate", "erwartet"),
    [(17.0, 0.85), (20.0, 0.72), (11.0, 1.32)],
)
def test_vorschlag_bringt_auf_angenehmes_tempo(rate: float, erwartet: float) -> None:
    from cloney.core.voices import suggested_speed

    assert suggested_speed(rate) == pytest.approx(erwartet, abs=0.01)


@pytest.mark.parametrize("rate", [14.0, 14.5, 15.0])
def test_kein_vorschlag_wenn_das_tempo_schon_passt(rate: float) -> None:
    """Ein Regler, der nichts ändert, ist nur Zierde."""
    from cloney.core.voices import suggested_speed

    assert suggested_speed(rate) is None


def test_vorschlag_bleibt_in_den_grenzen_des_reglers() -> None:
    from cloney.core.voices import suggested_speed

    assert suggested_speed(60.0) == 0.5
    assert suggested_speed(2.0) == 1.5
    assert suggested_speed(None) is None
    assert suggested_speed(0.0) is None


def test_zuegiges_sprechen_ist_keine_beanstandung() -> None:
    """17 Zeichen/s ist Podcast-Tempo -- zügig, aber normal. Wer das bemängelt,
    lenkt von echten Fehlern ab."""
    check = inspect_reference(_speech(10.0), SR, transcript="x" * 170)
    assert check.chars_per_second > 16
    assert not any("Referenztext" in w for w in check.warnings)
