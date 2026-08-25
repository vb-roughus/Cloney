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
