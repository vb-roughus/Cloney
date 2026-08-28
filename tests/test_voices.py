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


# -- Abruptes Ende der Referenz --------------------------------------------


def test_abruptes_ende_wird_bemaengelt() -> None:
    """F5-TTS verspricht dem Modell am Referenzende eine Pause. Bricht die
    Aufnahme mitten im Klang ab, bleibt ein Stück davon am Anfang des
    erzeugten Tons stehen.

    Das Prüfsignal muss dafür wirklich auf vollem Pegel abbrechen -- die
    Hüllkurve von ``_speech`` allein endet je nach Länge zufällig leise.
    """
    import numpy as np

    from cloney.core.voices import ends_abruptly

    abrupt = np.concatenate([_speech(8.0), np.full(3000, 0.3, dtype=np.float32)])
    assert ends_abruptly(abrupt, SR)
    assert any("endet abrupt" in w for w in inspect_reference(abrupt, SR).warnings)


def test_ausklingende_aufnahme_ist_in_ordnung() -> None:
    import numpy as np

    from cloney.core.audio import silence
    from cloney.core.voices import ends_abruptly

    mit_ausklang = np.concatenate([_speech(8.0), silence(0.5, SR)])
    assert not ends_abruptly(mit_ausklang, SR)
    assert not any("endet abrupt" in w for w in inspect_reference(mit_ausklang, SR).warnings)


# -- Die Aufnahme bleibt, wie sie ist ---------------------------------------


def _stereo_24bit(tmp_path, sekunden: float = 8.0, rate: int = 48000):  # noqa: ANN001, ANN202
    """Eine Quelle, die nichts mit Cloneys Innenleben gemein hat: Stereo, 48 kHz,
    24 Bit -- also alles, was frühere Fassungen wegkonvertiert haben."""
    import soundfile as sf

    t = np.arange(int(sekunden * rate), dtype=np.float32) / rate
    huelle = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    links = 0.3 * huelle * np.sin(2 * np.pi * 180 * t)
    pfad = tmp_path / "quelle.wav"
    sf.write(pfad, np.stack([links, np.roll(links, 200) * 0.9], axis=1), rate, subtype="PCM_24")
    return pfad


def test_referenz_wird_bitgenau_abgelegt(tmp_path) -> None:  # noqa: ANN001
    """Vorher wurde die Aufnahme nach Mono gemischt und auf 16 Bit gebracht.
    Die Referenz im Speicher klang danach schlechter als das Original -- ohne
    dass irgendwer davon etwas gehabt hätte."""
    from cloney.core.voices import VoiceStore

    quelle = _stereo_24bit(tmp_path)
    store = VoiceStore(tmp_path / "voices")
    _, check = store.add("anna", quelle, transcript="Der Wortlaut dieser Aufnahme steht hier.")

    abgelegt = store.get("anna").audio_path
    assert abgelegt.read_bytes() == quelle.read_bytes()
    assert store.format("anna").channels == 2
    assert store.format("anna").subtype == "PCM_24"
    assert store.format("anna").sample_rate == 48000
    # Die Prüfung nennt, was tatsächlich liegt -- nicht, was sie gelesen hat.
    assert (check.channels, check.subtype) == (2, "PCM_24")


def test_endung_der_quelle_bleibt_erhalten(tmp_path) -> None:  # noqa: ANN001
    import soundfile as sf

    from cloney.core.voices import VoiceStore

    t = np.arange(8 * 44100, dtype=np.float32) / 44100
    sf.write(tmp_path / "quelle.flac", 0.3 * np.sin(2 * np.pi * 160 * t), 44100, subtype="PCM_24")

    store = VoiceStore(tmp_path / "voices")
    store.add("anna", tmp_path / "quelle.flac", transcript="Der Wortlaut dieser Aufnahme.")
    assert store.get("anna").audio_path.name == "reference.flac"


def test_ersetzen_mit_anderem_format_laesst_keine_leiche(tmp_path) -> None:  # noqa: ANN001
    """Sonst lägen zwei Aufnahmen im Ordner und die Auswahl entschiede das Alphabet."""
    import soundfile as sf

    from cloney.core.voices import VoiceStore

    t = np.arange(8 * 44100, dtype=np.float32) / 44100
    sf.write(tmp_path / "quelle.flac", 0.3 * np.sin(2 * np.pi * 160 * t), 44100)
    store = VoiceStore(tmp_path / "voices")
    store.add("anna", tmp_path / "quelle.flac", transcript="Der Wortlaut dieser Aufnahme.")
    store.add("anna", _stereo_24bit(tmp_path), transcript="Der Wortlaut dieser Aufnahme.")

    dateien = sorted(p.name for p in (tmp_path / "voices" / "anna").iterdir())
    assert dateien == ["reference.wav", "voice.json"]


def test_seltsame_endung_wird_zu_wav(tmp_path) -> None:  # noqa: ANN001
    """Der Dateiname kommt beim Hochladen vom Browser und damit vom Benutzer."""
    import shutil

    from cloney.core.voices import VoiceStore

    quelle = _stereo_24bit(tmp_path)
    boese = tmp_path / "aufnahme.wav.../../etc"
    boese.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(quelle, boese)

    store = VoiceStore(tmp_path / "voices")
    store.add("anna", boese, transcript="Der Wortlaut dieser Aufnahme.")
    assert store.get("anna").audio_path.name == "reference.wav"


def test_stimme_von_vor_der_formatfreiheit_bleibt_lesbar(tmp_path) -> None:  # noqa: ANN001
    """Ältere Stimmen tragen kein 'file' in den Metadaten und immer eine WAV."""
    import json

    from cloney.core.audio import write_wav
    from cloney.core.voices import VoiceStore

    ordner = tmp_path / "voices" / "alt"
    ordner.mkdir(parents=True)
    write_wav(ordner / "reference.wav", _speech(8.0), SR)
    (ordner / "voice.json").write_text(
        json.dumps({"name": "alt", "transcript": "Wortlaut.", "duration_s": 8.0}),
        encoding="utf-8",
    )

    stimme = VoiceStore(tmp_path / "voices").get("alt")
    assert stimme.audio_path.name == "reference.wav"
    assert stimme.audio_path.exists()
