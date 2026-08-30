"""Trainingsmaterial aus langen Aufnahmen -- ohne GPU und ohne echte Erkennung.

Der Datensatz entscheidet über das Finetune. Was hier festgehalten wird, sind
deshalb keine Formalien, sondern die drei Regeln, an denen ein Datensatz
scheitert: an Pausen schneiden, den Text in Sprechfassung ablegen, und
Verworfenes benennen statt verschwinden lassen.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from cloney.asr.base import Transcript
from cloney.config import Settings
from cloney.core.audio import write_wav
from cloney.core.dataset import Dataset, build_dataset, find_segments

SR = 24000


def _sprache(sekunden: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(sekunden * SR), dtype=np.float32) / SR
    huelle = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (amplitude * huelle * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def _stille(sekunden: float) -> np.ndarray:
    return np.zeros(int(sekunden * SR), dtype=np.float32)


def _aufnahme(*teile: np.ndarray) -> np.ndarray:
    return np.concatenate(teile).astype(np.float32)


class FesterASR:
    """Liefert immer denselben Wortlaut. Die Kennung von DummyASR übersteht das
    Zerschneiden nicht -- für den Datensatz braucht es einen eigenen Ersatz."""

    def __init__(self, text: str = "Am 3. Mai 2024 begann alles ganz harmlos hier.") -> None:
        self.text = text
        self.aufrufe = 0

    def transcribe(self, audio, sample_rate, language="de") -> Transcript:  # noqa: ANN001
        self.aufrufe += 1
        return Transcript(self.text)

    def close(self) -> None:
        return None


# -- Segmentierung ----------------------------------------------------------


def test_geschnitten_wird_an_den_pausen() -> None:
    audio = _aufnahme(_sprache(5.0), _stille(0.8), _sprache(6.0), _stille(0.8), _sprache(4.0))
    gut, _ = find_segments(audio, SR)

    assert len(gut) == 3
    laengen = [(b - a) / SR for a, b in gut]
    assert laengen == pytest.approx([5.0, 6.0, 4.0], abs=0.3)


def test_zu_kurze_abschnitte_fallen_mit_grund_weg() -> None:
    """Allein stehend, mit weiten Lücken ringsum: hier hilft kein Nachbar."""
    audio = _aufnahme(_sprache(5.0), _stille(2.5), _sprache(1.2), _stille(2.5), _sprache(5.0))
    gut, schlecht = find_segments(audio, SR)

    assert len(gut) == 2
    assert len(schlecht) == 1
    assert schlecht[0][2].startswith("zu kurz -- ")
    # Der Abstand steht dabei: er sagt, ob es an der Länge lag oder an der Lücke.
    assert "Nachbar 2.5s entfernt" in schlecht[0][2]


def test_kurzer_abschnitt_wird_mit_seinem_nachbarn_zusammengefasst() -> None:
    """Wer selbst geschnitten hat, hat kurze Abschnitte -- ein Halbsatz, ein
    Name. Einzeln fallen sie durch die Mindestlänge; zusammen mit dem Nachbarn
    sind sie ein gewöhnliches Segment, und die Pause dazwischen gehört dazu."""
    audio = _aufnahme(_sprache(1.2), _stille(0.4), _sprache(5.0))
    gut, schlecht = find_segments(audio, SR)

    assert schlecht == []
    assert len(gut) == 1
    assert (gut[0][1] - gut[0][0]) / SR == pytest.approx(6.6, abs=0.3)


def test_mehrere_kurze_ergeben_zusammen_ein_segment() -> None:
    audio = _aufnahme(_sprache(1.5), _stille(0.4), _sprache(1.5), _stille(0.4), _sprache(1.5))
    gut, schlecht = find_segments(audio, SR)

    assert schlecht == []
    assert len(gut) == 1


def test_zusammenfassen_endet_an_der_hoechstlaenge() -> None:
    """Sonst wüchse ein Segment über die Länge hinaus, mit der trainiert wird."""
    audio = _aufnahme(_sprache(2.0), _stille(0.4), _sprache(9.0), _stille(0.4), _sprache(9.0))
    gut, _ = find_segments(audio, SR, max_seconds=15.0)

    assert len(gut) == 2
    assert all((b - a) / SR <= 15.0 for a, b in gut)


def test_zwei_lange_abschnitte_bleiben_getrennt() -> None:
    """Zusammengefasst wird nur, was allein durchfiele -- sonst entstünden aus
    zwei brauchbaren Beispielen eines, ohne dass jemand etwas gewönne."""
    audio = _aufnahme(_sprache(5.0), _stille(0.4), _sprache(6.0))
    gut, _ = find_segments(audio, SR)

    assert len(gut) == 2


def test_langer_abschnitt_wird_an_der_inneren_pause_geteilt() -> None:
    """Eine kurze Lücke trennt keine Äußerungen, taugt aber als Schnittstelle,
    wenn der Bereich sonst zu lang bliebe -- die Alternative wäre, achtzehn
    Sekunden brauchbare Sprache wegzuwerfen."""
    audio = _aufnahme(_sprache(9.0), _stille(0.25), _sprache(9.0))
    gut, schlecht = find_segments(audio, SR, max_seconds=15.0)

    assert schlecht == []
    assert len(gut) == 2
    assert all(3.0 <= (b - a) / SR <= 15.0 for a, b in gut)


def test_zu_langes_ohne_pause_wird_verworfen_statt_zerschnitten() -> None:
    """Ein harter Schnitt mitten im Wort brächte dem Modell einen Anfang bei,
    den es später produziert."""
    gut, schlecht = find_segments(_sprache(20.0), SR)

    assert gut == []
    assert schlecht[0][2].startswith("am Stück zu lang -- ")


def test_stille_aufnahme_ergibt_nichts() -> None:
    gut, schlecht = find_segments(_stille(10.0), SR)
    assert gut == [] and schlecht == []


# -- Bauen ------------------------------------------------------------------


def _quelle(tmp_path: Path, name: str = "lesung.wav") -> Path:
    pfad = tmp_path / name
    write_wav(pfad, _aufnahme(_sprache(6.0), _stille(0.8), _sprache(5.0)), SR)
    return pfad


def test_datensatz_liegt_im_f5_format(tmp_path: Path) -> None:
    dataset = build_dataset("Anna", [_quelle(tmp_path)], FesterASR(), tmp_path / "datasets")

    assert len(dataset.utterances) == 2
    for utterance in dataset.utterances:
        assert (dataset.root / utterance.file).exists()

    zeilen = list(csv.reader((dataset.root / "metadata.csv").open(encoding="utf-8"), delimiter="|"))
    assert zeilen[0][0] == "wavs/utt_00001.wav"
    assert zeilen[0][1] == dataset.utterances[0].text


def test_text_wird_wie_bei_der_synthese_normalisiert(tmp_path: Path) -> None:
    """Die Erkennung schreibt '3. Mai', gesprochen wurde 'dritten Mai'. Trainiert
    werden muss auf der Form, die später auch hineingeht."""
    dataset = build_dataset("Anna", [_quelle(tmp_path)], FesterASR(), tmp_path / "datasets")

    utterance = dataset.utterances[0]
    assert "dritten Mai zweitausendvierundzwanzig" in utterance.text
    # Der Wortlaut der Erkennung bleibt daneben stehen, nachvollziehbar.
    assert "3. Mai 2024" in utterance.raw_text


def test_unpassende_rueckschrift_wird_verworfen(tmp_path: Path) -> None:
    """Zu viel Text für die Dauer heißt fast immer: falsch transkribiert. Solche
    Paare bringen dem Modell falsche Längen bei."""
    lang = FesterASR("wort " * 200)
    dataset = build_dataset("Anna", [_quelle(tmp_path)], lang, tmp_path / "datasets")

    assert dataset.utterances == []
    assert all("Zeichen/s" in r.reason for r in dataset.rejected)


def test_leere_rueckschrift_wird_verworfen(tmp_path: Path) -> None:
    dataset = build_dataset("Anna", [_quelle(tmp_path)], FesterASR(""), tmp_path / "datasets")
    assert dataset.utterances == []
    assert "keine Rückschrift" in dataset.rejected[0].reason


def test_uebersteuertes_segment_wird_verworfen(tmp_path: Path) -> None:
    pfad = tmp_path / "laut.wav"
    write_wav(pfad, _aufnahme(_sprache(6.0, amplitude=1.0), _stille(0.8), _sprache(5.0)), SR)

    dataset = build_dataset("Anna", [pfad], FesterASR(), tmp_path / "datasets")

    assert len(dataset.utterances) == 1
    assert any("übersteuert" in r.reason for r in dataset.rejected)


def test_verworfenes_steht_mit_grund_im_manifest(tmp_path: Path) -> None:
    """Ein Datensatz, der stillschweigend die Hälfte wegwirft, ist nicht von
    einem zu unterscheiden, bei dem die Aufnahme schlecht war."""
    pfad = tmp_path / "kurz.wav"
    write_wav(pfad, _aufnahme(_sprache(6.0), _stille(0.8), _sprache(1.0)), SR)

    dataset = build_dataset("Anna", [pfad], FesterASR(), tmp_path / "datasets")
    geladen = Dataset.load(dataset.root)

    assert len(geladen.rejected) == 1
    assert geladen.rejected[0].source == "kurz.wav"
    assert geladen.rejected[0].start_s > 6.0


def test_gemischte_abtastraten_werden_abgelehnt(tmp_path: Path) -> None:
    """Sonst trainierte das Modell auf zwei verschiedene Stimmen."""
    a = _quelle(tmp_path, "a.wav")
    b = tmp_path / "b.wav"
    write_wav(b, _aufnahme(_sprache(6.0), _stille(0.8), _sprache(5.0)), 44100)

    with pytest.raises(ValueError, match="verschiedene Abtastraten"):
        build_dataset("Anna", [a, b], FesterASR(), tmp_path / "datasets")


def test_statistik_nennt_umfang_und_ausschuss(tmp_path: Path) -> None:
    dataset = build_dataset("Anna", [_quelle(tmp_path)], FesterASR(), tmp_path / "datasets")
    werte = dataset.statistik()

    assert werte["segmente"] == 2
    assert werte["minuten"] == pytest.approx(11 / 60, abs=0.02)
    assert werte["median_zeichen_pro_s"] > 0


def test_metadata_uebersteht_sonderzeichen_im_text(tmp_path: Path) -> None:
    """Ein Anführungszeichen im Text darf die Datei nicht zerreißen."""
    asr = FesterASR('Er sagte: "Na sowas!" und ging dann weiter fort.')
    dataset = build_dataset("Anna", [_quelle(tmp_path)], asr, tmp_path / "datasets")

    zeilen = list(csv.reader((dataset.root / "metadata.csv").open(encoding="utf-8"), delimiter="|"))
    assert len(zeilen) == len(dataset.utterances)
    assert zeilen[0][1] == dataset.utterances[0].text


def test_pfadangaben_im_namen_werden_entschaerft(tmp_path: Path) -> None:
    """Der Slug ersetzt alles außer Buchstaben und Ziffern; resolve ist die
    zweite Linie, falls sich daran je etwas ändert."""
    datasets = tmp_path / "datasets"
    datasets.mkdir()

    ziel = Dataset.resolve(datasets, "../../etc")
    assert ziel.parent == datasets.resolve()
    assert ziel.name == "etc"


# -- Die Schwelle kommt aus der Aufnahme, nicht aus einer Annahme -----------


def _mit_raumton(rausch_db: float, pausen_s: float = 0.5, stuecke: int = 4) -> np.ndarray:
    """Aufnahme mit Grundrauschen über der ganzen Länge -- so klingt ein Zimmer."""
    rng = np.random.default_rng(11)
    teile = []
    for _ in range(stuecke):
        teile += [_sprache(5.0), _stille(pausen_s)]
    rein = np.concatenate(teile)
    rauschen = 10 ** (rausch_db / 20) * rng.standard_normal(len(rein))
    return (rein + rauschen).astype(np.float32)


@pytest.mark.parametrize("rausch_db", [-70.0, -50.0, -38.0, -30.0])
def test_pausen_werden_auch_bei_lautem_raumton_gefunden(rausch_db: float) -> None:
    """Eine feste Schwelle von -40 dBFS findet ab -38 dBFS Raumton *keine*
    einzige Pause mehr -- die ganze Lesung gilt dann als ein Bereich ohne
    Schnittstelle. Genau daran ist der erste echte Datensatz gescheitert."""
    gut, schlecht = find_segments(_mit_raumton(rausch_db), SR)

    assert len(gut) == 4
    assert schlecht == []


def test_feste_schwelle_bleibt_uebersteuerbar() -> None:
    """Der automatische Weg ist der Normalfall, nicht der einzige."""
    audio = _mit_raumton(-30.0)
    assert find_segments(audio, SR, silence_db=-40.0)[0] == []
    assert len(find_segments(audio, SR)[0]) == 4


def test_aufnahme_ohne_dynamik_wird_benannt() -> None:
    """Durchgehend gesprochen oder stark komprimiert: ein Schnitt wäre geraten."""
    rng = np.random.default_rng(3)
    audio = (0.2 * rng.standard_normal(20 * SR)).astype(np.float32)

    gut, schlecht = find_segments(audio, SR)

    assert gut == []
    assert schlecht[0][2].startswith("keine Pausen erkennbar -- ")


def test_pegel_werden_aus_der_aufnahme_gemessen() -> None:
    from cloney.core.dataset import silence_levels

    rahmen = _mit_raumton(-45.0)[: 240 * 2000].reshape(-1, 240)
    rms = np.sqrt(np.mean(rahmen.astype(np.float64) ** 2, axis=1))
    pegel = 20 * np.log10(np.maximum(rms, 1e-12))

    grund, sprech = silence_levels(pegel)
    assert grund == pytest.approx(-45.0, abs=3.0)
    assert sprech > grund + 20


def test_ohne_pause_nennt_den_weg(tmp_path: Path) -> None:
    """Die Meldung soll sagen, was beim nächsten Take anders zu machen ist."""
    _, schlecht = find_segments(_sprache(20.0), SR)
    assert "deutlicher absetzen" in schlecht[0][2]


# -- Das Sprechtempo meint den Wortlaut, nicht die Sprechfassung ------------


def test_tempo_wird_auf_dem_wortlaut_gerechnet(tmp_path: Path) -> None:
    """Die Normalisierung bläht den Text auf: aus '3. Mai 2024' werden
    36 Zeichen. Auf der Sprechfassung gerechnet sähe jede Aufnahme mit Ziffern
    zu schnell aus -- und die Zahl wäre eine andere als die, mit der die
    Eingangsprüfung arbeitet."""
    dataset = build_dataset("Anna", [_quelle(tmp_path)], FesterASR(), tmp_path / "datasets")

    utterance = dataset.utterances[0]
    assert len(utterance.text) > len(utterance.raw_text)
    assert utterance.chars_per_second == pytest.approx(
        len(utterance.raw_text) / utterance.duration_s
    )


# -- Geschnittene Aufnahmen: exakte Null ist kein Raumton -------------------


def _mit_vorlauf(rausch_db: float | None, vorlauf_s: float = 0.5) -> np.ndarray:
    """Wie ein Schnittprogramm sie hinterlässt: harte Null am Anfang, dann die
    Lesung -- mit oder ohne Raumton in den Pausen."""
    rng = np.random.default_rng(2)
    kern = np.concatenate([_sprache(7.0), _stille(0.45), _sprache(5.5), _stille(0.45)])
    if rausch_db is not None:
        kern = (kern + 10 ** (rausch_db / 20) * rng.standard_normal(len(kern))).astype(np.float32)
    return np.concatenate([_stille(vorlauf_s), kern]).astype(np.float32)


@pytest.mark.parametrize("rausch_db", [-70.0, -45.0, -34.0, None])
def test_harte_null_am_dateianfang_verdirbt_die_schaetzung_nicht(rausch_db: float | None) -> None:
    """Eine halbe Sekunde exakte Null zog den Grundpegel auf -240 dBFS und die
    Schwelle auf -230 -- damit galt jedes Rauschen als Sprache, und die ganze
    Lesung wurde ein Block ohne Pause. Genau das kam beim zweiten Anlauf des
    ersten echten Datensatzes heraus."""
    gut, schlecht = find_segments(_mit_vorlauf(rausch_db), SR)

    assert len(gut) == 2
    assert schlecht == []


def test_grundpegel_ignoriert_exakt_stille_frames() -> None:
    """Sonst misst man das Schnittprogramm statt des Zimmers."""
    from cloney.core.dataset import silence_levels

    audio = _mit_vorlauf(-45.0)
    rahmen = audio[: len(audio) - len(audio) % 240].reshape(-1, 240)
    rms = np.sqrt(np.mean(rahmen.astype(np.float64) ** 2, axis=1))
    pegel = 20 * np.log10(np.maximum(rms, 1e-12))

    grund, _ = silence_levels(pegel)
    assert grund == pytest.approx(-45.0, abs=3.0)


def test_meldung_nennt_geschnittene_stille_statt_einer_sinnlosen_zahl() -> None:
    """'Raumton -240 dBFS' ist keine Aussage über die Aufnahme, sondern über
    die Rechengenauigkeit."""
    from cloney.core.dataset import LevelReport

    assert LevelReport(-21.4, -14.0, digital_silence=True).beschreibung() == "geschnittene Stille"
    assert LevelReport(-45.0, -14.0).beschreibung() == "Raumton -45 dBFS"
    assert "geschnittene Stille" in LevelReport(-45.0, -14.0, True).beschreibung()


# -- Nachsehen statt raten --------------------------------------------------


def _durchgehend(spitze_db: float = -20.0, rausch_db: float = -63.0) -> np.ndarray:
    """Durchgehend gesprochen: die Hüllkurve sinkt, aber nie lange genug tief genug."""
    rng = np.random.default_rng(21)
    t = np.arange(int(18.0 * SR), dtype=np.float32) / SR
    silben = 0.30 + 0.70 * (0.5 + 0.5 * np.sin(2 * np.pi * 4.4 * t)) ** 2
    g = np.sin(2 * np.pi * 135 * t) + 0.5 * np.sin(2 * np.pi * 270 * t)
    roh = (silben * g).astype(np.float32)
    rede = roh / np.max(np.abs(roh)) * 10 ** (spitze_db / 20)
    kern = np.concatenate([_stille(0.6), rede.astype(np.float32), _stille(0.5)])
    return (kern + 10 ** (rausch_db / 20) * rng.standard_normal(len(kern))).astype(np.float32)


def test_probe_misst_die_pegel_der_aufnahme() -> None:
    from cloney.core.dataset import probe_audio

    befund = probe_audio(_mit_raumton(-45.0), SR)

    assert befund.levels.floor_db == pytest.approx(-45.0, abs=3.0)
    assert befund.levels.speech_db > befund.levels.floor_db + 20
    assert befund.duration_s == pytest.approx(22.0, abs=0.5)


def test_probe_erkennt_eine_aufnahme_mit_pausen() -> None:
    from cloney.core.dataset import probe_audio

    befund = probe_audio(_mit_raumton(-45.0), SR)

    assert not befund.hoffnungslos
    assert befund.beste_schwelle() is not None


def test_probe_erkennt_durchgehendes_sprechen() -> None:
    """Der Fall, der von außen wie ein Schwellenproblem aussieht: keine
    Einstellung findet Pausen, weil keine da sind."""
    from cloney.core.dataset import probe_audio

    befund = probe_audio(_durchgehend(), SR)

    assert befund.hoffnungslos
    assert befund.beste_schwelle() is None


def test_schwelle_ueber_dem_sprechpegel_gilt_nicht_als_brauchbar() -> None:
    """Liegt sie darüber, ist die ganze Aufnahme 'still' und die Zahl der
    Pausen sagt nichts mehr aus -- so eine Zeile wäre sonst als Empfehlung
    durchgegangen."""
    from cloney.core.dataset import ProbeRow

    assert ProbeRow(-30.0, 3, 3, 0.45, silence_share=0.07).brauchbar
    assert not ProbeRow(-20.0, 1, 1, 19.4, silence_share=1.0).brauchbar
    assert not ProbeRow(-60.0, 0, 0, 0.0, silence_share=0.0).brauchbar


# -- Notausgang für durchgehend gesprochenes Material -----------------------


def test_force_split_rettet_material_ohne_pausen() -> None:
    """Ohne Notausgang geht ein durchgehend gesprochener Block vollständig
    verloren. Mit ihm wird an der leisesten Stelle getrennt -- kein guter
    Schnitt, aber besser als der ganze Bereich im Ausschuss."""
    audio = _durchgehend()

    ohne, verworfen = find_segments(audio, SR)
    mit, _ = find_segments(audio, SR, force_split=True)

    assert ohne == []
    assert "ohne Pause" in verworfen[0][2]
    assert len(mit) >= 2
    assert all(3.0 <= (b - a) / SR <= 15.0 for a, b in mit)


def test_force_split_bleibt_die_ausnahme() -> None:
    """Wo es echte Pausen gibt, ändert der Notausgang nichts."""
    audio = _mit_raumton(-45.0)
    assert find_segments(audio, SR)[0] == find_segments(audio, SR, force_split=True)[0]


def test_probe_zaehlt_nur_pausen_zwischen_der_sprache() -> None:
    """Vorlauf und Ausklang sind fast immer still, taugen aber nicht als
    Schnittstelle -- zu trennen ist ja das, was dazwischen liegt."""
    from cloney.core.dataset import probe_audio

    # Ruhiger Anfang und Schluss, dazwischen durchgehend gesprochen.
    befund = probe_audio(_durchgehend(), SR)

    assert befund.gefundene_pausen() == 0


def test_probe_misst_pausen_gegen_die_benoetigte_zahl() -> None:
    """Eine einzelne Pause auf vierzig Sekunden 'trennt Pausen' und ist trotzdem
    unbrauchbar. Die Bewertung muss das benennen."""
    from cloney.core.dataset import probe_audio

    lang = np.concatenate([_durchgehend(), _stille(0.8), _durchgehend()])
    befund = probe_audio(lang, SR)

    assert befund.benoetigte_pausen() >= 2
    assert befund.gefundene_pausen() == 1
    assert not befund.genug_pausen()


def test_probe_bestaetigt_ausreichendes_material() -> None:
    from cloney.core.dataset import probe_audio

    befund = probe_audio(_mit_raumton(-45.0), SR)
    assert befund.genug_pausen()


# -- Die Schwelle hängt auch am Sprechpegel ---------------------------------


def test_schwelle_beruecksichtigt_beide_anker() -> None:
    """Nur am Grundpegel festgemacht, greift die Schwelle zu tief, wenn die
    Aufnahme irgendwo sehr leise ist, die Sprechpausen aber viel höher liegen --
    weil in ihnen geatmet wird. Gemessen an einer echten Lesung: Grundpegel
    -63 dBFS, Sprechpegel -12, Pausen erst ab -50 aufwärts."""
    from cloney.core.dataset import threshold_for

    # Der Fall aus der Praxis: Sprech-25 gewinnt.
    assert threshold_for(-63.0, -12.0) == pytest.approx(-37.0)
    # Laute Umgebung: der Abstand über dem Grundpegel gewinnt.
    assert threshold_for(-34.0, -14.0) == pytest.approx(-24.0)
    # Nach oben und unten bleibt die Schwelle im Band.
    assert threshold_for(-10.0, -5.0) == pytest.approx(-20.0)
    assert threshold_for(-120.0, -100.0) == pytest.approx(-55.0)


def test_bewertung_zaehlt_die_verwendete_schwelle_nicht_die_beste() -> None:
    """Was eine andere Schwelle fände, hilft niemandem, solange sie nicht
    verwendet wird. Genau diese Verwechslung ließ eine unzerlegbare Aufnahme
    als in Ordnung durchgehen."""
    from cloney.core.dataset import LevelReport, Probe, ProbeRow

    befund = Probe(
        duration_s=42.0,
        sample_rate=SR,
        levels=LevelReport(-63.0, -12.0),
        digital_silence_share=0.12,
        threshold_db=-53.0,
        rows=[
            ProbeRow(-53.0, 1, 1, 2.12, silence_share=0.13),
            ProbeRow(-25.0, 11, 7, 2.20, silence_share=0.41),
        ],
    )

    assert befund.gefundene_pausen() == 1
    assert not befund.genug_pausen()
    assert befund.beste_zeile().pauses_split == 11


def test_erweitertes_material_ersetzt_den_datensatz(settings: Settings, tmp_path: Path) -> None:
    """Der Fall aus dem Betrieb: eine Aufnahme wird verlängert und erneut
    eingelesen. Derselbe Name heißt neu bauen -- die Segmente des vorherigen
    Laufs dürfen nicht als Karteileichen liegen bleiben."""
    quelle = tmp_path / "lesung.wav"
    write_wav(quelle, _aufnahme(*[_sprache(5.0), _stille(0.9)] * 5), SR)
    erst = build_dataset("anna", [quelle], FesterASR(), settings.datasets_dir)
    vorher = len(erst.utterances)
    assert vorher >= 4

    # Dieselbe Aufnahme, gekürzt: der zweite Lauf ergibt weniger Segmente.
    write_wav(quelle, _aufnahme(_sprache(5.0), _stille(0.9), _sprache(5.0)), SR)
    zweit = build_dataset("anna", [quelle], FesterASR(), settings.datasets_dir)

    assert len(zweit.utterances) < vorher
    dateien = sorted(p.name for p in (zweit.root / "wavs").glob("utt_*.wav"))
    assert dateien == sorted(Path(u.file).name for u in zweit.utterances)
