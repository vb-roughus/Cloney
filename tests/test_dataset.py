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
    audio = _aufnahme(_sprache(1.2), _stille(0.8), _sprache(5.0))
    gut, schlecht = find_segments(audio, SR)

    assert len(gut) == 1
    assert len(schlecht) == 1
    assert "kürzer als" in schlecht[0][2]


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
    assert "ohne Pause" in schlecht[0][2]


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
    assert "kaum Unterschied" in schlecht[0][2]


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
