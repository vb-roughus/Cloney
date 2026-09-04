"""Emotionslagen: mehrere Referenzaufnahmen je Stimme.

Eine Stimme ist nicht ein Klang, sondern eine Sprecherin in einer Haltung.
Dieselbe Person klingt ernst anders als beiläufig, und ein Kapitel braucht
beides. Cloney legt deshalb je Lage eine eigene Referenz ab; ein Satz wählt
eine davon, und gegen die wird er konditioniert.

Festgehalten wird hier vor allem, was beim Ändern leicht kaputtgeht: dass die
neutrale Lage die Hauptaufnahme bleibt, dass eine gelöschte Lage einen Lauf
nicht zum Absturz bringt, und dass der Wechsel einer Lage den Seed nicht
bewegt -- sonst ließe sich nie hören, was die Lage allein bewirkt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cloney.core.audio import write_wav
from cloney.core.project import ChunkStatus, Project
from cloney.core.voices import VoiceStore
from cloney.engines.base import NEUTRAL
from cloney.engines.dummy import DummyEngine
from cloney.pipeline import synthesize_chunks


@pytest.fixture
def zweite_aufnahme(tmp_path: Path) -> Path:
    """Sechs Sekunden, hörbar anders als die Fixture-Referenz."""
    sample_rate = 24000
    t = np.arange(6 * sample_rate, dtype=np.float32) / sample_rate
    audio = (0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)) * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32
    )
    pfad = tmp_path / "ernst.wav"
    write_wav(pfad, audio, sample_rate)
    return pfad


# -- Ablage ----------------------------------------------------------------


def test_eine_stimme_ohne_lagen_ist_eine_stimme_mit_neutral(voice_store: VoiceStore) -> None:
    """Bestandsstimmen sollen nichts nachtragen müssen."""
    assert voice_store.lagen("test-stimme") == [NEUTRAL]
    assert voice_store.get("test-stimme").lage == NEUTRAL


def test_lage_anlegen_und_wiederfinden(voice_store: VoiceStore, zweite_aufnahme: Path) -> None:
    voice_store.add_lage("test-stimme", "ernst", zweite_aufnahme, transcript="Ernst gesprochen.")

    assert voice_store.lagen("test-stimme") == [NEUTRAL, "ernst"]
    ernst = voice_store.get("test-stimme", "ernst")
    assert ernst.transcript == "Ernst gesprochen."
    assert ernst.lage == "ernst"
    assert ernst.audio_path.exists()


def test_die_lagen_teilen_sich_die_aufnahme_nicht(
    voice_store: VoiceStore, zweite_aufnahme: Path
) -> None:
    """Der eigentliche Punkt: es sind zwei Dateien, nicht zweimal dieselbe."""
    voice_store.add_lage("test-stimme", "ernst", zweite_aufnahme, transcript="Ernst gesprochen.")

    neutral = voice_store.get("test-stimme")
    ernst = voice_store.get("test-stimme", "ernst")
    assert neutral.audio_path != ernst.audio_path
    assert round(neutral.duration_s) == 8
    assert round(ernst.duration_s) == 6


def test_unbekannte_lage_faellt_auf_neutral_zurueck(voice_store: VoiceStore) -> None:
    """Der Name steht im Manifest und überlebt dort das Löschen der Lage. Ein
    Lauf soll dann mit der Hauptaufnahme weitergehen, nicht abbrechen."""
    zurueck = voice_store.get("test-stimme", "gibtesnicht")

    assert zurueck.lage == NEUTRAL
    assert zurueck.audio_path == voice_store.get("test-stimme").audio_path


def test_neutral_laesst_sich_nicht_als_lage_anlegen(
    voice_store: VoiceStore, zweite_aufnahme: Path
) -> None:
    """Sie ist die Hauptaufnahme. Zwei Wege zu derselben Datei wären zwei
    Wahrheiten."""
    with pytest.raises(ValueError, match="Hauptaufnahme"):
        voice_store.add_lage("test-stimme", "Neutral", zweite_aufnahme)


def test_neutral_laesst_sich_nicht_einzeln_loeschen(voice_store: VoiceStore) -> None:
    with pytest.raises(ValueError, match="Hauptaufnahme"):
        voice_store.delete_lage("test-stimme", NEUTRAL)


def test_lage_loeschen_nimmt_auch_die_datei_mit(
    voice_store: VoiceStore, zweite_aufnahme: Path
) -> None:
    voice_store.add_lage("test-stimme", "ernst", zweite_aufnahme, transcript="Ernst.")
    datei = voice_store.get("test-stimme", "ernst").audio_path

    voice_store.delete_lage("test-stimme", "ernst")

    assert voice_store.lagen("test-stimme") == [NEUTRAL]
    assert not datei.exists()


def test_wortlaut_einer_lage_laesst_sich_aendern(
    voice_store: VoiceStore, zweite_aufnahme: Path
) -> None:
    voice_store.add_lage("test-stimme", "ernst", zweite_aufnahme, transcript="Falscher Text.")

    voice_store.set_lage_transcript("test-stimme", "ernst", "Der richtige Wortlaut.")

    assert voice_store.get("test-stimme", "ernst").transcript == "Der richtige Wortlaut."
    assert voice_store.get("test-stimme").transcript == "Dies ist die Referenzaufnahme."


def test_die_laengste_lage_bestimmt_die_chunk_planung(
    voice_store: VoiceStore, zweite_aufnahme: Path, tmp_path: Path
) -> None:
    """Referenz und Erzeugtes teilen sich bei F5-TTS ein Zeitbudget. Gerechnet
    wird mit der längsten Lage: jeder Satz kann jede von ihnen wählen."""
    lang = tmp_path / "lang.wav"
    write_wav(lang, np.zeros(24000 * 11, dtype=np.float32), 24000)
    voice_store.add_lage("test-stimme", "gedehnt", lang, transcript="Sehr langsam gesprochen.")

    assert round(voice_store.longest_reference_seconds("test-stimme")) == 11


def test_eine_lage_wird_geprueft_wie_die_hauptaufnahme(
    voice_store: VoiceStore, tmp_path: Path
) -> None:
    """Eine übersteuerte Referenz bleibt auch wütend gesprochen unbrauchbar."""
    zu_kurz = tmp_path / "kurz.wav"
    write_wav(zu_kurz, np.full(24000 * 2, 0.9, dtype=np.float32), 24000)

    _, check = voice_store.add_lage("test-stimme", "knapp", zu_kurz, transcript="Kurz.")

    assert any("Referenz" in w for w in check.warnings)


# -- Manifest --------------------------------------------------------------


def _projekt(settings, text: str = "Erster Satz.\n\nZweiter Satz.") -> Project:
    from cloney.engines.registry import engine_info

    return Project.create(
        name="Lagenlauf",
        text=text,
        voice="test-stimme",
        engine=engine_info("dummy"),
        projects_dir=settings.projects_dir,
    )


def test_saetze_sind_zunaechst_neutral(settings, voice_store: VoiceStore) -> None:
    projekt = _projekt(settings)

    assert all(c.lage == "" for c in projekt.chunks)
    assert all(projekt.lage_of(c) == NEUTRAL for c in projekt.chunks)


def test_lage_wechseln_behaelt_den_seed(settings, voice_store: VoiceStore) -> None:
    """Der Unterschied zum Neuwürfeln: derselbe Wurf, eine andere Aufnahme. Nur
    so ist zu hören, was die Lage bewirkt, statt zugleich den Zufall zu bewegen."""
    projekt = _projekt(settings)
    vorher = projekt.chunks[0].seed

    projekt.set_lage(0, "ernst")

    assert projekt.chunks[0].seed == vorher
    assert projekt.chunks[0].attempts == 0
    assert projekt.chunks[0].lage == "ernst"


def test_lage_wechseln_verwirft_den_ton(settings, voice_store: VoiceStore) -> None:
    """Der vorhandene Ton stammt aus einer anderen Aufnahme. Ihn stehen zu
    lassen hieße, eine Lage zu zeigen, die nicht zu hören ist."""
    projekt = _projekt(settings)
    projekt.chunks[0].status = ChunkStatus.OK
    projekt.chunks[0].audio_file = "chunk_0000.wav"

    projekt.set_lage(0, "ernst")

    assert projekt.chunks[0].status == ChunkStatus.PENDING
    assert projekt.chunks[0].audio_file is None


def test_leer_heisst_der_vorgabe_folgen(settings, voice_store: VoiceStore) -> None:
    """Ein Satz trägt nur, was von der Vorgabe des Projekts abweicht."""
    projekt = _projekt(settings)
    projekt.set_default_lage("ernst")
    projekt.set_lage(0, "neutral")

    assert projekt.chunks[0].lage == "neutral"
    assert projekt.lage_of(projekt.chunks[0]) == NEUTRAL
    # Der zweite folgt weiterhin dem Projekt.
    assert projekt.chunks[1].lage == ""
    assert projekt.lage_of(projekt.chunks[1]) == "ernst"

    projekt.set_lage(0, "")
    assert projekt.lage_of(projekt.chunks[0]) == "ernst"


def test_die_lage_uebersteht_das_neu_segmentieren(settings, voice_store: VoiceStore) -> None:
    """Ein Tippfehler in Satz eins darf die Lagenarbeit an Satz zwei nicht kosten."""
    from cloney.engines.registry import engine_info

    projekt = _projekt(settings)
    projekt.set_lage(1, "ernst")
    projekt.chunks[1].status = ChunkStatus.OK
    projekt.chunks[1].audio_file = projekt.chunk_path(1).name
    projekt.chunk_path(1).parent.mkdir(parents=True, exist_ok=True)
    write_wav(projekt.chunk_path(1), np.zeros(2400, dtype=np.float32), 24000)
    projekt.save()

    projekt.reconfigure(
        text="Erster Satz, jetzt anders.\n\nZweiter Satz.",
        voice="test-stimme",
        engine=engine_info("dummy"),
    )

    assert projekt.chunks[1].lage == "ernst"


# -- Pipeline --------------------------------------------------------------


def test_jeder_satz_wird_gegen_seine_lage_konditioniert(
    settings, voice_store: VoiceStore, zweite_aufnahme: Path
) -> None:
    """Der Kern der Sache. Die Engine bekommt je Satz die Aufnahme, die zu
    seiner Lage gehört -- und nicht durchweg die neutrale."""
    voice_store.add_lage("test-stimme", "ernst", zweite_aufnahme, transcript="Ernst gesprochen.")
    projekt = _projekt(settings)
    projekt.set_lage(1, "ernst")
    projekt.save()

    benutzt: list[tuple[int, str]] = []

    class Mitschrift(DummyEngine):
        def synthesize(self, text, voice, seed):  # noqa: ANN001, ANN202
            benutzt.append((len(benutzt), voice.lage))
            return super().synthesize(text, voice, seed)

    synthesize_chunks(projekt, projekt.chunks, voice_store, Mitschrift)

    assert [lage for _, lage in benutzt] == [NEUTRAL, "ernst"]


def test_eine_geloeschte_lage_bricht_den_lauf_nicht_ab(
    settings, voice_store: VoiceStore, zweite_aufnahme: Path
) -> None:
    """Der Name steht im Manifest, die Aufnahme ist weg. Ein Kapitel soll dann
    mit der Hauptaufnahme durchlaufen statt mittendrin stehen zu bleiben."""
    voice_store.add_lage("test-stimme", "ernst", zweite_aufnahme, transcript="Ernst.")
    projekt = _projekt(settings)
    projekt.set_lage(1, "ernst")
    projekt.save()
    voice_store.delete_lage("test-stimme", "ernst")

    synthesize_chunks(projekt, projekt.chunks, voice_store, DummyEngine)

    assert all(c.status == ChunkStatus.SYNTHESIZED for c in projekt.chunks)


# -- Vorgabe des Projekts ---------------------------------------------------


def test_die_vorgabe_gilt_fuer_alle_ohne_eigene(settings, voice_store: VoiceStore) -> None:
    """Ein ganzes Kapitel ernst zu sprechen ist eine Einstellung, keine hundert
    Klicks."""
    projekt = _projekt(settings)

    projekt.set_default_lage("ernst")

    assert all(projekt.lage_of(c) == "ernst" for c in projekt.chunks)
    assert all(c.lage == "" for c in projekt.chunks)


def test_von_hand_gesetzte_lagen_ueberleben_den_wechsel(settings, voice_store: VoiceStore) -> None:
    """Sonst kostete jede Korrektur der Vorgabe die Feinarbeit an den Ausnahmen."""
    projekt = _projekt(settings)
    projekt.set_lage(1, "freundlich")

    projekt.set_default_lage("ernst")

    assert projekt.lage_of(projekt.chunks[0]) == "ernst"
    assert projekt.lage_of(projekt.chunks[1]) == "freundlich"


def test_die_vorgabe_verwirft_nur_den_ton_der_betroffenen(
    settings, voice_store: VoiceStore
) -> None:
    projekt = _projekt(settings)
    projekt.set_lage(1, "freundlich")
    for chunk in projekt.chunks:
        chunk.status = ChunkStatus.OK
        chunk.audio_file = f"chunk_{chunk.index:04d}.wav"

    betroffen = projekt.set_default_lage("ernst")

    assert betroffen == 1
    assert projekt.chunks[0].audio_file is None
    assert projekt.chunks[1].audio_file == "chunk_0001.wav"


def test_dieselbe_vorgabe_noch_einmal_kostet_nichts(settings, voice_store: VoiceStore) -> None:
    projekt = _projekt(settings)
    projekt.set_default_lage("ernst")
    for chunk in projekt.chunks:
        chunk.audio_file = f"chunk_{chunk.index:04d}.wav"

    assert projekt.set_default_lage("ernst") == 0
    assert all(c.audio_file for c in projekt.chunks)


def test_die_vorgabe_uebersteht_das_neu_segmentieren(settings, voice_store: VoiceStore) -> None:
    from cloney.engines.registry import engine_info

    projekt = _projekt(settings)
    projekt.set_default_lage("ernst")

    projekt.reconfigure(text="Ganz anderer Text.", voice="test-stimme", engine=engine_info("dummy"))

    assert projekt.lage == "ernst"


# -- Auf mehrere Sätze anwenden --------------------------------------------


def test_eine_lage_auf_mehrere_saetze(settings, voice_store: VoiceStore) -> None:
    projekt = _projekt(settings, "Eins.\n\nZwei.\n\nDrei.")

    geaendert = projekt.set_lage_many([0, 2], "ernst")

    assert geaendert == 2
    assert [projekt.lage_of(c) for c in projekt.chunks] == ["ernst", NEUTRAL, "ernst"]


def test_wer_schon_so_steht_kostet_keine_arbeit(settings, voice_store: VoiceStore) -> None:
    """Beim Anwenden auf eine Auswahl trifft man regelmäßig Sätze, die schon so
    stehen -- deren Ton darf das nicht kosten."""
    projekt = _projekt(settings, "Eins.\n\nZwei.")
    projekt.set_lage(0, "ernst")
    projekt.chunks[0].audio_file = "chunk_0000.wav"
    projekt.chunks[0].status = ChunkStatus.OK

    geaendert = projekt.set_lage_many([0, 1], "ernst")

    assert geaendert == 1
    assert projekt.chunks[0].audio_file == "chunk_0000.wav"
    assert projekt.chunks[0].status == ChunkStatus.OK


def test_der_lauf_folgt_der_vorgabe(
    settings, voice_store: VoiceStore, zweite_aufnahme: Path
) -> None:
    """Der eigentliche Punkt der Vorgabe: die Engine bekommt die Aufnahme, die
    dazu gehört -- auch für Sätze, an denen nichts steht."""
    voice_store.add_lage("test-stimme", "ernst", zweite_aufnahme, transcript="Ernst.")
    projekt = _projekt(settings)
    projekt.set_default_lage("ernst")
    projekt.set_lage(1, "neutral")
    projekt.save()

    benutzt: list[str] = []

    class Mitschrift(DummyEngine):
        def synthesize(self, text, voice, seed):  # noqa: ANN001, ANN202
            benutzt.append(voice.lage)
            return super().synthesize(text, voice, seed)

    synthesize_chunks(projekt, projekt.chunks, voice_store, Mitschrift)

    assert benutzt == ["ernst", NEUTRAL]
