"""Satzbau von Hand: einfügen, verschmelzen -- und wer danach die Vorlage ist.

Der eigene Schnitt trennt feiner, als es der Sprechfluss oft verträgt: F5-TTS
erzeugt nur rund 22 Sekunden am Stück, die Referenz geht davon ab, und was
übrig bleibt, teilt lange Absätze in kurze Stücke. Zwei kurze Sätze in einem Zug
gesprochen tragen die Betonung über eine Grenze hinweg, an der sonst zwei
getrennte Generierungen aneinanderstoßen.

Die Falle steckt nicht im Verschmelzen selbst, sondern danach: der Quelltext ist
die Vorlage, aus der geschnitten wird. Bliebe er stehen, wie er war, machte der
nächste Klick auf "Vorlage übernehmen" -- ein Wechsel der Stimme genügt -- jede
Handarbeit wortlos zunichte. Deshalb prüfen die Tests hier beides zusammen.
"""

from __future__ import annotations

import pytest

from cloney.config import Settings
from cloney.core.project import ChunkStatus, Project, derive_seed
from cloney.core.segment import join_raw
from cloney.core.voices import VoiceStore
from cloney.engines.dummy import DummyEngine


def _projekt(settings: Settings, text: str, target_seconds: float = 1.5) -> Project:
    return Project.create(
        name="Kapitel",
        text=text,
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
        target_seconds=target_seconds,
    )


def _gerendert(settings: Settings, voice_store: VoiceStore, text: str) -> Project:
    """Ein Projekt mit fertigem Ton -- die Ausgangslage jeder Änderung."""
    from cloney.asr.dummy import DummyASR
    from cloney.pipeline import run_project

    project = _projekt(settings, text)
    run_project(project, settings, voice_store, DummyEngine, DummyASR)
    return project


def _texte(project: Project) -> list[str]:
    return [c.raw_text for c in project.chunks]


# -- Verbinden zweier Rohtexte ---------------------------------------------


@pytest.mark.parametrize(
    ("links", "rechts", "erwartet"),
    [
        ("Erster Satz.", "Zweiter Satz.", "Erster Satz. Zweiter Satz."),
        # Eine Überschrift trägt keinen Punkt. Ohne ihn läse die Engine sie in
        # den folgenden Satz hinein -- genau das, was das Chunking verhindert.
        ("Kapitel eins", "Es begann früh.", "Kapitel eins. Es begann früh."),
        # Ein Komma bleibt stehen: dort war ein überlanger Satz zur Not getrennt
        # worden, und ein Punkt machte aus dem Teilsatz einen ganzen.
        ("Als er kam,", "war es zu spät.", "Als er kam, war es zu spät."),
        ("Und dann?", "Nichts.", "Und dann? Nichts."),
        # Das schließende Anführungszeichen steht hinter dem Punkt.
        ('Er sagte "Ja."', "Dann ging er.", 'Er sagte "Ja." Dann ging er.'),
        ("", "Allein.", "Allein."),
        ("Allein.", "", "Allein."),
    ],
)
def test_join_raw(links: str, rechts: str, erwartet: str) -> None:
    assert join_raw(links, rechts) == erwartet


# -- Einfügen ---------------------------------------------------------------


def test_einfuegen_davor_rueckt_die_folgenden_nach(settings: Settings) -> None:
    project = _projekt(settings, "Erster Satz. Zweiter Satz.")

    stelle = project.insert_chunk(1, "Dazwischen.")

    assert stelle == 1
    assert _texte(project) == ["Erster Satz.", "Dazwischen.", "Zweiter Satz."]
    assert [c.index for c in project.chunks] == [0, 1, 2]


def test_einfuegen_danach_setzt_ihn_dahinter(settings: Settings) -> None:
    project = _projekt(settings, "Erster Satz. Zweiter Satz.")

    stelle = project.insert_chunk(0, "Dazwischen.", danach=True)

    assert stelle == 1
    assert _texte(project) == ["Erster Satz.", "Dazwischen.", "Zweiter Satz."]


def test_der_neue_satz_tritt_dem_absatz_bei(settings: Settings) -> None:
    """Danach eingesetzt heißt: an dessen Stelle am Absatzende.

    An den Absätzen hängt beim Zusammenbau die Pausenlänge. Bliebe das Ende beim
    Vorgänger, stünde die lange Pause mitten im Absatz.
    """
    project = _projekt(settings, "Erster Absatz.\n\nZweiter Absatz.")
    assert project.chunks[0].ends_paragraph

    project.insert_chunk(0, "Noch dazu.", danach=True)

    assert not project.chunks[0].ends_paragraph
    assert project.chunks[1].ends_paragraph
    assert project.text_aus_chunks() == "Erster Absatz. Noch dazu.\n\nZweiter Absatz."


def test_davor_eingesetzt_beendet_keinen_absatz(settings: Settings) -> None:
    project = _projekt(settings, "Erster Absatz.\n\nZweiter Absatz.")

    project.insert_chunk(1, "Vorweg.")

    assert not project.chunks[1].ends_paragraph
    assert project.text_aus_chunks() == "Erster Absatz.\n\nVorweg. Zweiter Absatz."


def test_der_neue_satz_erbt_die_lage(settings: Settings) -> None:
    """Sonst fiele ein Satz mitten in einer ernst gesprochenen Passage als
    einziger in die Vorgabe des Projekts zurück."""
    project = _projekt(settings, "Erster Satz. Zweiter Satz.")
    project.chunks[1].lage = "ernst"

    project.insert_chunk(1, "Dazwischen.")

    assert project.chunks[1].lage == "ernst"


def test_eingefuegter_satz_wird_normalisiert(settings: Settings) -> None:
    project = _projekt(settings, "Erster Satz.")

    project.insert_chunk(0, "Am 3. Mai 2024.", danach=True)

    assert "dritten Mai" in project.chunks[1].normalized_text
    assert project.chunks[1].status == ChunkStatus.PENDING
    assert project.chunks[1].seed == derive_seed(project.id, 1, 0)


def test_ein_satz_ohne_inhalt_wird_abgelehnt(settings: Settings) -> None:
    """Ein leerer Chunk ginge in die Synthese und käme als Stille zurück."""
    project = _projekt(settings, "Erster Satz.")

    with pytest.raises(ValueError, match="keinen Inhalt"):
        project.insert_chunk(0, "   ")

    assert len(project.chunks) == 1


def test_einfuegen_zieht_den_ton_der_folgenden_mit(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Ohne das entwertete ein eingefügter Satz jede Aufnahme dahinter: die
    Dateien heißen nach ihrer Nummer, und die verschiebt sich."""
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz. Dritter Satz.")
    inhalte = {c.raw_text: project.chunk_path(c.index).read_bytes() for c in project.chunks}

    project.insert_chunk(1, "Dazwischen.")

    for chunk in project.chunks:
        if chunk.raw_text == "Dazwischen.":
            assert chunk.audio_file is None
            continue
        assert project.chunk_path(chunk.index).read_bytes() == inhalte[chunk.raw_text]
        assert chunk.status == ChunkStatus.OK
    # Keine Leichen und keine Zwischennamen.
    assert sorted(p.name for p in project.chunks_dir.iterdir()) == [
        "chunk_0000.wav",
        "chunk_0002.wav",
        "chunk_0003.wav",
    ]


# -- Verschmelzen -----------------------------------------------------------


def test_verschmelzen_fasst_zwei_saetze_zusammen(settings: Settings) -> None:
    project = _projekt(settings, "Erster Satz. Zweiter Satz. Dritter Satz.")
    assert len(project.chunks) == 3

    stelle = project.merge_chunks(0)

    assert stelle == 0
    assert _texte(project) == ["Erster Satz. Zweiter Satz.", "Dritter Satz."]
    assert project.chunks[0].normalized_text == "Erster Satz. Zweiter Satz."
    assert project.chunks[0].seed == derive_seed(project.id, 0, 0)


def test_verschmolzener_titel_bekommt_seinen_punkt(settings: Settings) -> None:
    """Ein Titel trägt keinen Punkt. Verschmolzen mit dem folgenden Satz liefe
    er ohne Absetzen in ihn hinein -- der Punkt steht deshalb im Rohtext, wo er
    zu sehen und zu ändern ist."""
    project = _projekt(settings, "Kapitel eins\n\nEs begann früh.", target_seconds=5.0)
    assert project.chunks[0].is_heading

    project.merge_chunks(0)

    assert project.chunks[0].raw_text == "Kapitel eins. Es begann früh."
    # Der neue Satz ist kein Titel mehr: er trägt Fließtext mit sich.
    assert not project.chunks[0].is_heading


def test_verschmelzen_nimmt_das_absatzende_des_zweiten(settings: Settings) -> None:
    project = _projekt(settings, "Erster Satz. Zweiter Satz.\n\nNeuer Absatz.")

    project.merge_chunks(0)

    assert project.chunks[0].ends_paragraph
    assert project.text_aus_chunks() == "Erster Satz. Zweiter Satz.\n\nNeuer Absatz."


def test_verschmelzen_verwirft_den_ton_beider(settings: Settings, voice_store: VoiceStore) -> None:
    """Aneinandergeklebt hörte man genau die Naht, deretwegen verschmolzen wird."""
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz. Dritter Satz.")
    dritter = project.chunk_path(2).read_bytes()

    project.merge_chunks(0)

    assert project.chunks[0].audio_file is None
    assert project.chunks[0].status == ChunkStatus.PENDING
    assert not project.chunk_path(0).exists()
    # Der unbeteiligte Satz rückt samt Aufnahme nach.
    assert project.chunks[1].raw_text == "Dritter Satz."
    assert project.chunk_path(1).read_bytes() == dritter
    # Die Datei mit der höchsten Nummer gehört zu keinem Satz mehr.
    assert sorted(p.name for p in project.chunks_dir.iterdir()) == ["chunk_0001.wav"]


def test_verschmelzen_verwirft_die_fertige_spur(
    settings: Settings, voice_store: VoiceStore
) -> None:
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz.")
    assert project.output_path.exists()

    project.merge_chunks(0)

    assert not project.output_path.exists()
    assert project.output_file is None


def test_hinter_dem_letzten_satz_gibt_es_nichts_zu_verschmelzen(settings: Settings) -> None:
    project = _projekt(settings, "Erster Satz. Zweiter Satz.")

    with pytest.raises(ValueError, match="kommt keiner mehr"):
        project.merge_chunks(1)


def test_verschmelzen_kennt_keine_obergrenze(settings: Settings) -> None:
    """Wer zwei Sätze zusammenzieht, meint es. Zu lang zu werden ist ein
    Hinweis in der Tabelle wert, aber kein Grund, den Handgriff zu verweigern."""
    project = _projekt(settings, "Erster Satz. Zweiter Satz.", target_seconds=1.5)
    grenze = project.target_chunk_seconds * 14.0

    project.merge_chunks(0)

    assert len(project.chunks) == 1
    assert len(project.chunks[0].normalized_text) > grenze


# -- Wer nach der Handarbeit die Vorlage ist --------------------------------


def test_quelltext_folgt_der_satzliste(settings: Settings) -> None:
    """Stünde dort weiter die alte Fassung, widerspräche das Textfeld in den
    Einstellungen dem, was gesprochen wird."""
    project = _projekt(settings, "Erster Satz.\n\nZweiter Satz.")

    project.insert_chunk(1, "Dazwischen.")

    assert project.source_text == "Erster Satz.\n\nDazwischen. Zweiter Satz."
    assert Project.load(project.root).source_text == project.source_text


def test_auch_ein_geaenderter_satztext_landet_im_quelltext(settings: Settings) -> None:
    project = _projekt(settings, "Erster Satz. Zweiter Satz.")

    project.retext(0, "Ganz anders.")

    assert project.source_text == "Ganz anders. Zweiter Satz."
    assert project.handschnitt


def test_handschnitt_ueberlebt_den_stimmwechsel(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Der eigentliche Punkt. Ein frischer Schnitt trennte das Verschmolzene
    wieder -- und ein Wechsel der Stimme genügt, um ihn auszulösen."""
    voice_store.add("zweite-stimme", voice_store.get("test-stimme").audio_path, transcript="Hallo.")
    project = _projekt(settings, "Erster Satz. Zweiter Satz. Dritter Satz.")
    project.merge_chunks(0)

    bericht = project.reconfigure(
        text=project.source_text,
        voice="zweite-stimme",
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    assert _texte(project) == ["Erster Satz. Zweiter Satz.", "Dritter Satz."]
    assert project.voice == "zweite-stimme"
    assert project.handschnitt
    assert not bericht["neu_geschnitten"]


def test_unveraendertes_uebernehmen_laesst_den_handschnitt_stehen(
    settings: Settings, voice_store: VoiceStore
) -> None:
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz. Dritter Satz.")
    project.merge_chunks(0)
    project.insert_chunk(1, "Neu dazu.")

    bericht = project.reconfigure(
        text=project.source_text,
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    assert _texte(project) == ["Erster Satz. Zweiter Satz.", "Neu dazu.", "Dritter Satz."]
    # Der eine Satz, der noch Ton hat, behält ihn; die beiden von Hand
    # entstandenen hatten nie welchen.
    assert bericht["behalten"] == 1
    assert project.chunk_path(2).exists()
    assert not bericht["neu_geschnitten"]


def test_ein_geaenderter_text_schneidet_neu(settings: Settings) -> None:
    """Wer im Textfeld etwas ändert, meint den Text -- und dann gilt wieder er."""
    project = _projekt(settings, "Erster Satz. Zweiter Satz. Dritter Satz.")
    project.merge_chunks(0)

    bericht = project.reconfigure(
        text="Erster Satz. Zweiter Satz. Dritter Satz. Vierter Satz.",
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    assert len(project.chunks) == 4
    assert not project.handschnitt
    assert bericht["neu_geschnitten"]


def test_die_grenze_der_engine_sticht_den_handschnitt(settings: Settings) -> None:
    """Passt ein verschmolzener Satz nicht mehr in eine Generierung, teilte die
    Engine ihn selbst -- mit einer Naht, die sich nicht nachbessern lässt."""
    project = _projekt(settings, "Erster Satz. Zweiter Satz.", target_seconds=1.5)
    project.merge_chunks(0)
    assert len(project.chunks) == 1

    bericht = project.reconfigure(
        text=project.source_text,
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.0,
        max_seconds=1.0,
    )

    assert len(project.chunks) == 2
    assert not project.handschnitt
    assert bericht["neu_geschnitten"]


def test_ohne_handarbeit_bleibt_alles_wie_bisher(settings: Settings) -> None:
    """Ein Projekt, an dem nichts von Hand geschnitten wurde, wird beim
    Übernehmen frisch geschnitten -- ein größeres Budget fasst dann zusammen."""
    project = _projekt(settings, "Erster Satz. Zweiter Satz.", target_seconds=1.0)
    assert len(project.chunks) == 2

    project.reconfigure(
        text=project.source_text,
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=20.0,
    )

    assert len(project.chunks) == 1
