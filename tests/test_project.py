from __future__ import annotations

from cloney.config import Settings
from cloney.core.project import ChunkStatus, Project, derive_seed
from cloney.core.voices import VoiceStore
from cloney.engines.dummy import DummyEngine
from cloney.engines.f5_german import F5_INFO

TEXT = "Erster Satz hier.\n\nZweiter Absatz mit Inhalt. Und noch ein Satz dazu."


def _create(settings: Settings, target_seconds: float = 3.0) -> Project:
    return Project.create(
        name="Testprojekt",
        text=TEXT,
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
        target_seconds=target_seconds,
    )


def test_manifest_roundtrip(settings: Settings) -> None:
    project = _create(settings)
    loaded = Project.load(project.root)
    assert loaded.id == project.id
    assert [c.normalized_text for c in loaded.chunks] == [c.normalized_text for c in project.chunks]
    assert loaded.root == project.root


def test_seeds_sind_reproduzierbar() -> None:
    assert derive_seed("abc", 3, 0) == derive_seed("abc", 3, 0)
    assert derive_seed("abc", 3, 0) != derive_seed("abc", 3, 1)
    assert derive_seed("abc", 3, 0) != derive_seed("abc", 4, 0)


def test_reroll_setzt_chunk_zurueck(settings: Settings) -> None:
    project = _create(settings)
    chunk = project.chunks[0]
    chunk.status = ChunkStatus.OK
    chunk.cer = 0.42
    old_seed = chunk.seed

    rerolled = project.reroll(0)
    assert rerolled.seed != old_seed
    assert rerolled.status == ChunkStatus.PENDING
    assert rerolled.cer is None
    assert rerolled.attempts == 1


def test_retext_normalisiert_neu(settings: Settings) -> None:
    project = _create(settings)
    project.retext(0, "Am 3. Mai 2024.")
    assert "dritten Mai" in project.chunks[0].normalized_text
    assert project.chunks[0].status == ChunkStatus.PENDING


def test_fortschritt_und_median(settings: Settings) -> None:
    project = _create(settings)
    for chunk in project.chunks:
        chunk.status = ChunkStatus.OK
        chunk.cer = 0.05
    assert project.is_complete
    assert project.progress == (len(project.chunks), len(project.chunks))
    assert project.median_cer() == 0.05


def test_liste_ist_nach_datum_sortiert(settings: Settings) -> None:
    _create(settings)
    assert len(Project.list_all(settings.projects_dir)) == 1


def test_engine_grenze_verkleinert_die_chunks(settings: Settings) -> None:
    """Hat eine Engine ein Budget je Generierung, muss die Chunk-Planung darunter
    bleiben -- sonst teilt das Modell selbst auf und erzeugt Nähte im Chunk, die
    sich nicht mehr einzeln nachbessern lassen."""
    text = " ".join(f"Dies ist der Satz Nummer {i} in einem längeren Absatz." for i in range(40))
    common = {
        "name": "Budget",
        "text": text,
        "voice": "test-stimme",
        "projects_dir": settings.projects_dir,
        "chars_per_second": 14.0,
        "target_seconds": 20.0,
    }

    ohne_grenze = Project.create(engine=DummyEngine.info, reference_seconds=9.0, **common)
    mit_grenze = Project.create(engine=F5_INFO, reference_seconds=9.0, **common)

    assert ohne_grenze.target_chunk_seconds == 20.0
    # 22 s Gesamtbudget minus 9 s Referenz.
    assert mit_grenze.target_chunk_seconds == 13.0
    assert len(mit_grenze.chunks) > len(ohne_grenze.chunks)
    assert all(len(c.normalized_text) <= 13.0 * 14.0 for c in mit_grenze.chunks)


def test_lange_referenz_schrumpft_das_budget_weiter(settings: Settings) -> None:
    kurz = Project.create(
        name="Kurz",
        text="Ein Satz hier.",
        voice="test-stimme",
        engine=F5_INFO,
        projects_dir=settings.projects_dir,
        reference_seconds=6.0,
        target_seconds=20.0,
    )
    lang = Project.create(
        name="Lang",
        text="Ein Satz hier.",
        voice="test-stimme",
        engine=F5_INFO,
        projects_dir=settings.projects_dir,
        reference_seconds=12.0,
        target_seconds=20.0,
    )
    assert kurz.target_chunk_seconds > lang.target_chunk_seconds


# -- Ein bestehendes Projekt ändern ----------------------------------------


def _gerendert(settings: Settings, voice_store: VoiceStore, text: str) -> Project:
    """Ein Projekt mit fertigem Ton -- die Ausgangslage jeder Änderung."""
    from cloney.asr.dummy import DummyASR
    from cloney.pipeline import run_project

    project = Project.create(
        name="Kapitel",
        text=text,
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
        target_seconds=1.5,
    )
    run_project(project, settings, voice_store, DummyEngine, DummyASR)
    return project


def test_unveraenderte_saetze_behalten_ihren_ton(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Ein Tippfehler in Satz eins darf nicht die Arbeit an Satz drei kosten."""
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz. Dritter Satz.")
    vorher = {c.normalized_text: c.seed for c in project.chunks}

    bericht = project.reconfigure(
        text="Erster Satz geändert. Zweiter Satz. Dritter Satz.",
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    nachher = {c.normalized_text: c for c in project.chunks}
    for satz in ("Zweiter Satz.", "Dritter Satz."):
        assert nachher[satz].seed == vorher[satz]
        assert nachher[satz].status == ChunkStatus.OK
        assert project.chunk_path(nachher[satz].index).exists()

    geaendert = nachher["Erster Satz geändert."]
    assert geaendert.status == ChunkStatus.PENDING
    assert not project.chunk_path(geaendert.index).exists()
    assert bericht == {"behalten": 2, "neu": 1, "entfernt": 1}


def test_umnummerierte_saetze_finden_ihren_ton_wieder(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Fällt ein Satz vorne weg, rutscht alles nach -- Quelle und Ziel des
    Umzugs überlappen sich dann."""
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz. Dritter Satz.")
    inhalte = {c.normalized_text: project.chunk_path(c.index).read_bytes() for c in project.chunks}

    project.reconfigure(
        text="Zweiter Satz. Dritter Satz.",
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    assert [c.normalized_text for c in project.chunks] == ["Zweiter Satz.", "Dritter Satz."]
    for chunk in project.chunks:
        assert project.chunk_path(chunk.index).read_bytes() == inhalte[chunk.normalized_text]
    # Keine Leichen: genau zwei Tondateien, keine Zwischennamen.
    assert sorted(p.name for p in project.chunks_dir.iterdir()) == [
        "chunk_0000.wav",
        "chunk_0001.wav",
    ]


def test_andere_schreibweise_gleiche_sprechfassung_behaelt_den_ton(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Verglichen wird die normalisierte Fassung: wer '3.' zu 'dritten' ändert,
    hört dasselbe und soll nicht neu rendern müssen."""
    project = _gerendert(settings, voice_store, "Am 3. Mai geschah es.")
    seed = project.chunks[0].seed

    project.reconfigure(
        text="Am dritten Mai geschah es.",
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    assert project.chunks[0].raw_text == "Am dritten Mai geschah es."
    assert project.chunks[0].seed == seed
    assert project.chunks[0].status == ChunkStatus.OK


def test_stimmwechsel_verwirft_allen_ton(settings: Settings, voice_store: VoiceStore) -> None:
    """Sonst entstünde eine Spur aus zwei Sprechern."""
    voice_store.add("zweite-stimme", voice_store.get("test-stimme").audio_path, transcript="Hallo.")
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz.")

    bericht = project.reconfigure(
        text=project.source_text,
        voice="zweite-stimme",
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    assert project.voice == "zweite-stimme"
    assert bericht["behalten"] == 0
    assert all(c.status == ChunkStatus.PENDING for c in project.chunks)
    assert list(project.chunks_dir.iterdir()) == []


def test_enginewechsel_zieht_samplerate_und_regler_nach(
    settings: Settings, voice_store: VoiceStore
) -> None:
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz.")
    project.engine_options = {"speed": 0.9, "pitch": 40.0}

    project.reconfigure(
        text=project.source_text,
        voice=project.voice,
        engine=F5_INFO,
        reference_seconds=8.0,
        target_seconds=20.0,
    )

    assert project.engine == "f5-de"
    assert project.sample_rate == F5_INFO.sample_rate
    # 'pitch' kennt F5-TTS nicht -- der Regler fällt weg, statt still weiterzuwirken.
    assert project.engine_options == {"speed": 0.9}
    # Die Engine erzeugt höchstens 22s am Stück, 8s davon gehen für die Referenz drauf.
    assert project.target_chunk_seconds == 14.0


def test_aenderung_verwirft_die_fertige_spur(settings: Settings, voice_store: VoiceStore) -> None:
    """Sie gehörte zum alten Text und wäre nach der Änderung eine Lüge."""
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz.")
    assert project.output_path.exists()

    project.reconfigure(
        text="Ganz anderer Satz.",
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    assert project.output_file is None
    assert not project.output_path.exists()


def test_aenderung_uebersteht_den_neustart(settings: Settings, voice_store: VoiceStore) -> None:
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz.")
    project.reconfigure(
        text="Erster Satz. Zweiter Satz. Dritter Satz.",
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    geladen = Project.load(project.root)
    assert geladen.source_text.endswith("Dritter Satz.")
    assert len(geladen.chunks) == 3
    assert sum(1 for c in geladen.chunks if c.status == ChunkStatus.OK) == 2


def test_unveraendertes_uebernehmen_laesst_alles_stehen(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Das Formular soll sich gefahrlos wiederholen lassen -- ein zweites
    Absenden derselben Werte darf die fertige Spur nicht kosten."""
    project = _gerendert(settings, voice_store, "Erster Satz. Zweiter Satz.")
    spur = project.output_path.read_bytes()

    bericht = project.reconfigure(
        text=project.source_text,
        voice=project.voice,
        engine=DummyEngine.info,
        target_seconds=1.5,
    )

    assert bericht == {"behalten": 2, "neu": 0, "entfernt": 0}
    assert project.output_file is not None
    assert project.output_path.read_bytes() == spur
