"""Vollständiger Durchstich ohne GPU und ohne Netz.

Diese Tests sind der Grund für DummyEngine und DummyASR: sie beweisen die
Orchestrierung -- Synthese, Qualitätskontrolle, Retry, Resume, Zusammenbau --
ohne ein einziges Modell zu laden.
"""

from __future__ import annotations

from cloney.asr.dummy import DummyASR
from cloney.config import Settings
from cloney.core.audio import duration_seconds, read_wav
from cloney.core.project import ChunkStatus, Project
from cloney.core.voices import VoiceStore
from cloney.engines.dummy import DummyEngine
from cloney.pipeline import ProgressEvent, run_project

TEXT = (
    "Am 3. Mai 2024 begann alles ganz harmlos.\n\n"
    "Dr. Meier sagte z.B., dass es 1.250,50 € kosten würde. "
    "Danach ging es schnell weiter."
)


def _project(settings: Settings) -> Project:
    return Project.create(
        name="Durchstich",
        text=TEXT,
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
        target_seconds=4.0,
    )


def test_durchstich_erzeugt_wav_und_manifest(settings: Settings, voice_store: VoiceStore) -> None:
    project = _project(settings)
    events: list[ProgressEvent] = []

    run_project(project, settings, voice_store, DummyEngine, DummyASR, events.append)

    assert project.is_complete
    assert project.output_path.exists()
    audio, rate = read_wav(project.output_path)
    assert rate == 24000
    assert duration_seconds(audio, rate) > 1.0

    # Jeder Chunk hat eine eigene Datei und eine gemessene Fehlerrate.
    for chunk in project.chunks:
        assert project.chunk_path(chunk.index).exists()
        assert chunk.cer == 0.0
        assert chunk.engine == "dummy"

    assert {e.phase for e in events} >= {"synth", "qc", "assemble"}
    # Das Manifest auf Platte spiegelt den Endzustand.
    assert Project.load(project.root).is_complete


def test_qc_erkennt_fehler_und_retry_repariert(settings: Settings, voice_store: VoiceStore) -> None:
    """Der erste Seed jedes Chunks liefert Müll, der zweite ist sauber --
    genau der Fall, den die Retry-Schleife abfangen soll."""
    project = _project(settings)
    bad_seeds = {c.seed for c in project.chunks}

    run_project(
        project,
        settings,
        voice_store,
        DummyEngine,
        lambda: DummyASR(corrupt_seeds=bad_seeds),
    )

    assert project.is_complete
    assert all(c.attempts == 1 for c in project.chunks)
    assert all(c.cer == 0.0 for c in project.chunks)


def test_dauerhaft_schlechte_chunks_werden_markiert(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Wenn kein Versuch hilft, bleibt der Chunk markiert statt still
    in die fertige Spur zu rutschen."""
    project = _project(settings)
    settings.max_retries = 1

    run_project(
        project,
        settings,
        voice_store,
        DummyEngine,
        lambda: DummyASR(always_corrupt=True),
    )

    assert not project.is_complete
    assert all(c.status == ChunkStatus.NEEDS_REVIEW for c in project.chunks)
    assert all(c.attempts == settings.max_retries for c in project.chunks)
    # Die Spur wird trotzdem gebaut -- man will hören, was schiefging.
    assert project.output_path.exists()


def test_resume_rendert_nur_offene_chunks(settings: Settings, voice_store: VoiceStore) -> None:
    project = _project(settings)
    run_project(project, settings, voice_store, DummyEngine, DummyASR)

    # Abbruch simulieren: ein Chunk verliert seine Datei und seinen Zustand.
    victim = project.chunks[-1]
    project.chunk_path(victim.index).unlink()
    victim.status = ChunkStatus.PENDING
    victim.audio_file = None
    project.save()

    rendered: list[int] = []

    class CountingEngine(DummyEngine):
        def synthesize(self, text, voice, seed):  # type: ignore[no-untyped-def]
            rendered.append(len(rendered))
            return super().synthesize(text, voice, seed)

    reloaded = Project.load(project.root)
    run_project(reloaded, settings, voice_store, CountingEngine, DummyASR)

    assert len(rendered) == 1
    assert reloaded.is_complete


def test_ohne_asr_entfaellt_die_messung(settings: Settings, voice_store: VoiceStore) -> None:
    project = _project(settings)
    run_project(project, settings, voice_store, DummyEngine, asr_factory=None)

    assert project.is_complete
    # cer = None sagt ehrlich: nicht geprüft, nicht "fehlerfrei".
    assert all(c.cer is None for c in project.chunks)


def test_synthesefehler_landet_im_manifest(settings: Settings, voice_store: VoiceStore) -> None:
    project = _project(settings)

    class BrokenEngine(DummyEngine):
        def synthesize(self, text, voice, seed):  # type: ignore[no-untyped-def]
            raise RuntimeError("Server nicht erreichbar")

    run_project(project, settings, voice_store, BrokenEngine, DummyASR)

    assert all(c.status == ChunkStatus.FAILED for c in project.chunks)
    assert all("Server nicht erreichbar" in (c.error or "") for c in project.chunks)


def test_referenz_vorspann_wird_abgeschnitten(settings: Settings, voice_store: VoiceStore) -> None:
    """F5-TTS lässt gelegentlich ein Stück der Referenz am Anfang stehen. Nach
    Lautstärke ist es nicht zu fassen -- es ist Sprache. Über die Rückschrift
    schon: sie sagt, ab welchem Wort der gewünschte Text beginnt."""
    from cloney.core.audio import duration_seconds, read_wav

    project = _project(settings)
    run_project(
        project,
        settings,
        voice_store,
        DummyEngine,
        lambda: DummyASR(bleed_words="Rest der Referenzaufnahme von vorher"),
    )

    erster = project.chunks[0]
    assert erster.trimmed_bleed_s is not None
    assert erster.trimmed_bleed_s > 0
    # Der Vorspann zählt nicht als Fehler -- verglichen wird ohne ihn.
    assert erster.cer == 0.0
    # Und er ist wirklich aus der Datei verschwunden.
    audio, rate = read_wav(project.chunk_path(0))
    assert duration_seconds(audio, rate) > 0


def test_ohne_vorspann_wird_nichts_angetastet(settings: Settings, voice_store: VoiceStore) -> None:
    from cloney.core.audio import read_wav

    project = _project(settings)
    run_project(project, settings, voice_store, DummyEngine, DummyASR)
    vorher = len(read_wav(project.chunk_path(0))[0])

    assert all(c.trimmed_bleed_s is None for c in project.chunks)
    assert len(read_wav(project.chunk_path(0))[0]) == vorher


def test_abschneiden_laesst_sich_abschalten(settings: Settings, voice_store: VoiceStore) -> None:
    settings.trim_reference_bleed = False
    project = _project(settings)
    run_project(
        project, settings, voice_store, DummyEngine, lambda: DummyASR(bleed_words="Rest davor")
    )
    assert all(c.trimmed_bleed_s is None for c in project.chunks)
