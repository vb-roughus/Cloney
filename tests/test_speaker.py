"""Stimmähnlichkeit als zweite Messgröße neben der Fehlerrate.

Die Fehlerrate prüft, ob die richtigen Wörter herauskommen. Ob es dabei noch
nach der Referenzstimme klingt, sagt sie nicht -- genau diese Lücke wird hier
geschlossen, und genau das halten diese Tests fest.
"""

from __future__ import annotations

import numpy as np
import pytest

from cloney.config import Settings
from cloney.core.metrics import cosine_similarity
from cloney.core.voices import VoiceStore
from cloney.speaker.dummy import DummySpeakerEmbedder

SR = 24000


def _ton(frequenz: float, sekunden: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(sekunden * SR), dtype=np.float32) / SR
    huelle = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (amplitude * huelle * np.sin(2 * np.pi * frequenz * t)).astype(np.float32)


def test_gleiches_audio_ist_identisch() -> None:
    e = DummySpeakerEmbedder()
    audio = _ton(150)
    assert cosine_similarity(e.embed(audio, SR), e.embed(audio, SR)) == pytest.approx(1.0)


def test_gleiche_stimme_andere_laenge_bleibt_aehnlich() -> None:
    """Die Einbettung soll die Stimme beschreiben, nicht die Dauer."""
    e = DummySpeakerEmbedder()
    kurz, lang = e.embed(_ton(150, 2.0), SR), e.embed(_ton(150, 5.0), SR)
    assert cosine_similarity(kurz, lang) > 0.9


def test_andere_stimme_faellt_deutlich_ab() -> None:
    e = DummySpeakerEmbedder()
    a, b = e.embed(_ton(150), SR), e.embed(_ton(700), SR)
    assert cosine_similarity(a, b) < 0.5


def test_leere_oder_unpassende_vektoren_ergeben_null() -> None:
    assert cosine_similarity(np.zeros(0), np.zeros(0)) == 0.0
    assert cosine_similarity(np.zeros(4), np.zeros(4)) == 0.0
    assert cosine_similarity(np.ones(4), np.ones(8)) == 0.0


# -- Zusammenspiel mit dem Lauf --------------------------------------------


def test_lauf_misst_die_stimmaehnlichkeit(settings: Settings, voice_store: VoiceStore) -> None:
    from cloney.asr.dummy import DummyASR
    from cloney.core.project import Project
    from cloney.engines.dummy import DummyEngine
    from cloney.pipeline import run_project

    project = Project.create(
        name="Ähnlichkeit",
        text="Am 3. Mai 2024 begann alles.",
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
    )
    run_project(
        project, settings, voice_store, DummyEngine, DummyASR, embedder_factory=DummySpeakerEmbedder
    )

    assert all(c.speaker_similarity is not None for c in project.chunks)
    assert project.median_similarity() is not None


def test_ohne_fabrik_wird_nicht_gemessen(settings: Settings, voice_store: VoiceStore) -> None:
    from cloney.asr.dummy import DummyASR
    from cloney.core.project import Project
    from cloney.engines.dummy import DummyEngine
    from cloney.pipeline import run_project

    project = Project.create(
        name="Ohne",
        text="Ein Satz.",
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
    )
    run_project(project, settings, voice_store, DummyEngine, DummyASR)

    # None heißt ehrlich 'nicht gemessen', nicht 'unähnlich'.
    assert all(c.speaker_similarity is None for c in project.chunks)
    assert project.median_similarity() is None


def test_ohne_schwelle_wird_nichts_markiert(settings: Settings, voice_store: VoiceStore) -> None:
    """Welchen Wert ein guter Klon erreicht, hängt an Modell und Aufnahme. Eine
    ungeprüfte Schwelle erzeugte Fehlalarme -- deshalb wird ohne Vorgabe nur
    gemessen."""
    from cloney.asr.dummy import DummyASR
    from cloney.core.project import ChunkStatus, Project
    from cloney.engines.dummy import DummyEngine
    from cloney.pipeline import run_project

    assert settings.similarity_threshold == 0.0
    project = Project.create(
        name="Schwelle",
        text="Ein Satz.",
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
    )

    class Fremd:
        """Liefert stets einen Vektor ohne Bezug zur Referenz."""

        def __init__(self) -> None:
            self._erster = True

        def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
            vektor = (
                np.array([1.0, 0.0], dtype=np.float32)
                if self._erster
                else np.array([0.0, 1.0], dtype=np.float32)
            )
            self._erster = False
            return vektor

        def close(self) -> None:
            return None

    run_project(project, settings, voice_store, DummyEngine, DummyASR, embedder_factory=Fremd)
    assert project.chunks[0].speaker_similarity == 0.0
    assert project.chunks[0].status == ChunkStatus.OK

    settings.similarity_threshold = 0.5
    project.chunks[0].status = ChunkStatus.OK
    from cloney.pipeline import check_speaker_similarity

    check_speaker_similarity(project, settings, voice_store, Fremd)
    assert project.chunks[0].status == ChunkStatus.NEEDS_REVIEW
