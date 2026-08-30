"""Die Zählung hinter der Startseite.

Geprüft wird hier die Arithmetik, nicht die Seite: sie ist der Teil, der falsch
sein kann, und sie lässt sich ohne HTTP festnageln.
"""

from __future__ import annotations

from cloney.config import Settings
from cloney.core.project import ChunkStatus, Project
from cloney.core.voices import VoiceStore
from cloney.engines.dummy import DummyEngine
from cloney.web.overview import summarize


def _projekt(settings: Settings, name: str, text: str) -> Project:
    return Project.create(
        name=name,
        text=text,
        voice="test-stimme",
        engine=DummyEngine.info,
        projects_dir=settings.projects_dir,
        target_seconds=1.5,
    )


def test_leere_uebersicht_traegt_die_anleitung() -> None:
    uebersicht = summarize([])

    assert uebersicht.leer
    assert uebersicht.projects == 0


def test_eine_einzige_stimme_genuegt_gegen_die_anleitung() -> None:
    """Wer eine Stimme angelegt hat, ist über den ersten Schritt hinaus --
    die Seite soll dann Kennzahlen zeigen, nicht wieder bei null anfangen."""
    assert not summarize([], voices=1).leer


def test_saetze_werden_ueber_alle_projekte_gezaehlt(
    settings: Settings, voice_store: VoiceStore
) -> None:
    eins = _projekt(settings, "Eins", "Erster Satz. Zweiter Satz.")
    zwei = _projekt(settings, "Zwei", "Dritter Satz.")
    eins.chunks[0].status = ChunkStatus.OK
    zwei.chunks[0].status = ChunkStatus.OK

    uebersicht = summarize([eins, zwei])

    assert (uebersicht.chunks_done, uebersicht.chunks_total) == (2, 3)
    assert uebersicht.complete == 1
    assert uebersicht.in_arbeit == 1


def test_zur_durchsicht_zaehlt_auffaellige_und_gescheiterte(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Die einzige Zahl auf der Seite, aus der unmittelbar Arbeit folgt --
    und beides gehört hinein: geprüft werden muss so oder so von Hand."""
    project = _projekt(settings, "Eins", "Erster Satz. Zweiter Satz. Dritter Satz.")
    project.chunks[0].status = ChunkStatus.NEEDS_REVIEW
    project.chunks[1].status = ChunkStatus.FAILED
    project.chunks[2].status = ChunkStatus.OK

    assert summarize([project]).review == 2


def test_bestandszahlen_werden_durchgereicht() -> None:
    uebersicht = summarize([], voices=2, comparisons=3, models=1, running=1)

    assert (uebersicht.voices, uebersicht.comparisons) == (2, 3)
    assert (uebersicht.models, uebersicht.running) == (1, 1)
