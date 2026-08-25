"""Gemeinsame Schnittstelle aller TTS-Engines.

Die Engines unterscheiden sich stark: Higgs v3 versteht Inline-Tags wie
``[freundlich]``, F5-TTS nicht; manche brauchen den Referenztext, andere nicht.
``EngineInfo`` macht diese Unterschiede zu Daten, damit die Pipeline sie behandeln
kann, ohne die konkrete Engine zu kennen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

_TAG = re.compile(r"\[([^\[\]]{1,40})\]")


@dataclass(frozen=True)
class EngineInfo:
    name: str
    #: Lizenz der Modellgewichte -- wird in der UI angezeigt, weil sie darüber
    #: entscheidet, wofür das Ergebnis verwendet werden darf.
    license: str
    vram_gb: float
    languages: tuple[str, ...]
    sample_rate: int
    requires_ref_text: bool
    #: Inline-Tags, die diese Engine versteht. Leer = die Engine kennt keine.
    supported_tags: frozenset[str] = frozenset()
    description: str = ""


@dataclass(frozen=True)
class VoiceRef:
    """Referenzstimme. Bleibt über das gesamte Projekt unverändert.

    Genau hier verhindert Cloney den Voice-Drift: jeder Chunk wird gegen dieselbe
    Referenz konditioniert, niemals gegen das Ergebnis des Vorgängers.
    """

    name: str
    audio_path: Path
    transcript: str = ""


@runtime_checkable
class TTSEngine(Protocol):
    info: EngineInfo

    def synthesize(self, text: str, voice: VoiceRef, seed: int) -> np.ndarray:
        """Erzeugt Mono-Float32-Audio in ``info.sample_rate``."""
        ...

    def close(self) -> None: ...


def strip_unsupported_tags(text: str, supported: frozenset[str]) -> str:
    """Entfernt Inline-Tags, die die aktive Engine nicht kennt.

    Damit muss der Director nicht wissen, welche Engine gerade läuft -- er
    annotiert frei, und die Pipeline räumt vor der Synthese auf.
    """

    def repl(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1).strip().lower() in supported else ""

    return " ".join(_TAG.sub(repl, text).split())


def find_tags(text: str) -> list[str]:
    return [m.group(1).strip().lower() for m in _TAG.finditer(text)]
