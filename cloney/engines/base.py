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


class EngineError(RuntimeError):
    """Fehler beim Erzeugen von Audio. Die Meldung ist für Menschen bestimmt
    und landet unverändert im Manifest und in der Oberfläche."""


@dataclass(frozen=True)
class EngineOption:
    """Ein Regler, den eine Engine anbietet.

    Welche Stellschrauben es gibt, ist von Engine zu Engine verschieden. Statt
    das in der Oberfläche als Sonderfall zu führen, beschreibt jede Engine ihre
    Regler selbst -- die Oberfläche zeigt einfach, was da ist.
    """

    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    default: float
    help: str = ""
    integer: bool = False

    def clamp(self, value: float) -> float:
        value = max(self.minimum, min(self.maximum, float(value)))
        return float(int(round(value))) if self.integer else value


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
    #: Gesamtbudget einer einzelnen Generierung in Sekunden, Referenz eingerechnet.
    #: None heißt: die Engine setzt keine harte Grenze.
    max_generation_seconds: float | None = None
    #: Länge, auf die die Engine die Referenz zurechtschneidet.
    max_reference_seconds: float | None = None
    #: Regler, die diese Engine anbietet.
    options: tuple[EngineOption, ...] = ()

    def option(self, key: str) -> EngineOption | None:
        return next((o for o in self.options if o.key == key), None)

    def clean_options(self, values: dict[str, float] | None) -> dict[str, float]:
        """Nimmt nur bekannte Regler an und hält sie in ihren Grenzen."""
        if not values:
            return {}
        gereinigt: dict[str, float] = {}
        for key, value in values.items():
            regler = self.option(key)
            if regler is None:
                continue
            try:
                gereinigt[key] = regler.clamp(float(value))
            except (TypeError, ValueError):
                continue
        return gereinigt

    def chunk_budget_seconds(self, reference_seconds: float, fallback: float) -> float:
        """Wie lang ein Chunk höchstens sein darf, damit die Engine ihn am Stück erzeugt.

        Modelle wie F5-TTS teilen zu lange Eingaben selbst auf und blenden die
        Teile ineinander. Das Ergebnis klingt zwar, aber ein Chunk enthielte dann
        Nähte, die sich nicht mehr einzeln nachbessern lassen -- und genau das
        einzelne Nachbessern ist der Zweck der Chunk-Aufteilung. Deshalb wird hier
        so zugeschnitten, dass die Engine nie selbst teilen muss.
        """
        if self.max_generation_seconds is None:
            return fallback
        reference = min(reference_seconds, self.max_reference_seconds or reference_seconds)
        usable = self.max_generation_seconds - reference
        # Unter vier Sekunden wird die Aufteilung sinnlos kleinteilig; dann ist
        # eher die Referenzaufnahme zu lang.
        return max(4.0, min(fallback, usable))


@dataclass(frozen=True)
class VoiceRef:
    """Referenzstimme. Bleibt über das gesamte Projekt unverändert.

    Genau hier verhindert Cloney den Voice-Drift: jeder Chunk wird gegen dieselbe
    Referenz konditioniert, niemals gegen das Ergebnis des Vorgängers.
    """

    name: str
    audio_path: Path
    transcript: str = ""
    #: Länge der Referenzaufnahme. Geht in die Chunk-Planung ein, siehe
    #: EngineInfo.chunk_budget_seconds.
    duration_s: float = 0.0


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
