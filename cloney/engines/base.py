"""Gemeinsame Schnittstelle aller TTS-Engines.

Die Engines unterscheiden sich stark: Higgs v3 versteht Inline-Tags wie
``[freundlich]``, F5-TTS nicht; manche brauchen den Referenztext, andere nicht.
``EngineInfo`` macht diese Unterschiede zu Daten, damit die Pipeline sie behandeln
kann, ohne die konkrete Engine zu kennen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
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

    def steps(self) -> list[float]:
        """Alle einstellbaren Werte, von klein nach groß.

        Damit kann die Oberfläche einen Regler als Auswahlliste anbieten statt
        als Feld, in das eine Zahl getippt wird. Gerundet wird auf die
        Nachkommastellen der Schrittweite: 0.5 plus sechsmal 0.05 ergibt in
        Fließkomma 0.7999999999999999, und das stünde so in der Liste.
        """
        stellen = max(0, -Decimal(str(self.step)).as_tuple().exponent)
        anzahl = int(round((self.maximum - self.minimum) / self.step)) + 1
        werte = [round(self.minimum + i * self.step, stellen) for i in range(max(1, anzahl))]
        return [self.clamp(w) for w in werte]

    def formatiere(self, wert: float) -> str:
        """Ein Wert, wie er in der Oberfläche stehen soll."""
        return f"{wert:.0f}" if self.integer else f"{wert:g}"


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
    #: Liefert dieselbe Eingabe mit demselben Seed dasselbe Audio? Manche
    #: Engines laufen über eine Schnittstelle, die gar keinen Seed entgegennimmt
    #: -- dort würfelt jeder Aufruf neu. Das Neuwürfeln eines Satzes funktioniert
    #: weiterhin, das Wiederherstellen eines früheren Ergebnisses nicht. Wer das
    #: verschweigt, verspricht eine Reproduzierbarkeit, die es nicht gibt.
    reproducible_seed: bool = True
    #: Leitet die Engine ihr Sprechtempo aus der Referenzaufnahme ab? Dann ist
    #: ein zügig gesprochenes Vorbild der Grund für zügige Ausgabe, und die
    #: Oberfläche kann den passenden Reglerwert vorrechnen. Wieder ein
    #: Unterschied als Datum statt als Sonderfall in der Oberfläche.
    derives_tempo_from_reference: bool = False

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


#: Name der Lage, gegen die gerendert wird, solange nichts anderes gewählt ist.
#: Eine Stimme hat immer genau eine davon -- es ist ihre Hauptaufnahme.
NEUTRAL = "neutral"


@dataclass(frozen=True)
class VoiceRef:
    """Eine Referenzaufnahme, gegen die konditioniert wird.

    Hier verhindert Cloney den Voice-Drift: jeder Chunk wird gegen eine
    unveränderte Aufnahme konditioniert, niemals gegen das Ergebnis des
    Vorgängers. Welche Aufnahme das ist, steht im Manifest -- eine Stimme kann
    mehrere Lagen haben, und ein Satz wählt eine davon.
    """

    name: str
    audio_path: Path
    transcript: str = ""
    #: Länge der Referenzaufnahme. Geht in die Chunk-Planung ein, siehe
    #: EngineInfo.chunk_budget_seconds.
    duration_s: float = 0.0
    #: Emotionslage dieser Aufnahme. Für Anzeige und Meldungen -- die Engines
    #: sehen nur Ton und Wortlaut und müssen von Lagen nichts wissen.
    lage: str = NEUTRAL


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
