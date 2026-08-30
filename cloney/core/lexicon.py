"""Das Aussprache-Wörterbuch: was Regeln nicht wissen können.

Deutsche Normalisierung ist berechenbar -- aus ``3.`` wird ``dritten``, und das
gilt immer. Wie ``SWIFT`` klingen soll, ist dagegen keine Rechnung: es hängt an
der Herkunft des Worts und daran, wie geläufig es im Deutschen ist. Zwei
Menschen können es verschieden wollen, und beide haben recht.

Deshalb steht es hier nicht als Regel, sondern als Eintrag: **Wort ->
Sprechweise**, geführt von dem, der den Text kennt. Die Ersetzung geschieht vor
allem anderen, damit die Sprechweise selbst noch normalisiert wird -- wer
``MP3`` als ``Em-Pe-3`` einträgt, bekommt am Ende ``Em-Pe-drei``.

Das Wörterbuch gilt über Projekte hinweg: wer ein Fachbuch in Kapiteln liest,
trägt einen Begriff einmal ein, nicht je Kapitel.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

_DATEI = "aussprache.json"


class Lexicon(BaseModel):
    """Wort zu Sprechweise. Die Reihenfolge der Einträge spielt keine Rolle --
    ersetzt wird immer der längste passende Eintrag zuerst."""

    entries: dict[str, str] = Field(default_factory=dict)

    # -- Laden und Speichern ----------------------------------------------

    @classmethod
    def path(cls, data_dir: Path) -> Path:
        return data_dir / _DATEI

    @classmethod
    def load(cls, data_dir: Path) -> Lexicon:
        pfad = cls.path(data_dir)
        if not pfad.exists():
            return cls()
        return cls.model_validate_json(pfad.read_text(encoding="utf-8"))

    def save(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        ziel = self.path(data_dir)
        tmp = ziel.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, ziel)

    # -- Pflegen -----------------------------------------------------------

    def set(self, wort: str, sprechweise: str) -> None:
        wort = wort.strip()
        if not wort:
            raise ValueError("Das Wort darf nicht leer sein")
        if not sprechweise.strip():
            raise ValueError(f"Zu '{wort}' fehlt die Sprechweise")
        self.entries[wort] = sprechweise.strip()

    def remove(self, wort: str) -> bool:
        return self.entries.pop(wort, None) is not None

    def sorted_entries(self) -> list[tuple[str, str]]:
        return sorted(self.entries.items(), key=lambda p: p[0].lower())

    # -- Anwenden ----------------------------------------------------------

    def apply(self, text: str) -> str:
        """Einträge im Text ersetzen.

        Verglichen wird ohne Rücksicht auf Groß- und Kleinschreibung: im Text
        steht mal ``SWIFT``, mal ``Swift``, und gemeint ist beide Male dasselbe.
        Eingesetzt wird die Sprechweise so, wie sie eingetragen wurde.
        """
        if not self.entries:
            return text
        return _muster(self.entries).sub(
            lambda m: self.entries[_schluessel(self.entries, m.group(0))], text
        )


def _muster(entries: dict[str, str]) -> re.Pattern[str]:
    # Längste zuerst: sonst schlüge 'SWIFT' bei 'SWIFT-Code' schon zu, bevor der
    # längere Eintrag zum Zug käme.
    keys = sorted(entries, key=len, reverse=True)
    # \b trägt bei Einträgen, die mit einem Wortzeichen beginnen und enden --
    # bei allen anderen (etwa '&') fiele die Grenze auf die falsche Seite.
    teile = [
        rf"\b{re.escape(k)}\b" if k[:1].isalnum() and k[-1:].isalnum() else re.escape(k)
        for k in keys
    ]
    return re.compile("|".join(teile), re.IGNORECASE)


def _schluessel(entries: dict[str, str], treffer: str) -> str:
    if treffer in entries:
        return treffer
    klein = treffer.lower()
    return next(k for k in entries if k.lower() == klein)
