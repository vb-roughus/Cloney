"""Trainierte Modelle: was aus einem Finetune herauskommt, benutzbar machen.

Ein Trainingslauf hinterlässt Checkpoints in F5s Ordnern -- ``model_last.pt``
und ``model_<schritt>.pt``. Damit ist das Modell noch nicht *verwendbar*: die
Engine liest ihren Checkpoint aus der Konfiguration, und eine Konfiguration hat
Platz für genau einen.

Hier bekommt jeder trainierte Stand einen Namen und einen Eintrag. Zwei Dinge
folgen daraus:

* Rendern lässt sich damit gegen einen bestimmten Stand, nicht nur gegen den
  Pretrain.
* Der Vergleichslauf kann Stände gegeneinander stellen -- Pretrain gegen
  Finetune, oder Schritt 4000 gegen Schritt 12000. Genau das ist die Frage, die
  ein Finetune aufwirft und die man sonst nur nach Gefühl beantwortet.

Das Vokabular gehört zum Checkpoint. Es wird deshalb mit eingetragen: ein
Finetune ist auf dem Vokabular seines Pretrains trainiert, und mit einem anderen
passt die Embedding-Matrix nicht zu den Gewichten.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

_META = "model.json"
_SLUG = re.compile(r"[^a-z0-9]+")


class ModelError(RuntimeError):
    """Die Meldung ist für Menschen."""


class TrainedModel(BaseModel):
    name: str
    #: Pfad des Checkpoints. Absolut, weil er außerhalb von Cloney liegt.
    ckpt_path: str
    #: Vokabular des Pretrains, auf dem trainiert wurde.
    vocab_path: str
    created_at: str
    #: Woher der Stand stammt -- Datensatz, Schrittzahl, was der Mensch notiert.
    note: str = ""

    @property
    def exists(self) -> bool:
        return Path(self.ckpt_path).exists() and Path(self.vocab_path).exists()


class ModelStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, name: str) -> Path:
        return self.root / slug(name)

    def resolve(self, name: str) -> Path:
        """Name zu Ordner -- und nur zu einem darunterliegenden."""
        ordner = self.path(name).resolve()
        if ordner.parent != self.root.resolve():
            raise ValueError(f"Ungültiger Modellname: {name!r}")
        return ordner

    def exists(self, name: str) -> bool:
        return (self.path(name) / _META).exists()

    def add(self, name: str, ckpt: Path, vocab: Path, note: str = "") -> TrainedModel:
        for pfad, was in ((ckpt, "Checkpoint"), (vocab, "Vokabular")):
            if not pfad.exists():
                raise ModelError(f"{was} nicht gefunden: {pfad}")

        modell = TrainedModel(
            name=name,
            ckpt_path=str(pfad_absolut(ckpt)),
            vocab_path=str(pfad_absolut(vocab)),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            note=note,
        )
        ordner = self.path(name)
        ordner.mkdir(parents=True, exist_ok=True)
        ziel = ordner / _META
        tmp = ziel.with_suffix(".json.tmp")
        tmp.write_text(modell.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, ziel)
        return modell

    def get(self, name: str) -> TrainedModel:
        pfad = self.path(name) / _META
        if not pfad.exists():
            raise ModelError(f"Modell '{name}' gibt es nicht")
        return TrainedModel.model_validate_json(pfad.read_text(encoding="utf-8"))

    def list_all(self) -> list[TrainedModel]:
        if not self.root.exists():
            return []
        gefunden = [
            TrainedModel.model_validate_json((d / _META).read_text(encoding="utf-8"))
            for d in sorted(self.root.iterdir())
            if (d / _META).exists()
        ]
        return sorted(gefunden, key=lambda m: m.created_at, reverse=True)

    def delete(self, name: str) -> None:
        """Nur den Eintrag entfernen. Der Checkpoint selbst bleibt liegen --
        er gehört F5, ist Gigabyte groß, und ihn beim Aufräumen einer Liste
        mitzulöschen wäre eine böse Überraschung."""
        import shutil

        shutil.rmtree(self.resolve(name), ignore_errors=True)


def pfad_absolut(pfad: Path) -> Path:
    return pfad if pfad.is_absolute() else pfad.resolve()


def settings_for(model: TrainedModel | None, settings):  # noqa: ANN001, ANN201
    """Einstellungen, die auf diesen trainierten Stand zeigen.

    Kein neuer Weg in die Engine, sondern derselbe wie bisher: die Engine liest
    Checkpoint und Vokabular aus der Konfiguration, und hier wird eine Kopie mit
    anderen Werten übergeben. Ohne Modell bleibt alles, wie es ist -- dann gilt
    der Pretrain aus der Konfiguration.
    """
    if model is None:
        return settings
    if not model.exists:
        raise ModelError(
            f"Zu '{model.name}' fehlt eine Datei: {model.ckpt_path} oder {model.vocab_path}. "
            "Wurde der Checkpoint verschoben oder gelöscht?"
        )
    return settings.model_copy(
        update={"f5_ckpt_path": model.ckpt_path, "f5_vocab_path": model.vocab_path}
    )


def find_checkpoints(directory: Path) -> list[Path]:
    """Checkpoints eines Trainingslaufs, jüngster Schritt zuerst.

    ``model_last.pt`` steht vorn: es ist der Stand, den man zuerst hören will.
    """
    if not directory.exists():
        return []
    letzte = [p for p in directory.glob("model_last.pt")]
    nummeriert = sorted(
        (p for p in directory.glob("model_*.pt") if p.name != "model_last.pt"),
        key=_schritt,
        reverse=True,
    )
    return letzte + nummeriert


def _schritt(pfad: Path) -> int:
    try:
        return int(pfad.stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


def slug(name: str) -> str:
    return _SLUG.sub("-", name.lower()).strip("-")[:40] or "modell"
