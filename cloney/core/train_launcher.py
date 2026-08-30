"""F5s Training starten, mit einer Worker-Zahl, die zur Maschine passt.

F5s Trainer legt den DataLoader mit sechzehn Worker-Prozessen an::

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed=None):

``finetune_cli.py`` reicht dazu nichts durch -- es ruft ``trainer.train(...)``
ohne ``num_workers`` auf, und PyTorch warnt dann selbst::

    This DataLoader will create 16 worker processes in total. Our suggested max
    number of worker in current system is 8

Auf Linux kostet das vor allem Kontextwechsel. Unter Windows startet jeder
Worker einen eigenen Interpreter samt Torch-Import -- sechzehn davon sind
Gigabyte an Arbeitsspeicher und eine Minute Anlauf, für nichts: mehr Worker als
Kerne laden keine Datei schneller.

Deshalb dieser Umweg. Er setzt genau eine Zahl und ruft dann F5s eigenes
Skript auf, unverändert. Geht am Patch etwas schief -- eine andere F5-Fassung,
eine geänderte Signatur --, läuft das Training trotzdem, nur eben mit F5s
Vorgabe. Ein Trainingslauf ist zu teuer, um an einer Bequemlichkeit zu
scheitern.
"""

from __future__ import annotations

import os
import sys

#: Mehr Worker als das bringen beim Laden von Audio nichts mehr.
MAX_WORKERS = 8


def cpu_count() -> int:
    """Kerne, die diesem Prozess tatsächlich zur Verfügung stehen.

    ``os.cpu_count()`` zählt die der Maschine; in einem Container ist das oft
    mehr, als der Prozess nutzen darf. Wo es die Zuteilung gibt, zählt sie.
    """
    zuteilung = getattr(os, "sched_getaffinity", None)
    if zuteilung is not None:
        return max(1, len(zuteilung(0)))
    return max(1, os.cpu_count() or 1)


def worker_count(kerne: int | None = None) -> int:
    """Wie viele Worker der DataLoader bekommen soll.

    Einer bleibt für den Hauptprozess frei -- er füttert die GPU, und wenn er
    auf einen Kern warten muss, wartet die GPU mit.
    """
    verfuegbar = kerne if kerne is not None else cpu_count()
    return max(1, min(MAX_WORKERS, verfuegbar - 1))


def patch_trainer(workers: int) -> bool:
    """F5s Trainer.train mit fester Worker-Zahl versehen. False, wenn es nicht ging."""
    try:
        from f5_tts.model.trainer import Trainer
    except Exception:  # noqa: BLE001 -- ohne F5 gibt es nichts zu patchen
        return False

    echt = Trainer.train

    def train(self, train_dataset, num_workers=workers, **kwargs):  # noqa: ANN001, ANN202
        return echt(self, train_dataset, num_workers=workers, **kwargs)

    Trainer.train = train
    return True


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    workers = worker_count()
    if patch_trainer(workers):
        print(f"DataLoader mit {workers} Worker-Prozessen ({cpu_count()} Kerne verfügbar).")

    from f5_tts.train.finetune_cli import main as f5_main

    sys.argv = ["finetune_cli", *args]
    f5_main()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
