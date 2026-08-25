"""Sequenzielle Modell-Lebenszyklen.

Auf einer Karte mit 8 bis 16 GB dürfen Text-LLM, TTS-Modell und ASR nie
gleichzeitig im Speicher liegen. Cloney lädt deshalb pro Phase genau ein Modell
und gibt es danach wieder frei -- ``model_slot`` ist die einzige Stelle, an der
das passiert, damit kein Pfad das Freigeben vergessen kann.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


def free_gpu_memory() -> None:
    """Gibt belegten GPU-Speicher frei, sofern torch vorhanden ist."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@contextmanager
def model_slot(factory: Callable[[], T]) -> Iterator[T]:
    """Lädt ein Modell, gibt es nach Gebrauch garantiert wieder frei."""
    instance = factory()
    try:
        yield instance
    finally:
        close = getattr(instance, "close", None)
        if callable(close):
            close()
        free_gpu_memory()
