"""Hintergrundläufe der Web-UI.

Ein Renderlauf dauert Minuten bis Stunden und darf die HTTP-Antwort nicht
blockieren. Er läuft deshalb in einem Thread; der Fortschritt wird nicht im
Speicher gehalten, sondern aus dem Manifest gelesen -- so zeigt die Oberfläche
immer den Zustand, der tatsächlich auf Platte steht, auch nach einem Neustart
des Servers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cloney.config import Settings
from cloney.core.compare import Comparison
from cloney.core.project import Project
from cloney.core.voices import VoiceStore
from cloney.engines.registry import create_engine
from cloney.pipeline import ProgressEvent, run_comparison, run_project


@dataclass
class Job:
    #: Kennung des Projekts oder des Vergleichs, zu dem dieser Lauf gehört.
    key: str
    running: bool = True
    message: str = "Wird vorbereitet"
    done: int = 0
    total: int = 0
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, event: ProgressEvent) -> None:
        with self._lock:
            self.message = event.message
            self.done = event.done
            self.total = event.total

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "running": self.running,
                "message": self.message,
                "done": self.done,
                "total": self.total,
                "error": self.error,
            }


class _Runner:
    """Hält höchstens einen laufenden Job je Kennung."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Job | None:
        with self._lock:
            return self._jobs.get(key)

    def is_running(self, key: str) -> bool:
        job = self.get(key)
        return bool(job and job.running)

    def _begin(self, key: str) -> tuple[Job, bool]:
        """Reserviert die Kennung. Zweiter Rückgabewert: ob schon einer lief."""
        with self._lock:
            existing = self._jobs.get(key)
            if existing and existing.running:
                return existing, True
            job = Job(key=key)
            self._jobs[key] = job
            return job, False

    def _spawn(self, job: Job, name: str, arbeit: Callable[[], None]) -> Job:
        def work() -> None:
            try:
                arbeit()
            except Exception as exc:  # noqa: BLE001 - der Fehler gehört in die UI
                job.error = str(exc)[:500]
            finally:
                job.running = False
                job.message = job.error or "Fertig"

        threading.Thread(target=work, daemon=True, name=name).start()
        return job


class JobRunner(_Runner):
    """Renderläufe einzelner Projekte."""

    def start(self, project_root: Path, asr_factory=None, embedder_factory=None) -> Job:  # noqa: ANN001
        project = Project.load(project_root)
        job, lief_schon = self._begin(project.id)
        if lief_schon:
            return job

        settings = self.settings
        voice_store = VoiceStore(settings.voices_dir)

        def arbeit() -> None:
            run_project(
                project,
                settings,
                voice_store,
                lambda: create_engine(project.engine, settings, project.engine_options),
                asr_factory,
                job.update,
                embedder_factory,
            )

        return self._spawn(job, f"render-{project.id}", arbeit)


class ComparisonRunner(_Runner):
    """Vergleichsläufe. Getrennt von den Projekten, damit ein Vergleich und ein
    Hörbuch sich nicht gegenseitig die Kennung wegnehmen."""

    def start(self, comparison_root: Path, asr_factory=None, embedder_factory=None) -> Job:  # noqa: ANN001
        comparison = Comparison.load(comparison_root)
        job, lief_schon = self._begin(comparison.id)
        if lief_schon:
            return job

        settings = self.settings
        voice_store = VoiceStore(settings.voices_dir)

        def arbeit() -> None:
            run_comparison(
                comparison,
                settings,
                voice_store,
                lambda options: create_engine(comparison.engine, settings, options),
                asr_factory,
                job.update,
                embedder_factory,
            )

        return self._spawn(job, f"vergleich-{comparison.id}", arbeit)
