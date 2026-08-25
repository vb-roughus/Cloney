"""Hintergrundläufe der Web-UI.

Ein Renderlauf dauert Minuten bis Stunden und darf die HTTP-Antwort nicht
blockieren. Er läuft deshalb in einem Thread; der Fortschritt wird nicht im
Speicher gehalten, sondern aus dem Manifest gelesen -- so zeigt die Oberfläche
immer den Zustand, der tatsächlich auf Platte steht, auch nach einem Neustart
des Servers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from cloney.config import Settings
from cloney.core.project import Project
from cloney.core.voices import VoiceStore
from cloney.engines.registry import create_engine
from cloney.pipeline import ProgressEvent, run_project


@dataclass
class Job:
    project_id: str
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


class JobRunner:
    """Hält höchstens einen laufenden Job pro Projekt."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, project_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(project_id)

    def is_running(self, project_id: str) -> bool:
        job = self.get(project_id)
        return bool(job and job.running)

    def start(self, project_root: Path, asr_factory=None) -> Job:  # noqa: ANN001
        project = Project.load(project_root)
        with self._lock:
            existing = self._jobs.get(project.id)
            if existing and existing.running:
                return existing
            job = Job(project_id=project.id)
            self._jobs[project.id] = job

        settings = self.settings
        voice_store = VoiceStore(settings.voices_dir)

        def work() -> None:
            try:
                run_project(
                    project,
                    settings,
                    voice_store,
                    lambda: create_engine(project.engine, settings, project.engine_options),
                    asr_factory,
                    job.update,
                )
            except Exception as exc:  # noqa: BLE001 - der Fehler gehört in die UI
                job.error = str(exc)[:500]
            finally:
                job.running = False
                job.message = job.error or "Fertig"

        threading.Thread(target=work, daemon=True, name=f"render-{project.id}").start()
        return job
