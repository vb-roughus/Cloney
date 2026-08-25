"""Projekt-Manifest: der Zustand eines Renderlaufs, vollständig auf Platte.

Jeder Pipeline-Schritt schreibt sofort ins Manifest. Damit ist jeder Abbruch
resumierbar und jeder Chunk einzeln neu renderbar -- und dieselbe Datenstruktur
bedient CLI, Web-UI und Wiederaufnahme, ohne dass es einen zweiten Zustand gibt,
der auseinanderlaufen könnte.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from cloney.core.segment import build_chunks

_MANIFEST = "project.json"
_SLUG = re.compile(r"[^a-z0-9]+")


class ChunkStatus(StrEnum):
    PENDING = "pending"
    SYNTHESIZED = "synthesized"
    OK = "ok"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class Chunk(BaseModel):
    index: int
    raw_text: str
    normalized_text: str
    ends_paragraph: bool = False
    seed: int
    status: ChunkStatus = ChunkStatus.PENDING
    audio_file: str | None = None
    asr_text: str | None = None
    cer: float | None = None
    attempts: int = 0
    engine: str | None = None
    error: str | None = None

    @property
    def needs_synthesis(self) -> bool:
        return self.status in (ChunkStatus.PENDING, ChunkStatus.FAILED)


class Project(BaseModel):
    id: str
    name: str
    created_at: str
    voice: str
    engine: str
    sample_rate: int
    source_text: str
    chunks: list[Chunk] = Field(default_factory=list)
    output_file: str | None = None
    #: Ordner des Projekts. Nicht Teil des Manifests -- er ergibt sich aus dem Ort.
    root: Path = Field(default=Path("."), exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    # -- Erzeugen und Laden ------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        name: str,
        text: str,
        voice: str,
        engine: str,
        sample_rate: int,
        projects_dir: Path,
        chars_per_second: float = 14.0,
        target_seconds: float = 20.0,
        max_seconds: float = 25.0,
    ) -> Project:
        project_id = _make_id(name)
        root = projects_dir / project_id
        root.mkdir(parents=True, exist_ok=True)

        chunks = [
            Chunk(
                index=i,
                raw_text=c.raw_text,
                normalized_text=c.normalized_text,
                ends_paragraph=c.ends_paragraph,
                seed=derive_seed(project_id, i, 0),
            )
            for i, c in enumerate(build_chunks(text, chars_per_second, target_seconds, max_seconds))
        ]

        project = cls(
            id=project_id,
            name=name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            voice=voice,
            engine=engine,
            sample_rate=sample_rate,
            source_text=text,
            chunks=chunks,
            root=root,
        )
        project.save()
        return project

    @classmethod
    def load(cls, root: Path) -> Project:
        project = cls.model_validate_json((root / _MANIFEST).read_text(encoding="utf-8"))
        project.root = root
        return project

    @classmethod
    def list_all(cls, projects_dir: Path) -> list[Project]:
        if not projects_dir.exists():
            return []
        found = [cls.load(d) for d in sorted(projects_dir.iterdir()) if (d / _MANIFEST).exists()]
        return sorted(found, key=lambda p: p.created_at, reverse=True)

    def save(self) -> None:
        """Atomar schreiben, damit ein Abbruch nie ein halbes Manifest hinterlässt."""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / _MANIFEST
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, target)

    # -- Pfade -------------------------------------------------------------

    @property
    def chunks_dir(self) -> Path:
        return self.root / "chunks"

    def chunk_path(self, index: int) -> Path:
        return self.chunks_dir / f"chunk_{index:04d}.wav"

    @property
    def output_path(self) -> Path:
        return self.root / "output.wav"

    # -- Abfragen ----------------------------------------------------------

    def chunk(self, index: int) -> Chunk:
        return self.chunks[index]

    def pending_synthesis(self) -> list[Chunk]:
        return [c for c in self.chunks if c.needs_synthesis]

    def pending_qc(self) -> list[Chunk]:
        return [c for c in self.chunks if c.status == ChunkStatus.SYNTHESIZED]

    def flagged(self) -> list[Chunk]:
        return [
            c for c in self.chunks if c.status in (ChunkStatus.NEEDS_REVIEW, ChunkStatus.FAILED)
        ]

    @property
    def is_complete(self) -> bool:
        return bool(self.chunks) and all(c.status == ChunkStatus.OK for c in self.chunks)

    @property
    def progress(self) -> tuple[int, int]:
        done = sum(1 for c in self.chunks if c.status in (ChunkStatus.OK, ChunkStatus.NEEDS_REVIEW))
        return done, len(self.chunks)

    def median_cer(self) -> float | None:
        values = sorted(c.cer for c in self.chunks if c.cer is not None)
        if not values:
            return None
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    def reroll(self, index: int) -> Chunk:
        """Neuer Seed für einen Chunk -- die Grundlage des 'Neu würfeln' in der UI."""
        chunk = self.chunks[index]
        chunk.attempts += 1
        chunk.seed = derive_seed(self.id, index, chunk.attempts)
        chunk.status = ChunkStatus.PENDING
        chunk.asr_text = None
        chunk.cer = None
        chunk.error = None
        return chunk

    def retext(self, index: int, raw_text: str) -> Chunk:
        """Text eines Chunks ersetzen und neu normalisieren."""
        from cloney.core.normalize import normalize_german

        chunk = self.chunks[index]
        chunk.raw_text = raw_text
        chunk.normalized_text = normalize_german(raw_text)
        return self.reroll(index)


def derive_seed(project_id: str, index: int, attempt: int) -> int:
    """Reproduzierbarer Seed. Gleiches Projekt + Chunk + Versuch = gleiches Audio."""
    digest = hashlib.sha1(f"{project_id}:{index}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _make_id(name: str) -> str:
    slug = _SLUG.sub("-", name.lower()).strip("-")[:40] or "projekt"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slug}"
