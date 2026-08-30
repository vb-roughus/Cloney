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
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from cloney.core.lexicon import Lexicon
from cloney.core.segment import build_chunks, spoken_form
from cloney.engines.base import EngineInfo

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
    #: Titel oder Kapitelüberschrift. Wird für sich gesprochen und
    #: bekommt beim Zusammenbau eine längere Pause.
    is_heading: bool = False
    seed: int
    status: ChunkStatus = ChunkStatus.PENDING
    audio_file: str | None = None
    asr_text: str | None = None
    cer: float | None = None
    attempts: int = 0
    engine: str | None = None
    error: str | None = None
    #: Sekunden Referenz-Vorspann, die am Anfang entfernt wurden.
    trimmed_bleed_s: float | None = None
    #: Ähnlichkeit zur Referenzstimme, 1.0 = identisch. None = nicht gemessen.
    speaker_similarity: float | None = None

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
    #: Tatsächlich verwendete Chunk-Länge. Kann unter dem Wunschwert liegen, wenn
    #: die Engine eine Obergrenze je Generierung hat -- siehe EngineInfo.
    target_chunk_seconds: float = 20.0
    #: Reglerstellung der Engine. Gehört ins Manifest, damit ein Lauf auch
    #: nachträglich reproduzierbar bleibt.
    engine_options: dict[str, float] = Field(default_factory=dict)
    #: Trainierter Stand, gegen den gerendert wird. Leer = der Pretrain aus der
    #: Konfiguration.
    model: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    output_file: str | None = None
    #: Warum die Stimmähnlichkeit nicht gemessen wurde. Steht im Manifest und
    #: nicht nur im Joblog, sonst stünde nach einem Neustart eine leere Spalte
    #: ohne Erklärung da.
    similarity_note: str | None = None
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
        engine: EngineInfo,
        projects_dir: Path,
        model: str = "",
        reference_seconds: float = 0.0,
        chars_per_second: float = 14.0,
        target_seconds: float = 20.0,
        max_seconds: float = 25.0,
        lexicon: Lexicon | None = None,
    ) -> Project:
        project_id = _make_id(name)
        root = projects_dir / project_id
        root.mkdir(parents=True, exist_ok=True)

        budget, roh = _plan_chunks(
            text, engine, reference_seconds, chars_per_second, target_seconds, max_seconds, lexicon
        )
        chunks = [
            Chunk(
                index=i,
                raw_text=c.raw_text,
                normalized_text=c.normalized_text,
                ends_paragraph=c.ends_paragraph,
                is_heading=c.is_heading,
                seed=derive_seed(project_id, i, 0),
            )
            for i, c in enumerate(roh)
        ]

        project = cls(
            id=project_id,
            name=name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            voice=voice,
            engine=engine.name,
            sample_rate=engine.sample_rate,
            source_text=text,
            target_chunk_seconds=budget,
            model=model,
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
    def resolve(cls, projects_dir: Path, project_id: str) -> Path:
        """Kennung zu Ordner -- und zwar nur zu einem darunterliegenden.

        Die Kennung kommt aus einer URL. Ohne diese Prüfung ließe sich mit
        '../..' aus dem Datenverzeichnis herausgreifen, was spätestens beim
        Löschen fatal wäre.
        """
        root = (projects_dir / project_id).resolve()
        if root.parent != projects_dir.resolve():
            raise ValueError(f"Ungültige Projektkennung: {project_id!r}")
        return root

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
        return _median(c.cer for c in self.chunks)

    def median_similarity(self) -> float | None:
        return _median(c.speaker_similarity for c in self.chunks)

    def reconfigure(
        self,
        *,
        text: str,
        voice: str,
        engine: EngineInfo,
        model: str | None = None,
        reference_seconds: float = 0.0,
        chars_per_second: float = 14.0,
        target_seconds: float = 20.0,
        max_seconds: float = 25.0,
        lexicon: Lexicon | None = None,
    ) -> dict[str, int]:
        """Text, Stimme oder Engine eines bestehenden Projekts ändern.

        Ein anderer Text heißt neu segmentieren, und damit wandern die
        Chunk-Grenzen. Trotzdem soll ein Tippfehler in Satz drei nicht die
        Arbeit an Satz siebzehn kosten: Chunks, deren Sprechfassung wörtlich
        gleich bleibt, behalten Ton, Seed und Messwerte. Verglichen wird die
        normalisierte Fassung, nicht der Rohtext -- wer nur die Schreibweise
        einer Zahl ändert, hört dasselbe und soll nicht neu rendern müssen.

        Stimme, Engine oder trainierter Stand gewechselt heißt dagegen: alles
        neu. Vorhandener Ton stammt dann von einem anderen Sprecher oder Modell,
        und ihn stehen zu lassen ergäbe eine Spur aus zwei Stimmen.

        Gibt zurück, wie viele Sätze behalten, neu angelegt und verworfen wurden.
        """
        # Kein Name heißt: der Stand bleibt, wie er ist. Sonst löschte ein
        # Aufrufer, der das Feld nicht kennt, stillschweigend den Finetune --
        # und das Projekt spräche danach mit einer anderen Stimme.
        stand = self.model if model is None else model
        uebernehmbar = voice == self.voice and engine.name == self.engine and stand == self.model
        budget, roh = _plan_chunks(
            text, engine, reference_seconds, chars_per_second, target_seconds, max_seconds, lexicon
        )

        frei: dict[str, list[Chunk]] = {}
        if uebernehmbar:
            for chunk in self.chunks:
                if chunk.audio_file and self.chunk_path(chunk.index).exists():
                    frei.setdefault(chunk.normalized_text, []).append(chunk)

        neue: list[Chunk] = []
        umzug: dict[int, int] = {}
        for i, c in enumerate(roh):
            passend = frei.get(c.normalized_text)
            alt = passend.pop(0) if passend else None
            if alt is None:
                neue.append(
                    Chunk(
                        index=i,
                        raw_text=c.raw_text,
                        normalized_text=c.normalized_text,
                        ends_paragraph=c.ends_paragraph,
                        is_heading=c.is_heading,
                        seed=derive_seed(self.id, i, 0),
                    )
                )
                continue
            uebernommen = alt.model_copy(
                update={
                    "index": i,
                    "raw_text": c.raw_text,
                    "ends_paragraph": c.ends_paragraph,
                    "is_heading": c.is_heading,
                    "audio_file": self.chunk_path(i).name,
                }
            )
            neue.append(uebernommen)
            umzug[alt.index] = i

        behalten = len(umzug)
        entfernt = len(self.chunks) - behalten
        self._move_chunk_audio(umzug)

        self.source_text = text
        self.voice = voice
        self.engine = engine.name
        self.model = stand
        self.sample_rate = engine.sample_rate
        self.target_chunk_seconds = budget
        # Regler, die die neue Engine nicht kennt, fallen weg statt still
        # weiterzuwirken.
        self.engine_options = engine.clean_options(self.engine_options)
        vorher = len(self.chunks)
        self.chunks = neue
        # Die fertige Spur gehört zum alten Satzbestand. Sie bleibt nur, wenn
        # sich an ihm nichts geändert hat -- sonst wäre sie eine Lüge. Dass sie
        # ein unverändertes Übernehmen übersteht, macht das Formular gefahrlos
        # wiederholbar.
        if not (behalten == len(neue) == vorher):
            self.output_path.unlink(missing_ok=True)
            self.output_file = None
        self.save()
        return {"behalten": behalten, "neu": len(neue) - behalten, "entfernt": entfernt}

    def _move_chunk_audio(self, umzug: dict[int, int]) -> None:
        """Tondateien auf die neuen Nummern umhängen.

        Der Umweg über Zwischennamen ist nötig, weil sich Quelle und Ziel
        überlappen können: Chunk 3 wird zu 2, während 2 zu 1 wird.
        """
        zwischen: dict[int, Path] = {}
        for alt_index in umzug:
            quelle = self.chunk_path(alt_index)
            ziel = self.chunks_dir / f"_umzug_{alt_index:04d}.wav"
            quelle.replace(ziel)
            zwischen[alt_index] = ziel

        for chunk in self.chunks:
            if chunk.index not in umzug.values():
                self.chunk_path(chunk.index).unlink(missing_ok=True)

        for alt_index, neu_index in umzug.items():
            zwischen[alt_index].replace(self.chunk_path(neu_index))

    def delete(self) -> None:
        """Projekt samt erzeugtem Ton entfernen."""
        shutil.rmtree(self.root, ignore_errors=True)

    def rename(self, name: str) -> None:
        """Nur die Anzeige ändern. Die Kennung bleibt, damit Pfade und die aus
        ihr abgeleiteten Seeds gültig bleiben."""
        self.name = name.strip() or self.name
        self.save()

    def duplicate(self, name: str, projects_dir: Path) -> Project:
        """Gleicher Text, neues Projekt -- ohne den erzeugten Ton.

        Der Weg, dieselbe Vorlage mit anderer Stimme, Engine oder Reglerstellung
        zu hören, ohne das Vorhandene zu verlieren.
        """
        kopie = Project.create(
            name=name,
            text=self.source_text,
            voice=self.voice,
            engine=_engine_info(self.engine),
            projects_dir=projects_dir,
            reference_seconds=0.0,
            target_seconds=self.target_chunk_seconds,
            max_seconds=self.target_chunk_seconds,
        )
        kopie.engine_options = dict(self.engine_options)
        kopie.save()
        return kopie

    def discard_audio(self, index: int) -> Chunk:
        """Erzeugten Ton eines Satzes verwerfen, Seed behalten.

        Anders als das Neuwürfeln: derselbe Seed mit veränderten Reglern ergibt
        ein anderes Ergebnis, und nur so lässt sich die Wirkung einer Einstellung
        an einem Satz beurteilen, ohne dass zugleich der Zufall wechselt.
        """
        chunk = self.chunks[index]
        self.chunk_path(index).unlink(missing_ok=True)
        chunk.audio_file = None
        chunk.status = ChunkStatus.PENDING
        chunk.asr_text = None
        chunk.cer = None
        chunk.error = None
        return chunk

    def discard_all_audio(self) -> int:
        """Ton aller Sätze verwerfen. Gibt zurück, wie viele betroffen waren."""
        betroffen = sum(1 for c in self.chunks if c.audio_file or c.status != ChunkStatus.PENDING)
        for chunk in self.chunks:
            self.discard_audio(chunk.index)
        self.output_path.unlink(missing_ok=True)
        self.output_file = None
        self.save()
        return betroffen

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

    def retext(self, index: int, raw_text: str, lexicon: Lexicon | None = None) -> Chunk:
        """Text eines Chunks ersetzen und neu normalisieren."""
        chunk = self.chunks[index]
        chunk.raw_text = raw_text
        chunk.normalized_text = spoken_form(raw_text, chunk.is_heading, lexicon)
        return self.reroll(index)

    def refresh_spoken(self, index: int, lexicon: Lexicon | None = None) -> bool:
        """Die Sprechfassung aus dem Rohtext neu erzeugen. True, wenn sie sich ändert.

        Die Sprechfassung steht im Manifest, seit der Satz angelegt wurde. Ein
        Eintrag im Aussprache-Wörterbuch, der später dazukommt, erreicht sie
        nicht von selbst -- wer danach einen Satz neu würfelt, hörte weiterhin
        die alte Fassung und suchte den Fehler im Modell.
        """
        chunk = self.chunks[index]
        neu = spoken_form(chunk.raw_text, chunk.is_heading, lexicon)
        if neu == chunk.normalized_text:
            return False
        chunk.normalized_text = neu
        return True

    def refresh_all_spoken(self, lexicon: Lexicon | None = None) -> list[int]:
        """Alle Sprechfassungen auffrischen. Gibt die geänderten Sätze zurück.

        Nur diese werden zum Neurendern vorgemerkt: was gleich klingt, muss
        nicht noch einmal erzeugt werden.
        """
        geaendert = [i for i in range(len(self.chunks)) if self.refresh_spoken(i, lexicon)]
        for index in geaendert:
            self.reroll(index)
        if geaendert:
            self.save()
        return geaendert


def _plan_chunks(
    text: str,
    engine: EngineInfo,
    reference_seconds: float,
    chars_per_second: float,
    target_seconds: float,
    max_seconds: float,
    lexicon: Lexicon | None = None,
) -> tuple[float, list]:
    """Chunk-Budget und Rohschnitt -- gemeinsam für Anlegen und Ändern.

    Engines wie F5-TTS erzeugen nur eine begrenzte Dauer am Stück und teilen
    längere Eingaben sonst selbst auf. Ein so entstandener Chunk enthielte
    Nähte, die sich nicht einzeln nachbessern lassen -- deshalb wird hier von
    vornherein kleiner geschnitten.
    """
    budget = engine.chunk_budget_seconds(reference_seconds, target_seconds)
    if budget < target_seconds:
        max_seconds = budget
    return budget, build_chunks(text, chars_per_second, budget, max_seconds, lexicon)


def _median(werte: Iterable[float | None]) -> float | None:
    vorhanden = sorted(w for w in werte if w is not None)
    if not vorhanden:
        return None
    mitte = len(vorhanden) // 2
    if len(vorhanden) % 2:
        return vorhanden[mitte]
    return (vorhanden[mitte - 1] + vorhanden[mitte]) / 2


def _engine_info(name: str) -> EngineInfo:
    from cloney.engines.registry import engine_info

    return engine_info(name)


def derive_seed(project_id: str, index: int, attempt: int) -> int:
    """Reproduzierbarer Seed. Gleiches Projekt + Chunk + Versuch = gleiches Audio."""
    digest = hashlib.sha1(f"{project_id}:{index}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _make_id(name: str) -> str:
    slug = _SLUG.sub("-", name.lower()).strip("-")[:40] or "projekt"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slug}"
