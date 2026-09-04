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
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from cloney.core.lexicon import Lexicon
from cloney.core.segment import TextChunk, build_chunks, join_raw, spoken_form
from cloney.engines.base import NEUTRAL, EngineInfo

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
    #: Emotionslage dieses Satzes. Leer heißt: die des Projekts. Ein Satz trägt
    #: also nur, was von ihr abweicht -- wird die Vorgabe des Projekts später
    #: geändert, ziehen alle nicht abweichenden mit, und die drei von Hand
    #: gesetzten bleiben stehen.
    #:
    #: Manifeste von vor den Lagen bleiben damit gültig: dort ist überall leer,
    #: die Vorgabe des Projekts ebenfalls, und leer heißt dort wie hier neutral.
    lage: str = ""
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
    #: Emotionslage, die für alle Sätze gilt, die keine eigene tragen. Leer =
    #: neutral. Ein ganzes Kapitel ernst zu sprechen ist damit eine Einstellung
    #: und nicht hundert Klicks.
    lage: str = ""
    #: Ob am Satzbau von Hand gearbeitet wurde -- eingefügt, verschmolzen oder
    #: neu getextet. Von da an ist die Satzliste die Vorlage und nicht mehr der
    #: Schnitt des Quelltexts: ein frischer Schnitt machte die Arbeit zunichte.
    #: Siehe ``_schnitt_erhalten``.
    #:
    #: Manifeste von vor dieser Möglichkeit bleiben gültig: dort steht überall
    #: False, und das ist genau der Zustand, in dem nichts von Hand geschnitten
    #: wurde.
    handschnitt: bool = False
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

    @property
    def prototype_path(self) -> Path:
        """Der Zwischenstand als eine Spur -- bewusst nicht ``output.wav``.

        Ein Prototyp ist unvollständig. Ihn unter demselben Namen abzulegen
        machte aus einem Zwischenstand unbemerkt ein Ergebnis: er wäre über
        dieselbe Adresse zu laden und stünde in der Oberfläche an der Stelle,
        an der sonst die fertige Spur steht.
        """
        return self.root / "prototyp.wav"

    @property
    def prototype_stale(self) -> bool:
        """Ist seit dem Prototyp ein Satz dazugekommen oder neu erzeugt worden?

        Beantwortet über die Uhrzeiten der Dateien und nicht über einen eigenen
        Eintrag im Manifest: die Dateien wissen es genauer, und ein Eintrag
        müsste an jeder Stelle mitgepflegt werden, an der ein Satz entsteht.

        Ohne diese Frage wäre der unangenehmste Fehler möglich: ein Prototyp,
        der aussieht wie der aktuelle Stand und einer von vor zwanzig Sätzen
        ist. Man hörte einem Ergebnis nach, das es so nicht mehr gibt.
        """
        try:
            stand = self.prototype_path.stat().st_mtime_ns
        except OSError:
            return False
        for pfad in self.chunks_dir.glob("chunk_*.wav"):
            with suppress(OSError):
                if pfad.stat().st_mtime_ns > stand:
                    return True
        return False

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

        Ein von Hand geänderter Satzbau bleibt stehen, solange der Text derselbe
        ist -- siehe ``_schnitt_erhalten``. Ohne das nähme ein Wechsel der Stimme
        jedes Einfügen und Verschmelzen zurück.

        Gibt zurück, wie viele Sätze behalten, neu angelegt und verworfen wurden
        und ob dabei ein Handschnitt verlorenging.
        """
        # Kein Name heißt: der Stand bleibt, wie er ist. Sonst löschte ein
        # Aufrufer, der das Feld nicht kennt, stillschweigend den Finetune --
        # und das Projekt spräche danach mit einer anderen Stimme.
        stand = self.model if model is None else model
        uebernehmbar = voice == self.voice and engine.name == self.engine and stand == self.model
        budget, grenze = _budget(engine, reference_seconds, target_seconds, max_seconds)
        erhalten = self._schnitt_erhalten(text, grenze * chars_per_second)
        roh: list[TextChunk]
        if erhalten:
            roh = [
                TextChunk(c.raw_text, c.normalized_text, c.ends_paragraph, c.is_heading)
                for c in self.chunks
            ]
        else:
            roh = build_chunks(text, chars_per_second, budget, grenze, lexicon)

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
            self._verwerfe_spur()
        verloren = self.handschnitt and not erhalten
        self.handschnitt = erhalten
        self.save()
        return {
            "behalten": behalten,
            "neu": len(neue) - behalten,
            "entfernt": entfernt,
            "neu_geschnitten": verloren,
        }

    def _schnitt_erhalten(self, text: str, grenze_zeichen: float) -> bool:
        """Bleibt der bestehende Satzbau beim Übernehmen stehen?

        Von Hand eingefügte, verschmolzene und neu getextete Sätze sind Arbeit,
        die ein frischer Schnitt zunichte machte. Sie bleiben deshalb stehen,
        solange der Quelltext derselbe ist. Er wird nach jeder Änderung an den
        Sätzen aus ihnen geschrieben (``text_aus_chunks``) -- ein Unterschied
        heißt also: im Textfeld geändert, und dann gilt wieder der Text.

        Die Grenze der Engine sticht das trotzdem. Passt ein Satz nicht mehr in
        eine Generierung, teilte die Engine ihn selbst, mit einer Naht, die sich
        nicht einzeln nachbessern lässt. Genau davor bewahrt der eigene Schnitt,
        und dieser Schutz wiegt schwerer als die Handarbeit.
        """
        if not (self.handschnitt and self.chunks and text == self.source_text):
            return False
        return all(len(c.normalized_text) <= grenze_zeichen for c in self.chunks)

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
        self._verwerfe_spur()
        self.save()
        return betroffen

    def _verwerfe_spur(self) -> None:
        """Fertige Spur und Prototyp verwerfen.

        Beide gehören zu einem Satzbestand. Ändert der sich, sind beide
        überholt -- der Prototyp sogar auf die stillere Art: er trüge Sätze in
        sich, die es nicht mehr gibt, und keine Uhrzeit verriete das.
        """
        self.output_path.unlink(missing_ok=True)
        self.output_file = None
        self.prototype_path.unlink(missing_ok=True)

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

    def lage_of(self, chunk: Chunk) -> str:
        """Die Lage, gegen die dieser Satz tatsächlich konditioniert wird.

        Ein Satz trägt nur, was von der Vorgabe des Projekts abweicht. Die Frage
        lässt sich deshalb nicht am Satz allein beantworten -- sie geht immer
        über das Projekt.
        """
        return chunk.lage or self.lage or NEUTRAL

    def set_lage(self, index: int, lage: str) -> bool:
        """Die Emotionslage eines Satzes wechseln. True, wenn sie sich ändert.

        Ein leerer Wert heißt: wieder der Vorgabe des Projekts folgen.

        Ändert sie sich, wird der Ton verworfen und der Seed behalten. Anders
        als beim Neuwürfeln wechselt also nur die Referenzaufnahme -- derselbe
        Wurf, eine andere Lage. Nur so ist zu hören, was die Lage bewirkt, statt
        zugleich den Zufall zu bewegen.

        Ändert sie sich nicht, bleibt alles stehen. Zweimal dieselbe Lage zu
        wählen ist keine Änderung und darf keine Arbeit kosten -- beim Anwenden
        auf eine Auswahl trifft man sonst regelmäßig Sätze, die schon so
        stehen.
        """
        chunk = self.chunks[index]
        vorher = self.lage_of(chunk)
        chunk.lage = (lage or "").strip()
        if self.lage_of(chunk) == vorher:
            return False
        self.discard_audio(index)
        return True

    def set_lage_many(self, indices: Iterable[int], lage: str) -> int:
        """Eine Lage auf mehrere Sätze anwenden. Gibt zurück, wie viele sich ändern."""
        geaendert = sum(1 for index in indices if self.set_lage(index, lage))
        if geaendert:
            self.save()
        return geaendert

    def set_default_lage(self, lage: str) -> int:
        """Die Vorgabe des Projekts setzen. Gibt zurück, wie viele Sätze das trifft.

        Getroffen wird nur, wer keine eigene trägt -- von Hand gesetzte Lagen
        überleben den Wechsel. Deren Ton fällt weg, denn er stammt aus einer
        anderen Aufnahme; alles Übrige bleibt stehen.
        """
        gewaehlt = (lage or "").strip()
        if gewaehlt == self.lage:
            return 0
        betroffen = [c.index for c in self.chunks if not c.lage and c.audio_file]
        self.lage = gewaehlt
        for index in betroffen:
            self.discard_audio(index)
        self.save()
        return len(betroffen)

    def retext(self, index: int, raw_text: str, lexicon: Lexicon | None = None) -> Chunk:
        """Text eines Chunks ersetzen und neu normalisieren.

        Von hier an ist die Satzliste die Vorlage: der Quelltext wird aus ihr
        geschrieben. Stünde dort weiter die alte Fassung, nähme das nächste
        'Vorlage übernehmen' die Änderung wortlos zurück -- und der Text im
        Einstellungsfeld widerspräche dem, was gesprochen wird.
        """
        chunk = self.chunks[index]
        chunk.raw_text = raw_text
        chunk.normalized_text = spoken_form(raw_text, chunk.is_heading, lexicon)
        ergebnis = self.reroll(index)
        self.source_text = self.text_aus_chunks()
        self.handschnitt = True
        return ergebnis

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

    # -- Satzbau von Hand --------------------------------------------------

    def insert_chunk(
        self,
        index: int,
        raw_text: str,
        *,
        danach: bool = False,
        lexicon: Lexicon | None = None,
    ) -> int:
        """Einen neuen Satz vor oder nach diesem einsetzen. Gibt seine Nummer zurück.

        Der neue Satz tritt dem Absatz dessen bei, an dem er eingesetzt wird:
        davor heißt in dessen Absatz hinein, danach heißt an dessen Stelle am
        Absatzende. An den Absätzen hängt beim Zusammenbau die Pausenlänge,
        deshalb wird das hier entschieden und nicht dem Zufall überlassen.

        Auch die Emotionslage kommt von ihm. Ein Satz, der mitten in eine ernst
        gesprochene Passage gesetzt wird, fiele sonst als einziger in die
        Vorgabe des Projekts zurück.

        Eine Überschrift entsteht so nicht: sie ist eine Sache des Layouts --
        eine kurze Zeile für sich --, und die ist beim Eintippen eines Satzes
        nicht zu erkennen.
        """
        if not 0 <= index < len(self.chunks):
            raise ValueError("Diesen Satz gibt es nicht.")
        nachbar = self.chunks[index]
        roh = raw_text.strip()
        gesprochen = spoken_form(roh, False, lexicon)
        if not gesprochen:
            raise ValueError("Der neue Satz hat keinen Inhalt.")

        stelle = index + 1 if danach else index
        neuer = Chunk(
            index=stelle,
            raw_text=roh,
            normalized_text=gesprochen,
            ends_paragraph=nachbar.ends_paragraph if danach else False,
            lage=nachbar.lage,
            seed=derive_seed(self.id, stelle, 0),
        )
        if danach:
            nachbar.ends_paragraph = False

        folge = list(self.chunks)
        folge.insert(stelle, neuer)
        self._neu_nummerieren(folge)
        return stelle

    def merge_chunks(self, index: int, *, lexicon: Lexicon | None = None) -> int:
        """Diesen Satz mit dem folgenden verschmelzen. Gibt seine Nummer zurück.

        Der Weg zurück, wenn der eigene Schnitt zu fein war: zwei kurze Sätze in
        einem Zug gesprochen tragen die Betonung über die Grenze hinweg, die
        sonst zwischen zwei Generierungen liegt.

        Der Ton beider fällt weg -- er gehörte zu zwei getrennten Läufen, und
        aneinandergeklebt hörte man genau die Naht, deretwegen verschmolzen
        wurde. Der Seed wird aus der neuen Stelle abgeleitet: der Satz ist ein
        anderer als beide vorher.

        Wie die beiden Rohtexte zusammenkommen, entscheidet ``join_raw``.
        Überschrift bleibt der neue Satz nur, wenn es beide waren.

        Die Grenze der Engine wird hier **nicht** erzwungen. Wer zwei Sätze
        zusammenzieht, meint es; wird das Ergebnis zu lang, weist die Tabelle
        darauf hin, statt den Handgriff zu verweigern.
        """
        if not 0 <= index < len(self.chunks) - 1:
            raise ValueError("Nach diesem Satz kommt keiner mehr.")
        erster, zweiter = self.chunks[index], self.chunks[index + 1]
        roh = join_raw(erster.raw_text, zweiter.raw_text)
        titel = erster.is_heading and zweiter.is_heading
        verschmolzen = Chunk(
            index=index,
            raw_text=roh,
            normalized_text=spoken_form(roh, titel, lexicon),
            ends_paragraph=zweiter.ends_paragraph,
            is_heading=titel,
            lage=erster.lage,
            seed=derive_seed(self.id, index, 0),
        )
        self.discard_audio(index)
        self.discard_audio(index + 1)

        folge = list(self.chunks)
        folge[index : index + 2] = [verschmolzen]
        self._neu_nummerieren(folge)
        return index

    def text_aus_chunks(self) -> str:
        """Der Quelltext, wie ihn die Satzliste jetzt ergibt.

        Absatzgrenzen bleiben erhalten: an ihnen hängt beim Zusammenbau die
        Pausenlänge, und ein Titel wird nur wieder als solcher erkannt, wenn er
        für sich steht.
        """
        absaetze: list[str] = []
        laufend: list[str] = []
        for chunk in self.chunks:
            if chunk.raw_text.strip():
                laufend.append(chunk.raw_text.strip())
            if chunk.ends_paragraph and laufend:
                absaetze.append(" ".join(laufend))
                laufend = []
        if laufend:
            absaetze.append(" ".join(laufend))
        return "\n\n".join(absaetze)

    def _neu_nummerieren(self, folge: list[Chunk]) -> None:
        """Eine geänderte Satzfolge übernehmen: Nummern, Ton, Quelltext, Manifest.

        ``folge`` trägt die Chunks in der gewünschten Reihenfolge, jeder noch mit
        seiner bisherigen Nummer. Der Ton zieht mit -- ein eingefügter Satz darf
        die Aufnahmen aller folgenden nicht entwerten.
        """
        umzug = {c.index: i for i, c in enumerate(folge) if c.audio_file}
        for i, chunk in enumerate(folge):
            chunk.index = i
            if chunk.audio_file:
                chunk.audio_file = self.chunk_path(i).name
        self.chunks = folge
        self._move_chunk_audio(umzug)
        self._entferne_ueberzaehligen_ton()
        # Die fertige Spur gehörte zum alten Satzbestand und wäre jetzt eine Lüge.
        self._verwerfe_spur()
        self.source_text = self.text_aus_chunks()
        self.handschnitt = True
        self.save()

    def _entferne_ueberzaehligen_ton(self) -> None:
        """Tondateien jenseits des letzten Satzes entfernen.

        Nach einem Verschmelzen gibt es einen Satz weniger. Die Datei mit der
        höchsten Nummer gehört dann zu keinem mehr und läge beim nächsten
        Einfügen unter einem Satz, der sie nie erzeugt hat.
        """
        for pfad in self.chunks_dir.glob("chunk_*.wav"):
            try:
                nummer = int(pfad.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if nummer >= len(self.chunks):
                pfad.unlink(missing_ok=True)


def _budget(
    engine: EngineInfo,
    reference_seconds: float,
    target_seconds: float,
    max_seconds: float,
) -> tuple[float, float]:
    """Chunk-Budget und Obergrenze je Satz, in Sekunden.

    Engines wie F5-TTS erzeugen nur eine begrenzte Dauer am Stück und teilen
    längere Eingaben sonst selbst auf. Ein so entstandener Chunk enthielte
    Nähte, die sich nicht einzeln nachbessern lassen -- deshalb wird hier von
    vornherein kleiner geschnitten.
    """
    budget = engine.chunk_budget_seconds(reference_seconds, target_seconds)
    if budget < target_seconds:
        max_seconds = budget
    return budget, max_seconds


def _plan_chunks(
    text: str,
    engine: EngineInfo,
    reference_seconds: float,
    chars_per_second: float,
    target_seconds: float,
    max_seconds: float,
    lexicon: Lexicon | None = None,
) -> tuple[float, list]:
    """Chunk-Budget und Rohschnitt -- gemeinsam für Anlegen und Ändern."""
    budget, grenze = _budget(engine, reference_seconds, target_seconds, max_seconds)
    return budget, build_chunks(text, chars_per_second, budget, grenze, lexicon)


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
