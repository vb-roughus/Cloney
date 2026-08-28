"""Vergleichslauf: dieselbe Textprobe, mehrere Reglerstellungen, eine Tabelle.

Die Regler einer Engine lassen sich nicht aus der Beschreibung ableiten. Welches
Sprechtempo, welche Zahl an Schritten und welche Führungsstärke zu einer
bestimmten Stimme passen, zeigt sich erst am Ohr -- und bisher hieß das: raten,
rendern, hören, wieder raten. Ein Vergleichslauf macht daraus eine Messung. Er
rendert eine kurze Probe einmal je Reglerstellung und stellt die Ergebnisse
nebeneinander: Fehlerrate, Stimmähnlichkeit, Dauer, dazu die Tonspuren.

Zwei Entscheidungen tragen das:

1. **Jede Variante ist ein vollwertiges Projekt.** Sie liegt im Ordner des
   Vergleichs und durchläuft dieselbe Pipeline wie ein Hörbuch. Damit misst der
   Vergleich, was später auch tatsächlich passiert -- es gibt keinen zweiten,
   abweichenden Renderweg, der auseinanderlaufen könnte.
2. **Alle Varianten teilen sich dieselben Seeds.** Sie werden aus der Kennung
   des Vergleichs abgeleitet, nicht aus der des Projekts. Sonst unterschieden
   sich zwei Varianten in zwei Dingen zugleich -- Regler und Zufall -- und die
   Tabelle beantwortete nicht mehr die gestellte Frage. Aus demselben Grund
   läuft ein Vergleich ohne Wiederholungsversuche: ein neuer Seed nach einem
   auffälligen Satz würde genau das verwischen, was gemessen werden soll.
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from cloney.core.project import Project, derive_seed
from cloney.engines.base import EngineInfo

_MANIFEST = "comparison.json"
_SLUG = re.compile(r"[^a-z0-9]+")

#: Obergrenze für ein Raster. Der Vergleich rendert die Probe je Variante einmal;
#: ein versehentliches Kreuzprodukt aus drei vollen Reglern liefe sonst stundenlang.
MAX_VARIANTS = 12


class VariantStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Variant(BaseModel):
    """Eine Reglerstellung samt Messergebnis."""

    slug: str
    label: str
    options: dict[str, float] = Field(default_factory=dict)
    #: Kennung des Projekts, das diese Variante gerendert hat. Leer = noch keins.
    project_id: str = ""
    status: VariantStatus = VariantStatus.PENDING
    median_cer: float | None = None
    median_similarity: float | None = None
    duration_s: float | None = None
    error: str | None = None

    @property
    def is_done(self) -> bool:
        return self.status == VariantStatus.DONE


class Comparison(BaseModel):
    id: str
    name: str
    created_at: str
    voice: str
    engine: str
    text: str
    variants: list[Variant] = Field(default_factory=list)
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
        grid: dict[str, list[float]],
        comparisons_dir: Path,
    ) -> Comparison:
        variants = build_variants(engine, grid)
        if not variants:
            raise ValueError("Kein Raster: es wurde kein einziger Reglerwert angegeben.")

        comparison_id = _make_id(name)
        root = comparisons_dir / comparison_id
        root.mkdir(parents=True, exist_ok=True)

        comparison = cls(
            id=comparison_id,
            name=name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            voice=voice,
            engine=engine.name,
            text=text,
            variants=variants,
            root=root,
        )
        comparison.save()
        return comparison

    @classmethod
    def load(cls, root: Path) -> Comparison:
        comparison = cls.model_validate_json((root / _MANIFEST).read_text(encoding="utf-8"))
        comparison.root = root
        return comparison

    @classmethod
    def resolve(cls, comparisons_dir: Path, comparison_id: str) -> Path:
        """Kennung zu Ordner -- und zwar nur zu einem darunterliegenden.

        Dieselbe Absicherung wie bei den Projekten: die Kennung kommt aus einer
        URL, und am Ende steht ein rekursives Löschen.
        """
        root = (comparisons_dir / comparison_id).resolve()
        if root.parent != comparisons_dir.resolve():
            raise ValueError(f"Ungültige Vergleichskennung: {comparison_id!r}")
        return root

    @classmethod
    def list_all(cls, comparisons_dir: Path) -> list[Comparison]:
        if not comparisons_dir.exists():
            return []
        found = [cls.load(d) for d in sorted(comparisons_dir.iterdir()) if (d / _MANIFEST).exists()]
        return sorted(found, key=lambda c: c.created_at, reverse=True)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / _MANIFEST
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, target)

    # -- Varianten ---------------------------------------------------------

    @property
    def variants_dir(self) -> Path:
        return self.root / "varianten"

    def variant(self, slug: str) -> Variant:
        found = next((v for v in self.variants if v.slug == slug), None)
        if found is None:
            raise KeyError(f"Variante '{slug}' gibt es in diesem Vergleich nicht")
        return found

    def variant_root(self, slug: str) -> Path:
        variant = self.variant(slug)
        if not variant.project_id:
            raise KeyError(f"Für Variante '{slug}' wurde noch kein Projekt angelegt")
        return self.variants_dir / variant.project_id

    def variant_project(self, slug: str) -> Project:
        return Project.load(self.variant_root(slug))

    def prepare(self, slug: str, engine: EngineInfo, reference_seconds: float) -> Project:
        """Legt das Projekt einer Variante an -- oder gibt das vorhandene zurück.

        Die Seeds kommen aus der Kennung des Vergleichs. Nur so unterscheiden
        sich zwei Varianten allein in der Reglerstellung.
        """
        variant = self.variant(slug)
        if (
            variant.project_id
            and (self.variants_dir / variant.project_id / "project.json").exists()
        ):
            return self.variant_project(slug)

        project = Project.create(
            name=f"{self.name} -- {variant.label}",
            text=self.text,
            voice=self.voice,
            engine=engine,
            projects_dir=self.variants_dir,
            reference_seconds=reference_seconds,
        )
        for chunk in project.chunks:
            chunk.seed = derive_seed(self.id, chunk.index, 0)
        project.engine_options = dict(variant.options)
        project.save()

        variant.project_id = project.id
        self.save()
        return project

    def record(self, slug: str, project: Project) -> Variant:
        """Übernimmt die Messwerte eines fertigen Variantenlaufs ins Manifest."""
        from cloney.core.audio import duration_seconds, read_wav

        variant = self.variant(slug)
        variant.median_cer = project.median_cer()
        variant.median_similarity = project.median_similarity()
        variant.duration_s = None
        if project.output_file and project.output_path.exists():
            audio, rate = read_wav(project.output_path)
            variant.duration_s = duration_seconds(audio, rate)
        variant.status = VariantStatus.DONE
        variant.error = None
        self.save()
        return variant

    def fail(self, slug: str, message: str) -> Variant:
        variant = self.variant(slug)
        variant.status = VariantStatus.FAILED
        variant.error = message[:500]
        self.save()
        return variant

    # -- Abfragen ----------------------------------------------------------

    @property
    def progress(self) -> tuple[int, int]:
        done = sum(
            1 for v in self.variants if v.status in (VariantStatus.DONE, VariantStatus.FAILED)
        )
        return done, len(self.variants)

    @property
    def is_complete(self) -> bool:
        return bool(self.variants) and all(v.status == VariantStatus.DONE for v in self.variants)

    def chars_per_second(self, slug: str) -> float | None:
        """Sprechtempo der fertigen Spur. Die Zahl, an der 'zu schnell' hängt."""
        variant = self.variant(slug)
        if not variant.duration_s:
            return None
        return len(self.text) / variant.duration_s

    def best_cer(self) -> set[str]:
        """Slugs der Varianten mit der niedrigsten Fehlerrate."""
        return _best(self.variants, "median_cer", niedriger_ist_besser=True)

    def best_similarity(self) -> set[str]:
        """Slugs der Varianten mit der höchsten Ähnlichkeit zur Referenzstimme."""
        return _best(self.variants, "median_similarity", niedriger_ist_besser=False)

    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _best(variants: list[Variant], feld: str, *, niedriger_ist_besser: bool) -> set[str]:
    """Spitzenreiter einer Spalte.

    Bewusst je Spalte und nicht als Gesamtnote: wie eine halbe Prozent
    Fehlerrate gegen zwei Hundertstel Ähnlichkeit aufzuwiegen wäre, ist eine
    Gewichtung, die niemand belegen kann. Der Vergleich zeigt die Zahlen und
    markiert je Spalte den Besten -- die Abwägung bleibt beim Menschen.

    Bei Gleichstand werden alle Gleichauf markiert; sind es alle, wird nichts
    markiert. Eine Auszeichnung, die jede Zeile trägt, sagt nichts aus, und eine,
    die bei Gleichstand willkürlich die erste Zeile trifft, sagt etwas Falsches.
    """
    kandidaten = [v for v in variants if v.is_done and getattr(v, feld) is not None]
    if not kandidaten:
        return set()
    wahl = min if niedriger_ist_besser else max
    bestwert = wahl(getattr(v, feld) for v in kandidaten)
    spitze = {v.slug for v in kandidaten if getattr(v, feld) == bestwert}
    return set() if len(spitze) == len(kandidaten) else spitze


def build_variants(engine: EngineInfo, grid: dict[str, list[float]]) -> list[Variant]:
    """Kreuzprodukt der angegebenen Reglerwerte.

    Unbekannte Regler und Werte außerhalb der Grenzen fallen weg -- dieselbe
    Regel wie bei ``EngineInfo.clean_options``, damit ein Vergleich nichts
    misst, was so gar nicht eingestellt werden kann.
    """
    achsen: list[tuple[str, list[float]]] = []
    for option in engine.options:
        werte = grid.get(option.key)
        if not werte:
            continue
        # Doppelte Werte ergäben zwei gleiche Varianten -- und damit zwei
        # identische Zeilen, die nur Rechenzeit kosten.
        eindeutig: list[float] = []
        for wert in werte:
            geklemmt = option.clamp(wert)
            if geklemmt not in eindeutig:
                eindeutig.append(geklemmt)
        achsen.append((option.key, eindeutig))

    if not achsen:
        return []

    #: Nur die Regler, die sich tatsächlich ändern, gehören in die Beschriftung.
    variabel = {key for key, werte in achsen if len(werte) > 1} or {key for key, _ in achsen}

    variants: list[Variant] = []
    for kombination in itertools.product(*(werte for _, werte in achsen)):
        options = {key: wert for (key, _), wert in zip(achsen, kombination, strict=True)}
        label = " · ".join(
            f"{_label(engine, key)} {_zahl(engine, key, wert)}"
            for key, wert in options.items()
            if key in variabel
        )
        variants.append(Variant(slug=_slugify(label), label=label, options=options))
        if len(variants) == MAX_VARIANTS:
            break
    return variants


def _label(engine: EngineInfo, key: str) -> str:
    option = engine.option(key)
    return option.label if option else key


def _zahl(engine: EngineInfo, key: str, wert: float) -> str:
    option = engine.option(key)
    return f"{wert:.0f}" if option and option.integer else f"{wert:g}"


def _slugify(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")[:40] or "variante"


def _make_id(name: str) -> str:
    slug = _SLUG.sub("-", name.lower()).strip("-")[:40] or "vergleich"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slug}"
