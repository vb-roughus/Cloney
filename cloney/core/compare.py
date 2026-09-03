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

   Das setzt voraus, dass die Engine überhaupt einen Seed entgegennimmt. Wo sie
   das nicht tut (``EngineInfo.reproducible_seed``), bleibt der Vergleich
   brauchbar, aber ein kleiner Unterschied zwischen zwei Zeilen kann auch aus
   dem Zufall stammen -- Oberfläche und Kommandozeile sagen das dazu.
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
from cloney.engines.base import NEUTRAL, EngineInfo

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
    #: Trainierter Stand, gegen den gerendert wird. Leer = der Pretrain aus der
    #: Konfiguration.
    model: str = ""
    #: Emotionslage, gegen deren Aufnahme konditioniert wird. Leer = neutral.
    lage: str = ""
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

    @property
    def kennung(self) -> tuple:
        """Was diese Variante ausmacht -- ohne Beschriftung und Messwerte.

        Daran erkennt ein bearbeiteter Vergleich seine alten Zeilen wieder: was
        gleich bleibt, behält sein Ergebnis, statt noch einmal gerendert zu
        werden.
        """
        return (tuple(sorted(self.options.items())), self.model, self.lage)


class Comparison(BaseModel):
    id: str
    name: str
    created_at: str
    voice: str
    engine: str
    text: str
    #: Trainierte Stände, die verglichen werden. Leer = nur der Pretrain.
    models: list[str] = Field(default_factory=list)
    #: Emotionslagen, die verglichen werden. Leer = nur die neutrale.
    lagen: list[str] = Field(default_factory=list)
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
        models: list[str] | None = None,
        lagen: list[str] | None = None,
    ) -> Comparison:
        variants = pruefe_raster(build_variants(engine, grid, models=models, lagen=lagen))

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
            models=list(models or []),
            lagen=list(lagen or []),
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
            # Die Lage gehört zur Variante, nicht zum einzelnen Satz: verglichen
            # wird eine Aufnahme gegen eine andere, nicht ein Satz gegen den
            # nächsten.
            chunk.lage = variant.lage
        project.engine_options = dict(variant.options)
        project.save()

        variant.project_id = project.id
        self.save()
        return project

    def reconfigure(
        self,
        *,
        name: str,
        text: str,
        voice: str,
        engine: EngineInfo,
        grid: dict[str, list[float]],
        models: list[str] | None = None,
        lagen: list[str] | None = None,
    ) -> dict[str, int]:
        """Einen bestehenden Vergleich ändern, statt einen neuen anzulegen.

        Ein Vergleich ist selten beim ersten Versuch richtig zugeschnitten: ein
        Wert fehlt, ein Regler war die falsche Frage, die Probe zu lang. Bisher
        hieß das, alles noch einmal einzugeben und die schon gerenderten
        Varianten wegzuwerfen.

        Was bleiben darf, entscheidet dieselbe Überlegung wie beim Projekt:

        * **Text, Stimme oder Engine gewechselt heißt alles neu.** Die Zahlen
          einer Zeile gelten für genau diese Probe an genau dieser Stimme;
          gemischt nebeneinandergestellt beantworteten sie keine Frage mehr.
        * **Sonst behält jede Zeile ihr Ergebnis**, die sich in Reglern, Modell
          und Lage nicht geändert hat. Einen vierten Wert nachzutragen kostet
          dann nur die eine neue Zeile.

        Gibt zurück, wie viele Varianten behalten, neu angelegt und verworfen
        wurden.
        """
        neue = pruefe_raster(build_variants(engine, grid, models=models, lagen=lagen))
        uebernehmbar = text == self.text and voice == self.voice and engine.name == self.engine

        vorhanden = {v.kennung: v for v in self.variants} if uebernehmbar else {}
        behalten = 0
        zusammengesetzt: list[Variant] = []
        for variante in neue:
            alt = vorhanden.pop(variante.kennung, None)
            if alt is None:
                zusammengesetzt.append(variante)
                continue
            # Die Beschriftung kommt aus dem neuen Raster: welche Regler eine
            # Zeile unterscheiden, hängt am ganzen Raster und kann sich mit ihm
            # geändert haben.
            zusammengesetzt.append(
                alt.model_copy(update={"slug": variante.slug, "label": variante.label})
            )
            behalten += 1

        # Was wegfällt, nimmt seinen erzeugten Ton mit. Er gehört zu einer Zeile,
        # die es nicht mehr gibt, und niemand käme je wieder an ihn heran.
        entfernt = len(self.variants) - behalten
        weg = vorhanden.values() if uebernehmbar else self.variants
        for verwaist in weg:
            self._verwerfen(verwaist)

        self.name = name.strip() or self.name
        self.text = text
        self.voice = voice
        self.engine = engine.name
        self.models = list(models or [])
        self.lagen = list(lagen or [])
        self.variants = zusammengesetzt
        self.save()
        return {"behalten": behalten, "neu": len(zusammengesetzt) - behalten, "entfernt": entfernt}

    def _verwerfen(self, variant: Variant) -> None:
        """Den Ordner einer weggefallenen Variante entfernen."""
        if variant.project_id:
            shutil.rmtree(self.variants_dir / variant.project_id, ignore_errors=True)

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


#: Weniger als zwei Zeilen sind kein Vergleich.
MIN_VARIANTS = 2


def pruefe_raster(variants: list[Variant]) -> list[Variant]:
    """Ein Raster mit weniger als zwei Zeilen zurückweisen -- mit dem Weg heraus.

    Seit die Regler mit ihrem Vorgabewert vorbelegt sind, entsteht das leicht
    aus Versehen: alle Achsen stehen auf einem Wert, und das Ergebnis wäre ein
    einzelner Lauf, der nichts vergleicht. Die Meldung sagt deshalb nicht nur,
    dass es zu wenig ist, sondern welche drei Achsen es gibt.
    """
    if len(variants) >= MIN_VARIANTS:
        return variants
    raise ValueError(
        "Für einen Vergleich braucht es mindestens zwei Varianten. Eine zweite "
        "entsteht durch einen zweiten Wert bei einem Regler, eine zweite "
        "Emotionslage oder ein zweites Modell."
    )


def build_variants(
    engine: EngineInfo,
    grid: dict[str, list[float]],
    models: list[str] | None = None,
    lagen: list[str] | None = None,
) -> list[Variant]:
    """Kreuzprodukt der angegebenen Reglerwerte.

    Unbekannte Regler und Werte außerhalb der Grenzen fallen weg -- dieselbe
    Regel wie bei ``EngineInfo.clean_options``, damit ein Vergleich nichts
    misst, was so gar nicht eingestellt werden kann.
    """
    # Wie bei den Reglerwerten: doppelt genannte Stände ergäben zwei gleiche
    # Zeilen, die nur Rechenzeit kosten. Die Reihenfolge bleibt, wie sie
    # angegeben wurde -- sie bestimmt die Reihenfolge der Zeilen.
    modelle = list(dict.fromkeys(models or []))
    # Dieselbe Regel für die Lagen: doppelt genannt ergäbe zwei gleiche Zeilen.
    stimmlagen = list(dict.fromkeys(lagen or []))
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

    # Ohne Regler ist der Vergleich trotzdem sinnvoll, sobald Modell oder Lage
    # mehr als einen Wert haben -- dann ist eben das die Achse.
    staende = list(modelle or [])
    if not achsen and len(staende) < 2 and len(stimmlagen) < 2:
        return []

    #: Nur was sich tatsächlich ändert, gehört in die Beschriftung. Ändert sich
    #: kein Regler, wären alle Zeilen gleich benannt -- dann müssen alle hinein.
    #: Es sei denn, Modell oder Lage unterscheiden die Zeilen ohnehin.
    andere_achsen = len(staende) > 1 or len(stimmlagen) > 1
    variabel = {key for key, werte in achsen if len(werte) > 1}
    if not variabel and not andere_achsen:
        variabel = {key for key, _ in achsen}

    kombinationen = list(itertools.product(*(werte for _, werte in achsen))) or [()]
    variants: list[Variant] = []
    vergeben: set[str] = set()
    for modell in staende or [""]:
        for lage in stimmlagen or [""]:
            for kombination in kombinationen:
                options = {key: wert for (key, _), wert in zip(achsen, kombination, strict=True)}
                teile = [
                    f"{_label(engine, key)} {_zahl(engine, key, wert)}"
                    for key, wert in options.items()
                    if key in variabel
                ]
                if len(stimmlagen) > 1:
                    teile.insert(0, lage or NEUTRAL)
                if len(staende) > 1:
                    teile.insert(0, modell or "Pretrain")
                label = " · ".join(teile) or (modell or "Pretrain")
                variants.append(
                    Variant(
                        slug=_eindeutig(_slugify(label), vergeben),
                        label=label,
                        options=options,
                        model=modell,
                        lage=lage,
                    )
                )
                if len(variants) == MAX_VARIANTS:
                    return variants
    return variants


def _eindeutig(slug: str, vergeben: set[str]) -> str:
    """Ein Slug, den es im Vergleich noch nicht gibt.

    Er wird aus der Beschriftung gebildet und auf vierzig Zeichen gekürzt. Mit
    drei Achsen werden Beschriftungen lang, und zwei können hinter der Kürzung
    gleich aussehen -- dann zeigten die Tonspur und der Weg zum Projekt beide
    auf dieselbe Zeile.
    """
    kandidat, zaehler = slug, 1
    while kandidat in vergeben:
        zaehler += 1
        kandidat = f"{slug}-{zaehler}"
    vergeben.add(kandidat)
    return kandidat


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
