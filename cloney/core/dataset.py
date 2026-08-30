"""Trainingsmaterial aus langen Aufnahmen.

Ein Finetune ist nur so gut wie sein Datensatz, und der Weg dorthin ist der
Teil, den kein Trainingsskript abnimmt: aus einer halben Stunde Vorlesen müssen
Segmente von wenigen Sekunden werden, jedes mit dem Wortlaut, den es tatsächlich
enthält.

Drei Entscheidungen tragen das:

1. **Geschnitten wird an Pausen, nie mitten im Klang.** Ein Segment, das mitten
   im Wort beginnt, bringt dem Modell einen Anfang bei, den es nachher
   produziert. Wo eine Pause fehlt, wird der Bereich verworfen statt zerteilt.

2. **Der Text durchläuft dieselbe Normalisierung wie bei der Synthese.** Die
   Spracherkennung schreibt "3. Mai", gesprochen wurde "dritten Mai". Trainiert
   werden muss auf der Form, die später auch hineingeht -- sonst lernt das
   Modell, Ziffern anders auszusprechen, als Cloney sie ihm vorlegt.

3. **Was durchfällt, wird benannt.** Jedes verworfene Segment steht mit Grund im
   Manifest. Ein Datensatz, der stillschweigend die Hälfte wegwirft, ist nicht
   von einem zu unterscheiden, bei dem die Aufnahme schlecht war.

Das Ergebnis liegt im Format, das F5-TTS erwartet: ein ``wavs``-Ordner und eine
``metadata.csv`` aus ``pfad|text``.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from cloney.asr.base import ASREngine
from cloney.core.audio import duration_seconds, peak_dbfs, read_wav, write_wav
from cloney.core.normalize import normalize_german
from cloney.core.voices import PLAUSIBLE_CHARS_PER_SECOND

_MANIFEST = "dataset.json"
_METADATA = "metadata.csv"
_WAVS = "wavs"
_SLUG = re.compile(r"[^a-z0-9]+")

#: Segmentlängen, mit denen F5-TTS trainiert wird. Kürzeres trägt zu wenig
#: Kontext, längeres sprengt die Speicherlänge je Beispiel.
MIN_SECONDS = 3.0
MAX_SECONDS = 15.0

#: Ab dieser Länge trennt eine Lücke zwei Äußerungen voneinander.
MIN_PAUSE_MS = 320

#: Bis zu dieser Lücke wird ein zu kurzer Abschnitt mit seinem Nachbarn zu einem
#: Segment zusammengefasst, statt weggeworfen zu werden.
#:
#: Beim Vorlesen dauert eine Kommapause zwei bis vier Zehntel, eine Satzpause
#: bis etwa acht -- so weit gehören zwei Abschnitte noch zusammen, und die Pause
#: dazwischen ist genau die, die auch im fertigen Hörbuch stünde. Darüber
#: beginnt eine Zäsur; sie ins Segment zu holen brächte dem Modell eine Pause
#: bei, die es später von sich aus macht.
MERGE_GAP_MS = 800

#: Ab dieser Länge taugt eine Lücke als Schnittstelle *innerhalb* eines zu lang
#: geratenen Bereichs. Bewusst niedriger: ein Atemzug reicht als Schnitt, wenn
#: die Alternative ist, zwanzig Sekunden brauchbare Sprache wegzuwerfen -- als
#: Grenze zwischen zwei Äußerungen wäre er zu wenig.
SPLIT_PAUSE_MS = 180

#: Abstand über dem gemessenen Grundpegel, ab dem ein Frame als Sprache zählt.
#:
#: Eine feste Schwelle wie -40 dBFS setzt eine leise Aufnahme voraus. Liegt der
#: Raumton darüber -- bei einer normalisierten oder in einem lebendigen Zimmer
#: entstandenen Aufnahme keine Seltenheit --, ist plötzlich *nichts* mehr still,
#: und die ganze Lesung gilt als ein einziger Bereich ohne Pause. Gemessen an
#: synthetischen Aufnahmen mit Raumton von -70 bis -30 dBFS findet dieser
#: Abstand durchgehend genau die echten Pausen; 14 dB zerfasert bereits die
#: Silbenlücken, 6 dB ist unnötig knapp.
SILENCE_MARGIN_DB = 10.0

#: Höher darf die selbst bestimmte Schwelle nicht liegen -- sonst schneidet eine
#: durchweg laute Aufnahme mitten in leisen Wörtern.
MAX_SILENCE_DB = -20.0

#: Abstand unter dem Sprechpegel, ab dem eine Stelle als Pause gilt.
#:
#: Die Schwelle allein am Grundpegel festzumachen reicht nicht. Eine Aufnahme
#: kann irgendwo eine sehr leise Stelle haben -- am Rand, an einer Schnittkante --
#: und trotzdem Sprechpausen enthalten, die viel höher liegen, weil in ihnen
#: geatmet wird oder der Raum nachklingt. Gemessen an einer echten Lesung:
#: Grundpegel -63 dBFS, Sprechpegel -12 dBFS, die Pausen aber erst ab -50
#: aufwärts. Grund+10 ergab -53 und fand genau eine Pause auf 42 Sekunden;
#: Sprech-25 ergibt -37 und findet fünf.
SPEECH_DROP_DB = 25.0

#: Und nicht tiefer. Ohne diese Grenze kippt die Rechnung ins Gegenteil: eine
#: bearbeitete Aufnahme mit exakt stillen Stellen ergibt einen Grundpegel von
#: -240 dBFS, eine Schwelle von -230 -- und damit gilt jedes Rauschen als
#: Sprache, sodass die ganze Lesung ein einziger Block ohne Pause wird.
MIN_SILENCE_DB = -55.0

#: Darunter ist ein Frame nicht leise, sondern numerisch null. Solche Frames
#: stammen vom Schnittprogramm, nicht aus dem Zimmer, und taugen nicht zur
#: Schätzung des Grundpegels.
DIGITAL_ZERO_DB = -100.0

#: Unterschreitet der Abstand zwischen leisen und lauten Stellen diesen Wert,
#: enthält die Aufnahme keine brauchbare Stille.
MIN_DYNAMIC_RANGE_DB = 12.0

#: Bleibt selbst der lauteste Teil darunter, ist auf der Spur nichts drauf.
SILENCE_FLOOR_DB = -60.0

#: Rand, der an beiden Enden eines Segments stehen bleibt. Ohne ihn klingt der
#: Einsatz abgehackt, mit zu viel wird jedes Beispiel vorn und hinten zäh.
MARGIN_MS = 60


class Utterance(BaseModel):
    index: int
    file: str
    #: Sprechfassung -- normalisiert wie bei der Synthese.
    text: str
    #: Was die Spracherkennung wörtlich geliefert hat, vor der Normalisierung.
    raw_text: str
    duration_s: float
    peak_dbfs: float
    source: str = ""

    @property
    def chars_per_second(self) -> float:
        """Sprechtempo auf dem Wortlaut, nicht auf der Sprechfassung.

        Die Normalisierung bläht den Text auf -- aus "3. Mai 2024" werden
        36 Zeichen. Auf der Sprechfassung gerechnet sähe jede Aufnahme mit
        Ziffern zu schnell aus, und die Zahl wäre eine andere als die, mit der
        die Eingangsprüfung arbeitet.
        """
        return len(self.raw_text) / self.duration_s if self.duration_s else 0.0


class Rejection(BaseModel):
    """Ein verworfener Abschnitt samt Grund. Gehört ins Manifest, nicht ins Log."""

    source: str
    start_s: float
    duration_s: float
    reason: str


class Dataset(BaseModel):
    name: str
    created_at: str
    sample_rate: int
    utterances: list[Utterance] = Field(default_factory=list)
    rejected: list[Rejection] = Field(default_factory=list)
    root: Path = Field(default=Path("."), exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    # -- Laden und Speichern ------------------------------------------------

    @classmethod
    def load(cls, root: Path) -> Dataset:
        dataset = cls.model_validate_json((root / _MANIFEST).read_text(encoding="utf-8"))
        dataset.root = root
        return dataset

    @classmethod
    def resolve(cls, datasets_dir: Path, name: str) -> Path:
        """Name zu Ordner -- und nur zu einem darunterliegenden.

        Schon der Slug entschärft Pfadangaben, weil er alles außer Buchstaben
        und Ziffern ersetzt; diese Prüfung ist die zweite Linie und stellt
        sicher, dass eine spätere Änderung am Slug nicht unbemerkt ein Löschen
        außerhalb des Datenverzeichnisses erlaubt.
        """
        root = (datasets_dir / slug(name)).resolve()
        if root.parent != datasets_dir.resolve():
            raise ValueError(f"Ungültiger Datensatzname: {name!r}")
        return root

    @classmethod
    def list_all(cls, datasets_dir: Path) -> list[Dataset]:
        if not datasets_dir.exists():
            return []
        gefunden = [cls.load(d) for d in sorted(datasets_dir.iterdir()) if (d / _MANIFEST).exists()]
        return sorted(gefunden, key=lambda d: d.created_at, reverse=True)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        ziel = self.root / _MANIFEST
        tmp = ziel.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, ziel)

    def write_metadata(self) -> Path:
        """``metadata.csv`` im Format von F5-TTS: ``wavs/datei.wav|Text``.

        Geschrieben wird über das csv-Modul, damit ein Text mit Anführungszeichen
        oder Zeilenumbruch die Datei nicht zerreißt.
        """
        ziel = self.root / _METADATA
        with ziel.open("w", encoding="utf-8", newline="") as datei:
            schreiber = csv.writer(datei, delimiter="|", quoting=csv.QUOTE_MINIMAL)
            for utterance in self.utterances:
                schreiber.writerow([utterance.file, utterance.text])
        return ziel

    # -- Kennzahlen ---------------------------------------------------------

    @property
    def wavs_dir(self) -> Path:
        return self.root / _WAVS

    @property
    def total_seconds(self) -> float:
        return sum(u.duration_s for u in self.utterances)

    @property
    def rejected_seconds(self) -> float:
        return sum(r.duration_s for r in self.rejected)

    def statistik(self) -> dict[str, float]:
        laengen = sorted(u.duration_s for u in self.utterances)
        tempi = sorted(u.chars_per_second for u in self.utterances)
        return {
            "segmente": len(self.utterances),
            "minuten": self.total_seconds / 60.0,
            "median_laenge_s": _median(laengen),
            "median_zeichen_pro_s": _median(tempi),
            "verworfen": len(self.rejected),
            "verworfene_minuten": self.rejected_seconds / 60.0,
        }


# -- Segmentierung ----------------------------------------------------------


def find_segments(
    audio: np.ndarray,
    sample_rate: int,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    silence_db: float | None = None,
    min_pause_ms: int = MIN_PAUSE_MS,
    split_pause_ms: int = SPLIT_PAUSE_MS,
    levels: list[LevelReport] | None = None,
    force_split: bool = False,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
    """Sprachbereiche zwischen den Pausen finden.

    Gibt die brauchbaren Bereiche zurück und daneben die verworfenen samt Grund.
    Zu lange Bereiche werden an ihrer längsten inneren Pause geteilt; findet sich
    keine, wird der Bereich verworfen -- ein harter Schnitt mitten im Wort brächte
    dem Modell einen Anfang bei, den es später produziert.

    ``silence_db=None`` bestimmt die Schwelle aus der Aufnahme selbst. Das ist
    der Normalfall: eine feste Schwelle scheitert an allem, was lauter rauscht,
    als sie annimmt.
    """
    frame = max(1, sample_rate // 100)
    if audio.size < frame:
        return [], []

    rahmen = audio[: len(audio) - len(audio) % frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(rahmen.astype(np.float64) ** 2, axis=1))
    with np.errstate(divide="ignore"):
        pegel = 20.0 * np.log10(np.maximum(rms, 1e-12))

    grundpegel, sprechpegel = silence_levels(pegel)
    hat_digitale_stille = float(np.mean(pegel <= DIGITAL_ZERO_DB)) >= _DIGITAL_SILENCE_SHARE
    if levels is not None:
        # Der Aufrufer will die gemessenen Pegel sehen -- sie sagen mehr über
        # die Aufnahme als jede Zahl, die wir selbst gewählt hätten.
        levels[:] = [LevelReport(grundpegel, sprechpegel, hat_digitale_stille)]
    if silence_db is None:
        silence_db = threshold_for(grundpegel, sprechpegel)
        if sprechpegel < SILENCE_FLOOR_DB:
            # Nichts zu hören. Kein Befund, sondern schlicht keine Aufnahme.
            return [], []
        # Enthält die Aufnahme exakt stille Stellen, ist die Frage nach der
        # Dynamik erledigt: dann gibt es Pausen, und zwar unbestreitbare. Der
        # Grundpegel ist dort aus dem Sprachanteil geschätzt und entsprechend
        # hoch -- das darf keine Beschwerde auslösen.
        if not hat_digitale_stille and sprechpegel - grundpegel < MIN_DYNAMIC_RANGE_DB:
            # Zwischen leise und laut liegt fast nichts: entweder durchgehend
            # gesprochen oder stark komprimiert. Ein Schnitt wäre geraten.
            return [], [
                (
                    0,
                    len(audio),
                    f"keine Pausen erkennbar -- kaum Unterschied zwischen leisen und "
                    f"lauten Stellen ({grundpegel:.0f} bis {sprechpegel:.0f} dBFS)",
                )
            ]
    laut = pegel > silence_db

    pause_frames = max(1, int(min_pause_ms / 10))
    schnitt_frames = max(1, int(split_pause_ms / 10))
    bereiche = _verschmelze(
        _speech_runs(laut, pause_frames),
        frame_seconds=frame / sample_rate,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        luecke_frames=max(1, int(MERGE_GAP_MS / 10)),
    )

    rand = int(sample_rate * MARGIN_MS / 1000)
    gut: list[tuple[int, int]] = []
    schlecht: list[tuple[int, int, str]] = []
    for i, (start_f, ende_f) in enumerate(bereiche):
        start = max(0, start_f * frame - rand)
        ende = min(len(audio), ende_f * frame + rand)
        _teile(
            audio,
            sample_rate,
            start,
            ende,
            laut,
            frame,
            schnitt_frames,
            min_seconds,
            max_seconds,
            gut,
            schlecht,
            pegel,
            force_split,
            _nachbarabstand(bereiche, i, frame / sample_rate),
        )
    return gut, schlecht


def _nachbarabstand(bereiche: list[tuple[int, int]], i: int, frame_seconds: float) -> float | None:
    """Wie weit der nächstgelegene Nachbar entfernt liegt, in Sekunden.

    Steht im Verwerfungsgrund eines zu kurzen Abschnitts. Er beantwortet die
    Frage, die sich sonst stellt: lag es an der Länge oder an der Lücke?
    """
    abstaende = [
        (bereiche[j][0] - bereiche[i][1]) if j > i else (bereiche[i][0] - bereiche[j][1])
        for j in (i - 1, i + 1)
        if 0 <= j < len(bereiche)
    ]
    return min(abstaende) * frame_seconds if abstaende else None


#: Fenster, über das der Grundpegel gesucht wird. Dieselbe Größenordnung wie
#: die kürzeste brauchbare Pause -- gesucht ist ja genau deren Pegel.
_FLOOR_WINDOW = 20

#: Ab diesem Anteil exakt stiller Frames gilt eine Aufnahme als geschnitten.
_DIGITAL_SILENCE_SHARE = 0.005


@dataclass(frozen=True)
class LevelReport:
    """Was die Pegel einer Aufnahme über sie aussagen."""

    floor_db: float
    speech_db: float
    #: Enthält die Aufnahme exakt stille Stellen? Dann stammt die Stille vom
    #: Schnittprogramm, und der Grundpegel ist aus dem Sprachanteil geschätzt.
    digital_silence: bool = False

    def beschreibung(self) -> str:
        """Ein Satzteil für die Ausgabe -- ohne eine Zahl zu nennen, die nichts bedeutet."""
        if self.digital_silence and self.floor_db > -30.0:
            # Der Grundpegel kommt hier aus der Sprache, nicht aus dem Zimmer.
            return "geschnittene Stille"
        if self.digital_silence:
            return f"Raumton {self.floor_db:.0f} dBFS, dazu geschnittene Stille"
        return f"Raumton {self.floor_db:.0f} dBFS"


def threshold_for(floor_db: float, speech_db: float) -> float:
    """Ab welchem Pegel eine Stelle in dieser Aufnahme als still gilt.

    Zwei Anker, und es gewinnt der höhere: ein Mindestabstand über dem
    Grundpegel, damit Rauschen nicht als Sprache zählt, und ein Mindestabstand
    unter dem Sprechpegel, damit eine Pause auch dann erkannt wird, wenn in ihr
    geatmet wird. Der zweite fehlte -- und ohne ihn blieb eine Lesung mit
    ruhigen Rändern und gefüllten Sprechpausen unzerlegbar.
    """
    return min(
        max(floor_db + SILENCE_MARGIN_DB, speech_db - SPEECH_DROP_DB, MIN_SILENCE_DB),
        MAX_SILENCE_DB,
    )


def silence_levels(pegel: np.ndarray) -> tuple[float, float]:
    """Grund- und Sprechpegel einer Aufnahme in dBFS.

    Der Grundpegel ist das Minimum der gleitenden Mediane: die leiseste Stelle,
    die *anhält*. Ein Perzentil taugt dafür nicht -- wer lange Passagen mit
    wenigen Pausen liest, hat so wenige stille Frames, dass schon das fünfte
    Perzentil mitten in der Sprache landet und den Raumton um zehn Dezibel zu
    hoch schätzt. Das gleitende Minimum trifft ihn über den ganzen Bereich von
    zwei bis siebzehn Prozent Pausenanteil auf ein halbes Dezibel genau.

    Der Sprechpegel ist das fünfundneunzigste Perzentil. Beide zusammen sagen,
    wo die Grenze zwischen still und gesprochen in *dieser* Aufnahme liegt --
    nicht in einer angenommenen.
    """
    if pegel.size == 0:
        return -120.0, -120.0

    # Exakt stille Frames stammen vom Schnittprogramm, nicht aus dem Zimmer.
    # Schon eine halbe Sekunde davon am Dateianfang zöge den Grundpegel auf
    # -240 dBFS und machte jede Schätzung wertlos.
    nutzbar = pegel[pegel > DIGITAL_ZERO_DB]
    if nutzbar.size == 0:
        return -120.0, -120.0

    if nutzbar.size < _FLOOR_WINDOW:
        grund = float(nutzbar.min())
    else:
        sicht = np.lib.stride_tricks.sliding_window_view(nutzbar, _FLOOR_WINDOW)
        grund = float(np.median(sicht, axis=1).min())
    return grund, float(np.percentile(pegel, 95))


def _speech_runs(laut: np.ndarray, pause_frames: int) -> list[tuple[int, int]]:
    """Zusammenhängende Sprachbereiche. Kurze Lücken zählen nicht als Grenze."""
    indizes = np.flatnonzero(laut)
    if indizes.size == 0:
        return []
    bereiche: list[list[int]] = [[int(indizes[0]), int(indizes[0]) + 1]]
    for i in indizes[1:]:
        i = int(i)
        if i - bereiche[-1][1] < pause_frames:
            bereiche[-1][1] = i + 1
        else:
            bereiche.append([i, i + 1])
    return [(a, b) for a, b in bereiche]


def _verschmelze(
    bereiche: list[tuple[int, int]],
    *,
    frame_seconds: float,
    min_seconds: float,
    max_seconds: float,
    luecke_frames: int,
) -> list[tuple[int, int]]:
    """Zu kurze Abschnitte mit ihrem Nachbarn zusammenfassen.

    Wer seine Aufnahmen selbst geschnitten hat, hat kurze Abschnitte -- ein
    Halbsatz, ein Einwurf, ein Name. Einzeln fallen sie durch die Mindestlänge,
    zusammen mit dem Nachbarn ergeben sie ein gewöhnliches Segment. Die Pause
    dazwischen bleibt erhalten: sie ist Teil der Sprache, nicht ihr Ende.

    Verschmolzen wird nur, wenn einer der beiden zu kurz ist -- zwei
    ausreichende Abschnitte zusammenzukleben brächte nichts als längere
    Beispiele.
    """
    zusammen: list[tuple[int, int]] = []
    for start, ende in bereiche:
        if zusammen:
            vorher_start, vorher_ende = zusammen[-1]
            zu_kurz = min(ende - start, vorher_ende - vorher_start) * frame_seconds < min_seconds
            passt = (ende - vorher_start) * frame_seconds <= max_seconds
            nah = start - vorher_ende <= luecke_frames
            if zu_kurz and passt and nah:
                zusammen[-1] = (vorher_start, ende)
                continue
        zusammen.append((start, ende))
    return zusammen


def _teile(  # noqa: PLR0913
    audio: np.ndarray,
    sample_rate: int,
    start: int,
    ende: int,
    laut: np.ndarray,
    frame: int,
    schnitt_frames: int,
    min_seconds: float,
    max_seconds: float,
    gut: list[tuple[int, int]],
    schlecht: list[tuple[int, int, str]],
    pegel: np.ndarray,
    force_split: bool,
    nachbar_s: float | None = None,
) -> None:
    dauer = (ende - start) / sample_rate
    if dauer < min_seconds:
        # Der Abstand gehört dazu: er sagt, ob der Abschnitt allein stand oder
        # ob die Lücke zum Nachbarn zu groß war, um beide zusammenzufassen.
        wie_weit = "" if nachbar_s is None else f"; Nachbar {nachbar_s:.1f}s entfernt"
        schlecht.append(
            (
                start,
                ende,
                f"zu kurz -- {dauer:.1f}s, kürzer als {min_seconds:.0f}s{wie_weit}",
            )
        )
        return
    if dauer <= max_seconds:
        gut.append((start, ende))
        return

    schnitt = _laengste_pause(laut, frame, start, ende, schnitt_frames)
    if schnitt is None and force_split:
        # Notausgang. Kein guter Schnitt, aber besser als der ganze Bereich in
        # den Ausschuss -- und die betroffenen Sätze sind im Manifest markiert.
        schnitt = _leiseste_stelle(pegel, frame, start, ende)
    if schnitt is None:
        schlecht.append(
            (
                start,
                ende,
                f"am Stück zu lang -- {dauer:.1f}s ohne Pause zum Schneiden. "
                "Beim Lesen zwischen den Sätzen deutlicher absetzen",
            )
        )
        return
    for a, b in ((start, schnitt), (schnitt, ende)):
        _teile(
            audio,
            sample_rate,
            a,
            b,
            laut,
            frame,
            schnitt_frames,
            min_seconds,
            max_seconds,
            gut,
            schlecht,
            pegel,
            force_split,
        )


def _leiseste_stelle(pegel: np.ndarray, frame: int, start: int, ende: int) -> int | None:
    """Mitte des leisesten Fensters im mittleren Drittel des Bereichs.

    Nur als Notausgang gedacht: gesucht ist die Stelle, an der ein Schnitt am
    wenigsten weh tut, wenn es keine echte Pause gibt. Das mittlere Drittel,
    damit nicht direkt am Rand getrennt wird und ein Schnipsel entsteht.
    """
    von, bis = start // frame, min(len(pegel), ende // frame)
    if bis - von < 6:
        return None
    drittel = (bis - von) // 3
    fenster = pegel[von + drittel : bis - drittel]
    if fenster.size == 0:
        return None
    return (von + drittel + int(np.argmin(fenster))) * frame


def _laengste_pause(
    laut: np.ndarray, frame: int, start: int, ende: int, schnitt_frames: int
) -> int | None:
    """Mitte der längsten Stille im Bereich, oder None."""
    von, bis = start // frame, min(len(laut), ende // frame)
    if bis - von < 3:
        return None
    fenster = laut[von:bis]
    beste: tuple[int, int, int] = (0, 0, 0)  # Länge, Anfang, Ende
    lauf_start = None
    for i, ist_laut in enumerate(fenster):
        if not ist_laut and lauf_start is None:
            lauf_start = i
        elif ist_laut and lauf_start is not None:
            if i - lauf_start > beste[0]:
                beste = (i - lauf_start, lauf_start, i)
            lauf_start = None
    if beste[0] < schnitt_frames:
        return None
    mitte = von + (beste[1] + beste[2]) // 2
    return mitte * frame


# -- Nachsehen, statt zu raten ----------------------------------------------


@dataclass(frozen=True)
class ProbeRow:
    """Was eine bestimmte Schwelle in dieser Aufnahme fände."""

    threshold_db: float
    pauses_split: int
    pauses_utterance: int
    longest_pause_s: float
    #: Anteil der Aufnahme, der bei dieser Schwelle als still gilt.
    silence_share: float = 0.0

    @property
    def brauchbar(self) -> bool:
        """Trennt diese Schwelle Sprache von Pause -- oder nur Alles von Nichts?

        Liegt sie über dem Sprechpegel, gilt die ganze Aufnahme als still und
        die Zahl der "Pausen" sagt nichts mehr aus. Genau so eine Zeile hätte
        sonst als brauchbare Einstellung durchgehen können.
        """
        return self.pauses_split > 0 and 0.005 <= self.silence_share <= 0.5


@dataclass(frozen=True)
class Probe:
    """Befund einer Aufnahme, ohne sie zu zerlegen.

    Warum das ein eigener Befehl ist: fällt eine Lesung durch, sind zwei
    Ursachen möglich -- eine Schwelle, die nicht zur Aufnahme passt, oder eine
    Leseweise ohne Pausen. Von außen sehen beide gleich aus. Diese Tabelle
    trennt sie: findet *keine* Schwelle Pausen, liegt es nicht an der Schwelle.
    """

    duration_s: float
    sample_rate: int
    levels: LevelReport
    digital_silence_share: float
    threshold_db: float
    rows: list[ProbeRow]

    @property
    def hoffnungslos(self) -> bool:
        """Keine einzige Schwelle trennt Sprache von Pause."""
        return not any(row.brauchbar for row in self.rows)

    def verwendete_zeile(self) -> ProbeRow | None:
        """Was die tatsächlich verwendete Schwelle findet."""
        return next((r for r in self.rows if r.threshold_db == self.threshold_db), None)

    def beste_zeile(self) -> ProbeRow | None:
        """Die brauchbare Schwelle mit den meisten Pausen."""
        brauchbar = [row for row in self.rows if row.brauchbar]
        return max(brauchbar, key=lambda r: r.pauses_split) if brauchbar else None

    def beste_schwelle(self) -> float | None:
        zeile = self.beste_zeile()
        return zeile.threshold_db if zeile else None

    def benoetigte_pausen(self, max_seconds: float = MAX_SECONDS) -> int:
        """Wie viele Schnittstellen die Aufnahme mindestens braucht.

        Ein Segment darf höchstens ``max_seconds`` lang sein; für n Segmente
        braucht es n-1 Trennstellen dazwischen.
        """
        return max(0, int(np.ceil(self.duration_s / max_seconds)) - 1)

    def gefundene_pausen(self) -> int:
        """Pausen bei der Schwelle, die tatsächlich verwendet wird.

        Bewusst nicht das Beste aus der Tabelle: was eine andere Schwelle fände,
        hilft niemandem, solange sie nicht verwendet wird. Genau diese
        Verwechslung ließ eine unzerlegbare Aufnahme als in Ordnung durchgehen.
        """
        zeile = self.verwendete_zeile()
        return zeile.pauses_split if zeile else 0

    def genug_pausen(self, max_seconds: float = MAX_SECONDS) -> bool:
        return self.gefundene_pausen() >= self.benoetigte_pausen(max_seconds)


def probe_audio(
    audio: np.ndarray,
    sample_rate: int,
    min_pause_ms: int = MIN_PAUSE_MS,
    split_pause_ms: int = SPLIT_PAUSE_MS,
) -> Probe:
    """Pegel messen und durchspielen, was verschiedene Schwellen fänden."""
    frame = max(1, sample_rate // 100)
    rahmen = audio[: len(audio) - len(audio) % frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(rahmen.astype(np.float64) ** 2, axis=1))
    with np.errstate(divide="ignore"):
        pegel = 20.0 * np.log10(np.maximum(rms, 1e-12))

    grund, sprech = silence_levels(pegel)
    anteil_null = float(np.mean(pegel <= DIGITAL_ZERO_DB))
    schwelle = threshold_for(grund, sprech)

    kandidaten = sorted({schwelle, *(float(w) for w in range(-60, -14, 5))})
    zeilen = [
        ProbeRow(
            threshold_db=wert,
            pauses_split=_zaehle_pausen(pegel, wert, max(1, split_pause_ms // 10)),
            pauses_utterance=_zaehle_pausen(pegel, wert, max(1, min_pause_ms // 10)),
            longest_pause_s=_laengste_stille(pegel, wert) * frame / sample_rate,
            silence_share=float(np.mean(pegel < wert)),
        )
        for wert in kandidaten
    ]
    return Probe(
        duration_s=duration_seconds(audio, sample_rate),
        sample_rate=sample_rate,
        levels=LevelReport(grund, sprech, anteil_null >= _DIGITAL_SILENCE_SHARE),
        digital_silence_share=anteil_null,
        threshold_db=schwelle,
        rows=zeilen,
    )


def _stille_laeufe(still: np.ndarray) -> list[tuple[int, int]]:
    """Zusammenhängende stille Abschnitte als (Anfang, Ende)."""
    if still.size == 0:
        return []
    wechsel = np.flatnonzero(np.diff(still.astype(np.int8)))
    grenzen = [0, *(int(i) + 1 for i in wechsel), len(still)]
    laeufe = []
    for a, b in zip(grenzen[:-1], grenzen[1:], strict=True):
        if still[a]:
            laeufe.append((a, b))
    return laeufe


def _zaehle_pausen(pegel: np.ndarray, schwelle: float, min_frames: int) -> int:
    """Pausen *innerhalb* der Aufnahme, ohne die stillen Ränder.

    Vorlauf und Ausklang sind fast immer still -- als Schnittstelle taugen sie
    nicht, denn zu trennen ist ja das, was dazwischen liegt. Zählte man sie mit,
    sähe eine durchgehend gesprochene Lesung mit ruhigem Anfang aus wie eine mit
    Pausen, und die Diagnose ginge in die Irre.
    """
    still = pegel < schwelle
    return sum(
        1 for a, b in _stille_laeufe(still) if b - a >= min_frames and a > 0 and b < len(still)
    )


def _laengste_stille(pegel: np.ndarray, schwelle: float) -> int:
    laengste, lauf = 0, 0
    for still in pegel < schwelle:
        lauf = lauf + 1 if still else 0
        laengste = max(laengste, lauf)
    return laengste


# -- Bauen ------------------------------------------------------------------


def build_dataset(
    name: str,
    sources: Iterable[Path],
    asr: ASREngine,
    datasets_dir: Path,
    language: str = "de",
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    force_split: bool = False,
    on_event=None,  # noqa: ANN001
) -> Dataset:
    """Aus langen Aufnahmen einen Trainingsdatensatz im F5-Format.

    ``force_split`` trennt zu lange Bereiche notfalls an ihrer leisesten Stelle,
    auch wenn dort keine echte Pause ist. Bewusst nicht der Normalfall -- aber
    wer durchgehend spricht, verlöre sonst sein ganzes Material.
    """
    melde = on_event or (lambda _text: None)
    root = datasets_dir / slug(name)
    root.mkdir(parents=True, exist_ok=True)
    (root / _WAVS).mkdir(exist_ok=True)

    # Derselbe Name heißt: neu bauen, nicht ergänzen. Die Segmente werden
    # durchnummeriert, ein kürzerer Lauf ließe die höheren Nummern des
    # vorherigen liegen -- Dateien, auf die nichts mehr zeigt und die beim
    # Nachsehen im Ordner das Bild verfälschen.
    alt = sorted((root / _WAVS).glob("utt_*.wav"))
    for datei in alt:
        datei.unlink()
    if alt:
        melde(f"Datensatz wird ersetzt: {len(alt)} Segmente des vorherigen Laufs entfernt")

    utterances: list[Utterance] = []
    verworfen: list[Rejection] = []
    raten: set[int] = set()

    for quelle in sources:
        audio, sample_rate = read_wav(quelle)
        raten.add(sample_rate)
        pegel: list[LevelReport] = []
        gut, schlecht = find_segments(
            audio,
            sample_rate,
            min_seconds,
            max_seconds,
            levels=pegel,
            force_split=force_split,
        )
        klang = f", {pegel[0].beschreibung()}" if pegel else ""
        melde(f"{quelle.name}: {len(gut)} brauchbar, {len(schlecht)} verworfen{klang}")

        for start, ende, grund in schlecht:
            verworfen.append(
                Rejection(
                    source=quelle.name,
                    start_s=start / sample_rate,
                    duration_s=(ende - start) / sample_rate,
                    reason=grund,
                )
            )

        for start, ende in gut:
            stueck = audio[start:ende]
            dauer = duration_seconds(stueck, sample_rate)
            roh = str(asr.transcribe(stueck, sample_rate, language)).strip()
            grund = _pruefe(stueck, sample_rate, dauer, roh)
            if grund:
                verworfen.append(
                    Rejection(
                        source=quelle.name,
                        start_s=start / sample_rate,
                        duration_s=dauer,
                        reason=grund,
                    )
                )
                continue

            index = len(utterances) + 1
            datei = f"{_WAVS}/utt_{index:05d}.wav"
            write_wav(root / datei, stueck, sample_rate)
            utterances.append(
                Utterance(
                    index=index,
                    file=datei,
                    text=normalize_german(roh),
                    raw_text=roh,
                    duration_s=round(dauer, 3),
                    peak_dbfs=round(peak_dbfs(stueck), 1),
                    source=quelle.name,
                )
            )

    if len(raten) > 1:
        raise ValueError(
            f"Die Aufnahmen haben verschiedene Abtastraten ({sorted(raten)}). "
            "Vor dem Bauen auf eine Rate bringen -- ein Datensatz mit gemischten "
            "Raten trainiert das Modell auf zwei verschiedene Stimmen."
        )

    dataset = Dataset(
        name=name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sample_rate=raten.pop() if raten else 0,
        utterances=utterances,
        rejected=verworfen,
        root=root,
    )
    dataset.save()
    dataset.write_metadata()
    return dataset


def _pruefe(audio: np.ndarray, sample_rate: int, dauer: float, text: str) -> str | None:
    """Warum dieses Segment nicht ins Training gehört -- oder None."""
    if not text:
        return "keine Rückschrift -- vermutlich kein Sprachanteil"
    if len(text) < 5:
        return f"Rückschrift zu kurz ({text!r})"

    spitze = peak_dbfs(audio)
    if spitze > -0.5:
        return f"übersteuert ({spitze:.1f} dBFS)"
    if spitze < -35.0:
        return f"zu leise ({spitze:.1f} dBFS)"

    tempo = len(text) / dauer if dauer else 0.0
    unten, oben = PLAUSIBLE_CHARS_PER_SECOND
    if not (unten <= tempo <= oben):
        # Fast immer eine Fehltranskription: zu viel oder zu wenig Text für die
        # Dauer. Solche Paare bringen dem Modell falsche Längen bei.
        return f"{tempo:.1f} Zeichen/s -- Text passt nicht zur Länge"
    return None


def _median(werte: list[float]) -> float:
    if not werte:
        return 0.0
    mitte = len(werte) // 2
    if len(werte) % 2:
        return werte[mitte]
    return (werte[mitte - 1] + werte[mitte]) / 2


def slug(name: str) -> str:
    return _SLUG.sub("-", name.lower()).strip("-")[:40] or "datensatz"
