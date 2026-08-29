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
    levels: list[float] | None = None,
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
    if levels is not None:
        # Der Aufrufer will die gemessenen Pegel sehen -- sie sagen mehr über
        # die Aufnahme als jede Zahl, die wir selbst gewählt hätten.
        levels[:] = [grundpegel, sprechpegel]
    if silence_db is None:
        silence_db = min(grundpegel + SILENCE_MARGIN_DB, MAX_SILENCE_DB)
        if sprechpegel < SILENCE_FLOOR_DB:
            # Nichts zu hören. Kein Befund, sondern schlicht keine Aufnahme.
            return [], []
        if sprechpegel - grundpegel < MIN_DYNAMIC_RANGE_DB:
            # Zwischen leise und laut liegt fast nichts: entweder durchgehend
            # gesprochen oder stark komprimiert. Ein Schnitt wäre geraten.
            return [], [
                (
                    0,
                    len(audio),
                    f"kaum Unterschied zwischen leisen und lauten Stellen "
                    f"({grundpegel:.0f} bis {sprechpegel:.0f} dBFS) -- keine Pausen erkennbar",
                )
            ]
    laut = pegel > silence_db

    pause_frames = max(1, int(min_pause_ms / 10))
    schnitt_frames = max(1, int(split_pause_ms / 10))
    bereiche = _speech_runs(laut, pause_frames)

    rand = int(sample_rate * MARGIN_MS / 1000)
    gut: list[tuple[int, int]] = []
    schlecht: list[tuple[int, int, str]] = []
    for start_f, ende_f in bereiche:
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
        )
    return gut, schlecht


#: Fenster, über das der Grundpegel gesucht wird. Dieselbe Größenordnung wie
#: die kürzeste brauchbare Pause -- gesucht ist ja genau deren Pegel.
_FLOOR_WINDOW = 20


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
    if pegel.size < _FLOOR_WINDOW:
        grund = float(pegel.min())
    else:
        sicht = np.lib.stride_tricks.sliding_window_view(pegel, _FLOOR_WINDOW)
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
) -> None:
    dauer = (ende - start) / sample_rate
    if dauer < min_seconds:
        schlecht.append((start, ende, f"nur {dauer:.1f}s -- kürzer als {min_seconds:.0f}s"))
        return
    if dauer <= max_seconds:
        gut.append((start, ende))
        return

    schnitt = _laengste_pause(laut, frame, start, ende, schnitt_frames)
    if schnitt is None:
        schlecht.append(
            (
                start,
                ende,
                f"{dauer:.1f}s ohne Pause zum Schneiden -- am Stück zu lang. "
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
        )


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


# -- Bauen ------------------------------------------------------------------


def build_dataset(
    name: str,
    sources: Iterable[Path],
    asr: ASREngine,
    datasets_dir: Path,
    language: str = "de",
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    on_event=None,  # noqa: ANN001
) -> Dataset:
    """Aus langen Aufnahmen einen Trainingsdatensatz im F5-Format."""
    melde = on_event or (lambda _text: None)
    root = datasets_dir / slug(name)
    root.mkdir(parents=True, exist_ok=True)
    (root / _WAVS).mkdir(exist_ok=True)

    utterances: list[Utterance] = []
    verworfen: list[Rejection] = []
    raten: set[int] = set()

    for quelle in sources:
        audio, sample_rate = read_wav(quelle)
        raten.add(sample_rate)
        pegel: list[float] = []
        gut, schlecht = find_segments(audio, sample_rate, min_seconds, max_seconds, levels=pegel)
        raumton = f", Raumton {pegel[0]:.0f} dBFS" if pegel else ""
        melde(f"{quelle.name}: {len(gut)} Abschnitte, {len(schlecht)} verworfen{raumton}")

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
