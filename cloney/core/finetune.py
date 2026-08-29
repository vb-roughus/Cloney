"""Vorbereiten und Starten eines F5-TTS-Finetunes.

Cloney trainiert nicht selbst. Das übernehmen die Skripte von F5-TTS, und das
soll so bleiben: ihr Datenformat und ihre Trainingsschleife nachzubauen hieße,
sie bei jeder Änderung nachzuziehen. Was Cloney beisteuert, ist alles davor --
und genau dort liegen die Fallen:

1. **Das Eingabeformat weicht ab.** ``prepare_csv_wavs.py`` will eine CSV mit
   der Kopfzeile ``audio_file|text`` und **absoluten** Pfaden. Cloneys
   Datensatz führt relative Pfade ohne Kopfzeile, weil das für alles andere
   handlicher ist. Übersetzt wird beim Vorbereiten.

   Übergeben wird dabei die **Datei**, nicht ihr Ordner -- der Parameter heißt
   zwar ``inp_dir``, das Skript prüft aber auf die Endung ``.csv`` und liest die
   Tonpfade unverändert aus der Tabelle. Dem Namen zu folgen statt dem Hilfetext
   endet in ``ValueError: input must be a .csv file``.

2. **Das Vokabular muss vom Pretrain stammen.** F5 kopiert im Finetune-Zweig
   sein eigenes, fest eingetragenes Vokabular -- das des englisch-chinesischen
   Basismodells. Beim Finetune eines *deutschen* Modells passt das nicht zu den
   geladenen Gewichten: ``text_num_embeds`` ist die Vokabulargröße, und eine
   andere Größe heißt eine andere Embedding-Matrix.

   Schlimmer noch: der Finetune-Zweig prüft die Datei mit einem ``assert``, und
   sie liegt unter ``<f5>/data/Emilia_ZH_EN_pinyin/vocab.txt`` -- ein Pfad in
   den Trainingsdaten, die bei einer Installation über pip gar nicht mitkommen.
   Der Aufruf bricht dort also ab, bevor er irgendetwas tut.

   Deshalb wird ``--pretrain`` mitgegeben. Der Name führt in die Irre: das Flag
   steuert im Skript ausschließlich, ob das Vokabular aus den eigenen Texten
   erzeugt (``--pretrain``) oder das fest eingetragene kopiert wird. Alles
   andere ist in beiden Zweigen identisch. Erzeugt wird also zunächst ein
   Vokabular aus unseren Texten, und danach ersetzt Cloney es durch das des
   deutschen Pretrains -- das einzige, das zu den Gewichten passt.

3. **Die Ordner liegen bei F5, nicht bei Cloney.** Der Datenlader sucht unter
   ``<f5>/data/<name>_<tokenizer>``, die Checkpoints landen unter
   ``<f5>/ckpts/<name>``. Beides ergibt sich aus dem Ort des installierten
   Pakets, nicht aus dem Arbeitsverzeichnis.

4. **``batch_size_per_gpu`` zählt Frames, keine Beispiele.** Der Standard 3200
   entspricht rund 34 Sekunden Ton je Schritt. Was auf 16 GB durchläuft, ist
   hier nicht gemessen -- der Vorschlag ist ein Ausgangspunkt, kein Befund.
"""

from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cloney.core.dataset import Dataset

#: Mel-Frames je Sekunde bei 24 kHz und Hop 256 -- die Umrechnung, mit der aus
#: ``batch_size_per_gpu`` eine Tondauer wird.
FRAMES_PER_SECOND = 24000 / 256

#: Vorschlag für 16 GB. Der Standard von F5 liegt bei 3200 Frames; das ist auf
#: einer 24-GB-Karte erprobt, hier aber nicht gemessen.
BATCH_FRAMES_16GB = 1600

#: Aufwärmen über 20000 Schritte ist für einen Lauf von Grund auf gedacht. Ein
#: Finetune auf eine Stimme ist nach wenigen tausend Schritten vorbei -- bis
#: dahin wäre die Lernrate nie oben angekommen.
WARMUP_UPDATES = 200

#: Unterhalb dieser Menge ist ein Finetune ein Versuch, keine begründete
#: Erwartung. F5s eigene Angabe für "in-set inference" lautet 10 bis 100
#: Stunden; die dokumentierten Erfolge einzelner Sprecher liegen bei zwölf
#: Stunden und darüber. Für weniger gibt es keinen belegten Fall -- was nicht
#: heißt, dass es nicht geht, sondern dass es niemand gezeigt hat.
KNAPPES_MATERIAL_MINUTEN = 30.0

#: Häufiger sichern als F5s Standard von 50000: sonst gibt es bei einem kurzen
#: Lauf keinen einzigen Zwischenstand zum Anhören.
SAVE_PER_UPDATES = 1000
LAST_PER_UPDATES = 500


class FinetuneError(RuntimeError):
    """Fehler beim Vorbereiten oder Starten. Die Meldung ist für Menschen."""


def f5_root() -> Path:
    """Wurzel, unter der F5-TTS ``data`` und ``ckpts`` erwartet.

    F5 leitet beides aus dem Ort des Pakets ab (``files("f5_tts")/../..``), nicht
    aus dem Arbeitsverzeichnis. Bei einer Installation über pip liegt das also
    im site-packages-Baum -- unerwartet, aber es ist ihre Rechnung, und sie hier
    nachzubilden ist verlässlicher, als einen eigenen Ort zu erfinden.
    """
    try:
        import f5_tts
    except ImportError as exc:
        raise FinetuneError(
            'f5-tts ist nicht installiert. Installation: pip install -e ".[f5]"'
        ) from exc
    paket = Path(next(iter(f5_tts.__path__)))
    return (paket / ".." / "..").resolve()


@dataclass(frozen=True)
class TrainingPlan:
    """Alles, was ein Trainingslauf braucht -- berechnet, nicht geraten."""

    dataset_name: str
    #: Ordner, in dem F5 die vorbereiteten Daten sucht.
    data_dir: Path
    #: Ordner, in den F5 die Checkpoints schreibt.
    checkpoint_dir: Path
    #: Checkpoint und Vokabular des Pretrains.
    pretrain_ckpt: Path
    vocab_path: Path
    exp_name: str = "F5TTS_Base"
    batch_frames: int = BATCH_FRAMES_16GB
    learning_rate: float = 1e-5
    epochs: int = 100
    warmup_updates: int = WARMUP_UPDATES
    save_per_updates: int = SAVE_PER_UPDATES
    last_per_updates: int = LAST_PER_UPDATES
    #: Gesamtdauer des Datensatzes in Sekunden.
    total_seconds: float = 0.0
    extra: list[str] = field(default_factory=list)

    @property
    def seconds_per_step(self) -> float:
        return self.batch_frames / FRAMES_PER_SECOND

    @property
    def steps_per_epoch(self) -> int:
        """Wie viele Schritte eine Epoche etwa dauert.

        Macht aus 'batch_size_per_gpu = 1600' eine Aussage, mit der sich rechnen
        lässt: so viel Ton geht je Schritt durch, so oft kommt der Datensatz vor.
        """
        if self.seconds_per_step <= 0:
            return 0
        return max(1, round(self.total_seconds / self.seconds_per_step))

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs

    @property
    def knappes_material(self) -> bool:
        return self.total_seconds / 60.0 < KNAPPES_MATERIAL_MINUTEN

    def command(self) -> list[str]:
        """Der Aufruf von F5s finetune_cli, vollständig und nachvollziehbar."""
        return [
            sys.executable,
            "-m",
            "f5_tts.train.finetune_cli",
            "--exp_name",
            self.exp_name,
            "--dataset_name",
            self.dataset_name,
            "--finetune",
            "--pretrain",
            str(self.pretrain_ckpt),
            "--tokenizer",
            "custom",
            "--tokenizer_path",
            str(self.vocab_path),
            "--learning_rate",
            f"{self.learning_rate:g}",
            "--batch_size_per_gpu",
            str(self.batch_frames),
            "--batch_size_type",
            "frame",
            "--epochs",
            str(self.epochs),
            "--num_warmup_updates",
            str(self.warmup_updates),
            "--save_per_updates",
            str(self.save_per_updates),
            "--last_per_updates",
            str(self.last_per_updates),
            *self.extra,
        ]


def write_f5_metadata(dataset: Dataset, target: Path) -> Path:
    """Cloneys metadata.csv in die Form bringen, die F5 einliest.

    Zwei Unterschiede: eine Kopfzeile ``audio_file|text``, und absolute Pfade --
    ``prepare_csv_wavs.py`` prüft ausdrücklich darauf.
    """
    target.mkdir(parents=True, exist_ok=True)
    ziel = target / "metadata.csv"
    with ziel.open("w", encoding="utf-8", newline="") as datei:
        schreiber = csv.writer(datei, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        schreiber.writerow(["audio_file", "text"])
        for utterance in dataset.utterances:
            schreiber.writerow([str((dataset.root / utterance.file).resolve()), utterance.text])
    return ziel


def prepare_command(metadata_csv: Path, output_dir: Path) -> list[str]:
    """Aufruf von F5s Vorbereitung.

    Erstes Argument ist die **CSV-Datei**, obwohl der Parameter dort ``inp_dir``
    heißt: das Skript prüft auf die Endung und liest die Tonpfade unverändert
    aus der Tabelle.

    ``--pretrain`` trotz Finetune: das Flag steuert im Skript allein die Herkunft
    des Vokabulars. Ohne es prüft der Finetune-Zweig mit einem ``assert`` auf
    eine Datei in den Emilia-Trainingsdaten, die bei einer pip-Installation nicht
    vorhanden ist -- der Aufruf bricht dann ab, ohne etwas getan zu haben. Das
    passende Vokabular legt ``install_vocab`` danach hin.
    """
    return [
        sys.executable,
        "-m",
        "f5_tts.train.datasets.prepare_csv_wavs",
        str(metadata_csv),
        str(output_dir),
        "--pretrain",
    ]


def data_dir_for(dataset_name: str, tokenizer: str = "custom", root: Path | None = None) -> Path:
    """Wo F5 die vorbereiteten Daten sucht.

    ``root`` ist herausgezogen, damit sich die Pfadbildung ohne installiertes
    F5-TTS prüfen lässt -- und damit sich ein anderer Auscheckstand angeben
    lässt, falls jemand F5 nicht über pip installiert hat.
    """
    return (root or f5_root()) / "data" / f"{dataset_name}_{tokenizer}"


def checkpoint_dir_for(dataset_name: str, root: Path | None = None) -> Path:
    return (root or f5_root()) / "ckpts" / dataset_name


def install_vocab(vocab: Path, data_dir: Path) -> Path:
    """Das Vokabular des Pretrains an die Stelle legen, die F5 liest.

    ``prepare_csv_wavs.py`` kopiert im Finetune-Zweig sein eigenes, fest
    eingetragenes Vokabular. Das gehört zum englisch-chinesischen Basismodell.
    Wer ein deutsches Modell weitertrainiert, bekäme damit eine Embedding-Matrix
    anderer Größe als die geladenen Gewichte.
    """
    if not vocab.exists():
        raise FinetuneError(f"Vokabular des Pretrains nicht gefunden: {vocab}")
    ziel = data_dir / "vocab.txt"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vocab, ziel)
    return ziel


def check_prepared(data_dir: Path) -> None:
    """Liegt alles, was der Datenlader braucht?"""
    fehlend = [
        name
        for name in ("raw.arrow", "duration.json", "vocab.txt")
        if not (data_dir / name).exists()
    ]
    if fehlend:
        raise FinetuneError(
            f"In {data_dir} fehlt: {', '.join(fehlend)}. "
            "Zuerst 'cloney finetune prepare' laufen lassen."
        )


def plan_training(
    dataset: Dataset,
    pretrain_ckpt: Path,
    vocab_path: Path,
    *,
    batch_frames: int = BATCH_FRAMES_16GB,
    epochs: int = 100,
    learning_rate: float = 1e-5,
    tokenizer: str = "custom",
    f5_dir: Path | None = None,
) -> TrainingPlan:
    # Der Name wird zu einem Ordnernamen bei F5. Maßgeblich ist deshalb der
    # Ordner, unter dem der Datensatz tatsächlich liegt -- nicht die Anzeige,
    # die Leerzeichen und Umlaute enthalten darf.
    return TrainingPlan(
        dataset_name=dataset.root.name,
        data_dir=data_dir_for(dataset.root.name, tokenizer, f5_dir),
        checkpoint_dir=checkpoint_dir_for(dataset.root.name, f5_dir),
        pretrain_ckpt=pretrain_ckpt,
        vocab_path=vocab_path,
        batch_frames=batch_frames,
        epochs=epochs,
        learning_rate=learning_rate,
        total_seconds=dataset.total_seconds,
    )
