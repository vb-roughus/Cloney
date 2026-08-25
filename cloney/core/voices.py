"""Verwaltung der Referenzstimmen samt Eingangsprüfung.

Eine schlechte Referenzaufnahme ist die häufigste Ursache für einen schlechten
Klon. Die Prüfung läuft deshalb beim Anlegen der Stimme -- nicht erst, wenn ein
ganzes Kapitel gerendert ist.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cloney.core.audio import duration_seconds, peak_dbfs, read_wav, trim_silence, write_wav
from cloney.engines.base import VoiceRef

_SLUG = re.compile(r"[^a-z0-9]+")
_META = "voice.json"
_REFERENCE = "reference.wav"


#: Übliches deutsches Sprechtempo in Zeichen je Sekunde.
#:
#: Hergeleitet aus 120 bis 150 Wörtern je Minute und rund sechs Zeichen je Wort
#: samt Leerzeichen (Normseite: 1500 Zeichen auf 250 Wörter). Zügiges Sprechen
#: im Podcast-Tempo liegt bei etwa 17 -- das ist normal, kein Mangel.
TYPICAL_CHARS_PER_SECOND = (12.0, 18.0)

#: Außerhalb dieses Bereichs gehört das Transkript vermutlich nicht zur Aufnahme.
#: Bewusst weit: er soll keine langsamen oder schnellen Sprecher bemängeln.
PLAUSIBLE_CHARS_PER_SECOND = (7.0, 24.0)

#: Angenehmes Tempo für längeres Zuhören. Dient nur als Ausgangspunkt für den
#: Vorschlag unten, nicht als Sollwert.
COMFORTABLE_CHARS_PER_SECOND = 14.5


def suggested_speed(chars_per_second: float | None) -> float | None:
    """Reglerwert, der ein zügiges Vorbild auf angenehmes Tempo bringt.

    F5-TTS übernimmt die Sprechgeschwindigkeit der Referenz. Wer ruhiger vorgelesen
    haben will als sein Vorbild spricht, stellt das über den Regler ein -- nicht
    über den Referenztext, der schlicht stimmen muss.
    """
    if not chars_per_second or chars_per_second <= 0:
        return None
    wert = COMFORTABLE_CHARS_PER_SECOND / chars_per_second
    if 0.95 <= wert <= 1.05:
        return None  # nah genug, ein Regler wäre nur Zierde
    return round(max(0.5, min(1.5, wert)), 2)


@dataclass(frozen=True)
class VoiceCheck:
    """Ergebnis der Eingangsprüfung. ``ok`` heißt: brauchbar, nicht perfekt."""

    ok: bool
    warnings: list[str]
    duration_s: float
    sample_rate: int
    peak_dbfs: float
    speech_ratio: float
    #: Zeichen je Sekunde aus Transkriptlänge und Dauer. None ohne Transkript.
    chars_per_second: float | None = None


def ends_abruptly(audio: np.ndarray, sample_rate: int, tail_ms: int = 120) -> bool:
    """Endet die Aufnahme mitten im Klang statt auszuklingen?

    F5-TTS hängt an den Referenztext stets ein Satzende samt Pause an, gibt der
    Aufnahme aber nur 50 ms Stille. Bricht sie zusätzlich mitten im Wort ab,
    passen Text und Klang am Ende nicht zusammen -- das Modell dehnt dann den
    Referenzteil, und ein Rest davon ist am Anfang des Ergebnisses zu hören.
    """
    tail = int(sample_rate * tail_ms / 1000)
    if audio.size < 2 * tail:
        return False
    gesamt = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    ende = float(np.sqrt(np.mean(audio[-tail:].astype(np.float64) ** 2)))
    if gesamt <= 0:
        return False
    # Ein natürliches Ausklingen liegt deutlich unter dem Mittel der Aufnahme.
    return ende > 0.5 * gesamt


def inspect_reference(
    audio: np.ndarray,
    sample_rate: int,
    min_seconds: float = 5.0,
    max_seconds: float = 12.0,
    transcript: str = "",
) -> VoiceCheck:
    duration = duration_seconds(audio, sample_rate)
    peak = peak_dbfs(audio)
    speech = trim_silence(audio, sample_rate)
    ratio = (len(speech) / len(audio)) if len(audio) else 0.0

    warnings: list[str] = []
    if duration < min_seconds:
        warnings.append(
            f"Nur {duration:.1f}s Referenz. Unter {min_seconds:.0f}s wird der Klon instabil."
        )
    if duration > max_seconds:
        warnings.append(
            f"{duration:.1f}s Referenz. Über {max_seconds:.0f}s bringt keinen Gewinn -- "
            "F5-TTS kürzt selbst auf 12s, und jede Sekunde Referenz verkürzt zugleich, "
            "was pro Chunk erzeugt werden kann."
        )
    if peak > -0.5:
        warnings.append("Aufnahme übersteuert (Peak nahe 0 dBFS). Verzerrungen werden mitgeklont.")
    if peak < -30.0:
        warnings.append(f"Sehr leise Aufnahme (Peak {peak:.0f} dBFS). Rauschen wird mitgeklont.")
    if sample_rate < 16000:
        warnings.append(f"Samplerate nur {sample_rate} Hz. Hohe Frequenzen fehlen dauerhaft.")
    if ratio < 0.5:
        warnings.append(
            f"Nur {ratio:.0%} der Aufnahme ist Sprache. Lange Pausen schwächen die Referenz."
        )

    # Der Referenztext ist keine Beschriftung, sondern der Wortlaut. F5-TTS
    # leitet aus dem Verhältnis von Textlänge zu Dauer die Sprechgeschwindigkeit
    # ab und bemisst danach, wie lang das Erzeugte werden darf. Ein Transkript,
    # das nicht zur Aufnahme gehört, verzerrt genau diese Rechnung -- das
    # Ergebnis klingt dann verhaspelt oder zerdehnt, ohne dass etwas abstürzt.
    # Gerechnet wird auf dem Sprachanteil, nicht auf der Dateilänge: F5-TTS
    # schneidet lange Stille aus der Referenz heraus, bevor es das Tempo
    # ableitet. Eine Aufnahme mit viel Vorlauf wirkt sonst langsamer als sie ist.
    speech_seconds = duration_seconds(speech, sample_rate) or duration
    rate: float | None = None
    if transcript.strip() and speech_seconds > 0:
        rate = len(transcript.strip()) / speech_seconds
        low, high = PLAUSIBLE_CHARS_PER_SECOND
        if rate < low:
            warnings.append(
                f"Der Referenztext ist mit {len(transcript.strip())} Zeichen sehr kurz für "
                f"{speech_seconds:.1f}s Sprache ({rate:.1f} Zeichen/s). Er muss der Wortlaut "
                "des Gesprochenen sein, keine Beschriftung -- sonst verschätzt sich das "
                "Modell beim Tempo und das Ergebnis wird unverständlich."
            )
        elif rate > high:
            warnings.append(
                f"Der Referenztext ist mit {len(transcript.strip())} Zeichen sehr lang für "
                f"{speech_seconds:.1f}s Sprache ({rate:.1f} Zeichen/s). Enthält er mehr, als "
                "in der Aufnahme gesprochen wird?"
            )

    if ends_abruptly(audio, sample_rate):
        warnings.append(
            "Die Aufnahme endet abrupt statt auszuklingen. F5-TTS erwartet am Ende "
            "eine kurze Pause; fehlt sie, kann ein Stück der Referenz am Anfang des "
            "erzeugten Tons stehen bleiben. Am besten nach dem letzten Wort noch "
            "einen Moment weiterlaufen lassen."
        )

    blocking = duration < 1.0 or ratio < 0.2
    return VoiceCheck(
        ok=not blocking,
        warnings=warnings,
        duration_s=duration,
        sample_rate=sample_rate,
        peak_dbfs=peak,
        speech_ratio=ratio,
        chars_per_second=rate,
    )


class VoiceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, name: str) -> Path:
        return self.root / _slug(name)

    def exists(self, name: str) -> bool:
        return (self.path(name) / _META).exists()

    def add(
        self,
        name: str,
        audio_path: Path,
        transcript: str = "",
        min_seconds: float = 5.0,
        max_seconds: float = 12.0,
    ) -> tuple[VoiceRef, VoiceCheck]:
        audio, sample_rate = read_wav(audio_path)
        check = inspect_reference(audio, sample_rate, min_seconds, max_seconds, transcript)

        directory = self.path(name)
        directory.mkdir(parents=True, exist_ok=True)
        write_wav(directory / _REFERENCE, audio, sample_rate)
        (directory / _META).write_text(
            json.dumps(
                {
                    "name": name,
                    "transcript": transcript,
                    "sample_rate": sample_rate,
                    "duration_s": round(check.duration_s, 2),
                    "warnings": check.warnings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.get(name), check

    def set_transcript(self, name: str, transcript: str) -> None:
        self.set_metadata(name, transcript=transcript)

    def resolve(self, name: str) -> Path:
        """Name zu Ordner -- und nur zu einem darunterliegenden.

        Der Name kommt aus einer URL. Schon der Slug entschärft Pfadangaben, weil
        er alles außer Buchstaben und Ziffern ersetzt; diese Prüfung ist die
        zweite Linie und stellt sicher, dass eine spätere Änderung am Slug nicht
        unbemerkt ein Löschen außerhalb des Datenverzeichnisses erlaubt.
        """
        directory = self.path(name).resolve()
        if directory.parent != self.root.resolve():
            raise ValueError(f"Ungültiger Stimmenname: {name!r}")
        return directory

    def set_metadata(self, name: str, **felder: object) -> None:
        meta_path = self.resolve(name) / _META
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(felder)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, name: str) -> None:
        shutil.rmtree(self.resolve(name), ignore_errors=True)

    def recheck(self, name: str) -> VoiceCheck:
        """Die Aufnahme erneut prüfen, etwa nachdem der Wortlaut geändert wurde.

        Das Sprechtempo ergibt sich aus Transkript und Dauer -- ein geänderter
        Text ändert also den Befund, ohne dass die Aufnahme angefasst wurde.
        """
        stimme = self.get(name)
        audio, sample_rate = read_wav(stimme.audio_path)
        pruefung = inspect_reference(audio, sample_rate, transcript=stimme.transcript)
        self.set_metadata(name, warnings=pruefung.warnings)
        return pruefung

    def speaking_rate(self, name: str) -> float | None:
        """Gemessenes Sprechtempo der Referenz in Zeichen je Sekunde."""
        stimme = self.get(name)
        if not stimme.transcript.strip():
            return None
        audio, sample_rate = read_wav(stimme.audio_path)
        return inspect_reference(audio, sample_rate, transcript=stimme.transcript).chars_per_second

    def get(self, name: str) -> VoiceRef:
        directory = self.path(name)
        meta = json.loads((directory / _META).read_text(encoding="utf-8"))
        return VoiceRef(
            name=meta["name"],
            audio_path=directory / _REFERENCE,
            transcript=meta.get("transcript", ""),
            duration_s=float(meta.get("duration_s", 0.0)),
        )

    def list_all(self) -> list[VoiceRef]:
        if not self.root.exists():
            return []
        return [self.get(d.name) for d in sorted(self.root.iterdir()) if (d / _META).exists()]


def _slug(name: str) -> str:
    return _SLUG.sub("-", name.lower()).strip("-") or "stimme"
