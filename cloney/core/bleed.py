"""Erkennung des Referenz-Vorspanns am Anfang eines erzeugten Chunks.

F5-TTS erzeugt Referenz und neuen Text in einem Stück und schneidet den
Referenzteil anschließend an einer berechneten Stelle wieder ab::

    generated = generated[:, ref_audio_len:, :]

Passt die Aufnahme nicht genau zu ihrem Wortlaut -- und F5-TTS hängt an den
Referenztext stets ein Satzende samt Pause an, während es der Aufnahme nur 50 ms
Stille gibt --, dehnt das Modell den Referenzteil. Ein Rest landet dann hinter
der Schnittstelle und ist am Anfang des Ergebnisses zu hören.

Wegschneiden lässt sich das nicht nach Lautstärke, denn der Vorspann ist
Sprache. Wohl aber nach Inhalt: die Rückschrift sagt, welche Wörter zu hören
sind, und ihre Zeitangaben sagen, ab wann der gewünschte Text beginnt.
"""

from __future__ import annotations

from cloney.asr.base import TranscribedWord
from cloney.core.metrics import normalize_for_comparison

#: So viele Wörter müssen in Folge passen, damit der Anfang als gefunden gilt.
#: Ein einzelnes Wort genügt nicht -- gerade kurze Wörter wie "und" oder "der"
#: kommen im Vorspann genauso vor wie im gewünschten Text.
_BESTAETIGENDE_WOERTER = 3


def _wortliste(text: str) -> list[str]:
    return normalize_for_comparison(text).split()


def find_content_start(
    words: list[TranscribedWord] | tuple[TranscribedWord, ...],
    expected_text: str,
) -> tuple[float | None, int]:
    """Sucht, ab wann der gewünschte Text beginnt.

    Gibt Startzeit und Anzahl der Vorspann-Wörter zurück. ``(None, 0)`` heißt:
    kein Vorspann gefunden -- entweder ist keiner da, oder die Rückschrift passt
    so wenig zum Erwarteten, dass ein Schnitt ein Ratespiel wäre.
    """
    erwartet = _wortliste(expected_text)
    if not words or not erwartet:
        return None, 0

    gehoert = [normalize_for_comparison(w.text) for w in words]
    noetig = min(_BESTAETIGENDE_WOERTER, len(erwartet))

    for start in range(len(gehoert)):
        if len(gehoert) - start < noetig:
            break
        if all(gehoert[start + i] == erwartet[i] for i in range(noetig)):
            if start == 0:
                return None, 0
            return words[start].start, start
    return None, 0
