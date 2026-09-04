"""Erkennung des Referenz-Vorspanns am Anfang eines erzeugten Chunks.

F5-TTS erzeugt Referenz und neuen Text in einem Stück und schneidet den
Referenzteil anschließend an einer berechneten Stelle wieder ab::

    generated = generated[:, ref_audio_len:, :]

Passt die Aufnahme nicht genau zu ihrem Wortlaut -- und F5-TTS hängt an den
Referenztext stets ein Satzende samt Pause an, während es der Aufnahme nur 50 ms
Stille gibt --, dehnt das Modell den Referenzteil. Ein Rest landet dann hinter
der Schnittstelle und ist am Anfang des Ergebnisses zu hören.

*Finden* lässt sich das nicht nach Lautstärke, denn der Vorspann ist Sprache.
Wohl aber nach Inhalt: die Rückschrift sagt, welche Wörter zu hören sind, und
ihre Zeitangaben sagen, ab wann der gewünschte Text beginnt. Die Lautstärke hat
danach trotzdem ihren Platz -- aber nur, um die gefundene Stelle genauer zu
legen, siehe ``cut_point``.
"""

from __future__ import annotations

import numpy as np

from cloney.asr.base import TranscribedWord
from cloney.core.metrics import normalize_for_comparison

#: So viele Wörter müssen in Folge passen, damit der Anfang als gefunden gilt.
#: Ein einzelnes Wort genügt nicht -- gerade kurze Wörter wie "und" oder "der"
#: kommen im Vorspann genauso vor wie im gewünschten Text.
_BESTAETIGENDE_WOERTER = 3

#: Wie weit vor dem gemeldeten Wortanfang nach der Ruhestelle gesucht wird.
#: Bemessen nach dem, was zu berichtigen ist -- der Ungenauigkeit von Whispers
#: Wortzeiten --, und nicht nach der Länge des Vorspanns. Ein weites Fenster
#: fände irgendeine Senke mitten im Vorspann, und der Schnitt fiele zu früh:
#: gemessen blieb die Silbe dann stehen, nur kürzer.
_SUCHFENSTER_SEKUNDEN = 0.15

#: Länge eines Rahmens bei dieser Suche.
_RAHMEN_SEKUNDEN = 0.01

#: Wie weit die Suche über den gemeldeten Wortanfang hinausgeht.
_NACHLAUF_SEKUNDEN = 0.05

#: Ab welchem Anteil des lautesten Rahmens im Fenster ein Rahmen als Ruhe gilt.
#: Kein absoluter Pegel: wie laut der Vorspann ist, hängt an Aufnahme und
#: Modell -- wie viel leiser die Lücke dazwischen ist, nicht.
_RUHE_ANTEIL = 0.15


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
            # Auch wenn der gewünschte Text gleich mit dem ersten gehörten Wort
            # beginnt, kann ein Vorspann davorstehen: ein angerissenes Wort oder
            # eine einzelne Silbe schreibt die Erkennung oft gar nicht auf --
            # hörbar ist sie trotzdem. Der Zeitpunkt des ersten Wortes sagt es
            # dennoch. Was davor liegt, gehört nicht zum Satz: entweder Vorspann
            # oder Stille, und beides ist am Anfang eines Chunks wegzuschneiden.
            return words[start].start, start
    return None, 0


def cut_point(
    audio: np.ndarray,
    sample_rate: int,
    kandidat: float,
    fenster: float = _SUCHFENSTER_SEKUNDEN,
) -> float:
    """Legt den Schnitt auf die leiseste Stelle in der Nähe des Kandidaten.

    Die Wortanfänge der Rückschrift sind geschätzt, nicht gemessen: Whisper
    leitet sie aus der Aufmerksamkeit ab und liegt regelmäßig ein paar
    Hundertstel daneben. Genau auf den gemeldeten Anfang zu schneiden träfe
    deshalb mal zu früh -- dann bleibt ein Rest des Vorspanns stehen -- und mal
    zu spät, und dann fehlt dem ersten Wort sein Anlaut. Das Zweite ist der
    schlimmere Fall: aus "Bargeld" würde "argeld", und niemand käme darauf, das
    im Schnitt zu suchen.

    Zwischen Vorspann und Satz liegt fast immer eine kurze Ruhestelle. Gesucht
    wird deshalb im Fenster um den Kandidaten die **letzte** ruhige Stelle: dort
    hört der Vorspann auf und der Satz fängt an.

    Nicht die leiseste. Der erste Anlauf tat das und schnitt zu früh -- Sprache
    hat Senken, eine Verschlusslaut-Pause mitten im Vorspann ist leiser als die
    Lücke danach, und getroffen wurde sie. Hörbar war das als eine Silbe, die
    blieb, nur kürzer.

    Und deshalb reicht das Fenster auch nur so weit zurück, wie Whispers Zeiten
    danebenliegen, nicht so weit, wie der Vorspann lang ist. Sonst geriete die
    letzte Ruhe schon in den Satz hinein -- eine Verschlusslaut-Pause im ersten
    Wort --, und der Schnitt nähme ihm seinen Anlaut.

    Ist im Fenster nichts ruhig, läuft der Vorspann ohne Absetzen in den Satz.
    Dann bleibt es beim Kandidaten: eine Senke zu suchen, die es nicht gibt,
    hieße raten.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dauer = len(audio) / sample_rate if sample_rate else 0.0
    ab = max(0.0, kandidat - fenster)
    bis = min(dauer, kandidat + _NACHLAUF_SEKUNDEN)
    schritt = max(1, int(_RAHMEN_SEKUNDEN * sample_rate))
    stueck = audio[int(ab * sample_rate) : int(bis * sample_rate)]
    rahmen = len(stueck) // schritt
    if rahmen < 2:
        return kandidat

    werte = stueck[: rahmen * schritt].astype(np.float64).reshape(rahmen, schritt)
    energie = np.sqrt((werte**2).mean(axis=1))
    ruhig = np.flatnonzero(energie <= energie.max() * _RUHE_ANTEIL)
    if not ruhig.size:
        return kandidat
    # An den Anfang des letzten ruhigen Rahmens und nicht an sein Ende: die
    # zehn Hundertstel Ruhe, die dadurch stehen bleiben, hört niemand -- einen
    # angeschnittenen Anlaut schon.
    return ab + int(ruhig[-1]) * _RAHMEN_SEKUNDEN
