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

from dataclasses import dataclass

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

# -- Der Fetzen am Anfang ---------------------------------------------------
#
# Ein Vorspann von wenigen Hundertstel -- der auslaufende Nasal eines
# "Washington." etwa -- ist für die Rückschrift unsichtbar. Nicht, weil Whisper
# ihn überhört, sondern weil seine Zeitangaben ihn nicht auflösen können: sie
# sind auf Hundertstel gerundet und entstehen aus einer Ausrichtung über die
# Aufmerksamkeit, geglättet mit einem Medianfilter über sieben Rahmen zu je
# 20 ms (``faster_whisper/transcribe.py``, ``find_alignment``). Die Unschärfe
# liegt damit bei rund ±70 ms und ist größer als das, was zu finden wäre.
#
# Hier zählt deshalb nur die Wellenform. Der Fetzen hat eine eigene Gestalt: er
# liegt ganz am Anfang -- F5 trennt genau dort --, er ist kurz, und hinter ihm
# steht eine Pause, weil F5 an den Referenztext ein Satzende anhängt und das
# Modell danach absetzt.

#: Bis hierhin muss ein Fetzen anfangen. Was später beginnt, ist der Satz.
FETZEN_BEGINN_MAX = 0.05

#: Länger als das ist kein Fetzen mehr, sondern eine Silbe. Ein Fetzen ist der
#: Rest eines einzelnen Lautes -- das auslaufende "n" eines "Washington." --,
#: keine gesprochene Einheit.
FETZEN_DAUER_MAX = 0.12

#: So lange muss die Pause dahinter mindestens sein. Bemessen an dem, wovon sie
#: zu unterscheiden ist: ein Verschlusslaut mitten im ersten Wort -- das /p/ in
#: "Kapitel" -- ist drei bis acht Hundertstel still. Wäre die Grenze dort, hielte
#: die Erkennung eine Anfangssilbe für einen Fetzen und schnitte sie weg, ohne
#: dass es auffiele: die Fehlerrate misst gegen die Rückschrift von vorher.
#:
#: Die Pause hinter einem echten Fetzen ist länger, und das hat einen Grund:
#: F5 hängt an den Referenztext ein Satzende samt Pause an, das Modell setzt
#: danach also ab.
PAUSE_MIN_SEKUNDEN = 0.12

#: So weit wird am Anfang überhaupt gesucht.
_FETZEN_FENSTER_SEKUNDEN = 0.6

#: Ab welchem Anteil des lautesten Rahmens im Fenster etwas als hörbar gilt.
#: Tiefer als die Ruheschwelle: ein auslaufender Nasal ist deutlich leiser als
#: ein Vokal, und er soll trotzdem gefunden werden.
_FETZEN_SCHWELLE = 0.08

#: Wie viel kürzer als das erste erwartete Wort ein Fetzen sein muss. Ein Satz,
#: der mit "Ja," beginnt, sieht sonst aus wie ein Fetzen mit Pause dahinter --
#: und würde weggeschnitten.
_FETZEN_ANTEIL_VOM_WORT = 0.5


def _wortliste(text: str) -> list[str]:
    return normalize_for_comparison(text).split()


def _rahmenenergie(stueck: np.ndarray, sample_rate: int) -> np.ndarray:
    """Lautstärke je Rahmen von ``_RAHMEN_SEKUNDEN``."""
    schritt = max(1, int(_RAHMEN_SEKUNDEN * sample_rate))
    rahmen = len(stueck) // schritt
    if rahmen < 1:
        return np.zeros(0)
    werte = stueck[: rahmen * schritt].astype(np.float64).reshape(rahmen, schritt)
    return np.sqrt((werte**2).mean(axis=1))


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
    energie = _rahmenenergie(audio[int(ab * sample_rate) : int(bis * sample_rate)], sample_rate)
    if energie.size < 2 or not energie.max():
        return kandidat

    ruhig = np.flatnonzero(energie <= energie.max() * _RUHE_ANTEIL)
    if not ruhig.size:
        return kandidat
    # An den Anfang des letzten ruhigen Rahmens und nicht an sein Ende: die
    # zehn Hundertstel Ruhe, die dadurch stehen bleiben, hört niemand -- einen
    # angeschnittenen Anlaut schon.
    return ab + int(ruhig[-1]) * _RAHMEN_SEKUNDEN


def leading_fragment(
    audio: np.ndarray,
    sample_rate: int,
    erstes_wort: str = "",
    chars_per_second: float = 14.0,
) -> float | None:
    """Ein kurzer Fetzen ganz am Anfang, durch eine Pause vom Satz getrennt.

    Gibt die Stelle zurück, an der der Satz beginnt -- oder ``None``, wenn da
    kein Fetzen ist.

    Der Weg über die Rückschrift greift hier nicht: ein Vorspann von wenigen
    Hundertstel liegt unterhalb dessen, was Whispers Wortzeiten auflösen. Was
    ihn trotzdem verrät, ist seine Gestalt -- ganz am Anfang, kurz, und dahinter
    eine Pause.

    Drei Bedingungen zusammen, weil jede für sich zu wenig ist. Der Anfang
    allein nicht: ein Satz fängt auch dort an. Die Kürze allein nicht: "Ja,"
    ist auch kurz. Die Pause allein nicht: nach "Ja," steht auch eine. Deshalb
    kommt das erste erwartete Wort als vierte Bedingung dazu -- ein Fetzen ist
    ein Bruchteil eines Lautes und damit deutlich kürzer, als dieses Wort
    dauern kann.
    """
    befund = describe_start(audio, sample_rate)
    if befund is None or befund.beginn > FETZEN_BEGINN_MAX:
        # Fängt erst später an: dann ist das der Satz, und davor war Stille.
        return None
    if befund.dauer > FETZEN_DAUER_MAX:
        return None
    if befund.dauer >= _hoechstdauer(erstes_wort, chars_per_second):
        return None
    if not befund.danach or befund.pause < PAUSE_MIN_SEKUNDEN:
        # Ohne Pause dahinter -- oder ohne alles dahinter -- ist nicht zu
        # unterscheiden, ob das der Satz war.
        return None

    # An den Anfang des letzten stillen Rahmens, aus demselben Grund wie in
    # ``cut_point``: stehen gebliebene Stille hört niemand, einen
    # angeschnittenen Anlaut schon.
    return befund.beginn + befund.dauer + befund.pause - _RAHMEN_SEKUNDEN


@dataclass(frozen=True)
class Anfang:
    """Was am Anfang eines Chunks steht, in Zahlen statt in Adjektiven.

    Getrennt von der Entscheidung, weil die Zahlen für sich nützlich sind:
    ``cloney vorspann`` zeigt sie für ein ganzes Projekt, und erst daran ist zu
    sehen, ob eine Schwelle passt oder danebenliegt.
    """

    #: Wann das erste Hörbare beginnt.
    beginn: float
    #: Wie lange es am Stück anhält.
    dauer: float
    #: Wie lange es danach ruhig bleibt.
    pause: float
    #: Ob nach dieser Ruhe überhaupt noch etwas kommt.
    danach: bool


def describe_start(audio: np.ndarray, sample_rate: int) -> Anfang | None:
    """Das erste Geräusch am Anfang und die Ruhe dahinter -- ohne Wertung."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    energie = _rahmenenergie(audio[: int(_FETZEN_FENSTER_SEKUNDEN * sample_rate)], sample_rate)
    if energie.size < 3 or not energie.max():
        return None

    # '>=' und nicht '>': ist der Fetzen selbst das Lauteste im Fenster --
    # weil dahinter nur noch Stille kommt --, fiele er sonst durch.
    hoerbar = energie >= energie.max() * _FETZEN_SCHWELLE
    if not hoerbar.any():
        return None

    beginn = int(np.argmax(hoerbar))
    ende = beginn
    while ende < hoerbar.size and hoerbar[ende]:
        ende += 1
    weiter = ende
    while weiter < hoerbar.size and not hoerbar[weiter]:
        weiter += 1

    return Anfang(
        beginn=beginn * _RAHMEN_SEKUNDEN,
        dauer=(ende - beginn) * _RAHMEN_SEKUNDEN,
        pause=(weiter - ende) * _RAHMEN_SEKUNDEN,
        danach=weiter < hoerbar.size,
    )


def _hoechstdauer(erstes_wort: str, chars_per_second: float) -> float:
    """Wie lang ein Fetzen höchstens sein darf, damit er keiner ist.

    Ohne bekanntes erstes Wort bleibt es bei der festen Grenze -- dann ist die
    Kürze das einzige Maß.
    """
    if not erstes_wort or chars_per_second <= 0:
        return FETZEN_DAUER_MAX + 1.0
    return len(erstes_wort) / chars_per_second * _FETZEN_ANTEIL_VOM_WORT


def first_word(text: str) -> str:
    """Das erste Wort der Sprechfassung, ohne Satzzeichen."""
    woerter = _wortliste(text)
    return woerter[0] if woerter else ""
