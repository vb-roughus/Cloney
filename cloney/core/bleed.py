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
from cloney.core.audio import SILENCE_THRESHOLD_DB
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

# Alle drei Zahlen unten sind gemessen und nicht geschätzt. Vier Sätze aus einem
# echten Lauf, gegen eine Stimme, deren Referenz auf "Washington." endet:
#
#     Satz   Beginn   Dauer   Pause
#        1     0.26    0.13    0.01   <- kein Vorspann, der Satz selbst
#        2     0.10    0.14    0.36
#        3     0.10    0.14    0.36
#        4     0.09    0.14    0.37
#
# Daran ist zweierlei abzulesen. Erstens liegt der Fetzen *nicht* bei null: F5
# gibt der Referenz 50 ms Stille mit, und die steht mit davor. Zweitens ist die
# Pause dahinter riesig -- das ist die Pause des Satzendes, das F5 an den
# Referenztext anhängt. Genau sie trennt den Vorspann vom Satz, und genau sie
# fehlt bei Satz 1.

#: Bis hierhin muss ein Fetzen anfangen. Was später beginnt, ist der Satz.
FETZEN_BEGINN_MAX = 0.15

#: Länger als das ist kein Fetzen mehr, sondern eine Silbe.
FETZEN_DAUER_MAX = 0.20

#: So lange muss die Pause dahinter mindestens sein -- die tragende Bedingung.
#: Bemessen an dem, wovon sie zu unterscheiden ist: ein Verschlusslaut mitten im
#: ersten Wort (das /p/ in "Kapitel") ist drei bis acht Hundertstel still, eine
#: Kommapause anderthalb bis zwei Zehntel. Die Pause hinter einem Fetzen ist ein
#: Satzende und liegt darüber.
#:
#: Läge die Grenze tiefer, hielte die Erkennung eine Anfangssilbe für einen
#: Fetzen und schnitte sie weg, ohne dass es auffiele: die Fehlerrate misst
#: gegen die Rückschrift von vorher.
PAUSE_MIN_SEKUNDEN = 0.25

#: So weit wird am Anfang gesucht. Großzügig: Beginn, Fetzen und Pause zusammen
#: kamen in der Messung auf 0,60 s, und dahinter muss der Satz noch Platz haben.
#: Ein knapperes Fenster endete genau dort, wo der Satz anfängt -- dann sähe es
#: so aus, als käme nach der Pause nichts mehr, und nichts würde geschnitten.
_FETZEN_FENSTER_SEKUNDEN = 1.5

#: So lange muss es still sein, damit es als Ruhe zählt und nicht als Flackern.
#: Ein auslaufender Nasal schwankt um die Schwelle; ohne dieses Maß zerfiele er
#: in Bruchstücke, und hinter dem ersten stünde eine "Pause" von einem Rahmen.
_RUHE_MIN_SEKUNDEN = 0.06

#: Satzzeichen, hinter denen eine Pause gewollt ist.
_PAUSENZEICHEN = ",;:.!?…–—-"


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
    raw_text: str = "",
    schwelle_db: float = SILENCE_THRESHOLD_DB,
) -> float | None:
    """Ein kurzer Fetzen ganz am Anfang, durch eine Pause vom Satz getrennt.

    Gibt die Stelle zurück, an der der Satz beginnt -- oder ``None``, wenn da
    kein Fetzen ist.

    Der Weg über die Rückschrift greift hier nicht: ein Vorspann von wenigen
    Hundertstel liegt unterhalb dessen, was Whispers Wortzeiten auflösen. Was
    ihn trotzdem verrät, ist seine Gestalt -- ganz am Anfang, kurz, und dahinter
    eine Pause.

    Vier Bedingungen zusammen, weil jede für sich zu wenig ist. Der Anfang
    allein nicht: ein Satz fängt auch dort an. Die Kürze allein nicht: "Ja,"
    ist auch kurz. Die Pause allein nicht: nach "Ja," steht auch eine. Deshalb
    zählt als vierte der Text mit -- steht hinter dem ersten Wort ein
    Satzzeichen, ist die Pause gewollt und das davor kein Fetzen.

    Welche Bedingung im Einzelfall greift, beantwortet ``beurteile``.
    """
    return beurteile(audio, sample_rate, raw_text, schwelle_db).schnitt


@dataclass(frozen=True)
class Urteil:
    """Wo geschnitten wird -- und wenn nicht, warum nicht.

    Der Grund ist nicht Beiwerk. Steht am Anfang etwas Hörbares und wird
    trotzdem nicht geschnitten, ist die Frage, *welche* Bedingung das verhindert;
    ``cloney vorspann`` zeigt genau das. Ohne die Antwort bliebe nur Raten, und
    davon hat dieses Problem schon genug gehabt.
    """

    schnitt: float | None
    grund: str
    befund: Anfang | None


def beurteile(
    audio: np.ndarray,
    sample_rate: int,
    raw_text: str = "",
    schwelle_db: float = SILENCE_THRESHOLD_DB,
) -> Urteil:
    """Ist das am Anfang ein Fetzen? Und wenn nein, woran liegt es?"""
    befund = describe_start(audio, sample_rate, schwelle_db)
    if befund is None:
        return Urteil(None, "am Anfang ist nichts zu hören", None)
    if befund.beginn > FETZEN_BEGINN_MAX:
        return Urteil(None, f"beginnt erst bei {befund.beginn:.2f}s", befund)
    if befund.dauer > FETZEN_DAUER_MAX:
        return Urteil(None, f"dauert {befund.dauer:.2f}s -- das ist Sprache", befund)
    if not befund.danach:
        return Urteil(None, "nach der Pause kommt nichts mehr", befund)
    if befund.pause < PAUSE_MIN_SEKUNDEN:
        return Urteil(None, f"nur {befund.pause:.2f}s Pause dahinter", befund)
    if pause_erwartet(raw_text):
        return Urteil(None, "hinter dem ersten Wort steht ein Satzzeichen", befund)

    # An den Anfang des letzten stillen Rahmens, aus demselben Grund wie in
    # ``cut_point``: stehen gebliebene Stille hört niemand, einen
    # angeschnittenen Anlaut schon.
    return Urteil(befund.beginn + befund.dauer + befund.pause - _RAHMEN_SEKUNDEN, "", befund)


def pause_erwartet(raw_text: str) -> bool:
    """Steht hinter dem ersten Wort ein Satzzeichen?

    Das ersetzt die frühere Rechnung über die Länge des Wortes, und zwar weil
    sie die falsche Frage beantwortete. Gesucht ist nicht "könnte das ein Wort
    sein?", sondern "ist eine Pause an dieser Stelle gewollt?" -- und darauf
    antwortet der Text unmittelbar. "Ja, ..." und "Nun, ..." setzen ab, weil es
    so dasteht; ein "Sie" ohne Satzzeichen tut es nicht.

    Die Längenrechnung hätte in der Messung einen echten Vorspann verworfen:
    "sie" ist drei Zeichen, der Fetzen davor war vierzehn Hundertstel lang, und
    damit galt er als zu lang für einen Fetzen. Er war trotzdem einer.
    """
    erstes = raw_text.strip().split(maxsplit=1)
    if not erstes:
        return False
    return erstes[0].rstrip("\"')]»›“”")[-1:] in _PAUSENZEICHEN


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


def describe_start(
    audio: np.ndarray,
    sample_rate: int,
    schwelle_db: float = SILENCE_THRESHOLD_DB,
) -> Anfang | None:
    """Das erste Geräusch am Anfang und die Ruhe dahinter -- ohne Wertung.

    Die Schwelle ist **absolut** und nicht relativ zum lautesten Rahmen im
    Fenster. Der Unterschied ist nicht theoretisch: mit einer relativen Schwelle
    hing das Ergebnis an der Fensterbreite. Wurde das Fenster weiter, kamen die
    lauten Vokale des Satzes mit hinein, das Maximum stieg, die Schwelle stieg --
    und derselbe auslaufende Nasal fiel darunter und zerfiel in Bruchstücke.
    Dieselbe Aufnahme ergab dann 0,14 s Fetzen und 0,36 s Pause oder 0,02 s und
    0,01 s, je nachdem, wie weit gerade gesucht wurde. Ein Messgerät, das seinen
    Maßstab aus dem Gemessenen zieht, misst nichts.

    Gesucht wird die erste **echte** Ruhe, nicht die erste stille Stelle: ein
    Nasal schwankt um jede Schwelle, und ein Rahmen Stille darin ist keine Pause.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    energie = _rahmenenergie(audio[: int(_FETZEN_FENSTER_SEKUNDEN * sample_rate)], sample_rate)
    if energie.size < 3:
        return None

    with np.errstate(divide="ignore"):
        pegel = 20.0 * np.log10(np.maximum(energie, 1e-12))
    hoerbar = pegel > schwelle_db
    if not hoerbar.any():
        return None

    beginn = int(np.argmax(hoerbar))
    mindestens = max(1, round(_RUHE_MIN_SEKUNDEN / _RAHMEN_SEKUNDEN))
    ende, weiter = _erste_ruhe(hoerbar, beginn, mindestens)

    return Anfang(
        beginn=beginn * _RAHMEN_SEKUNDEN,
        dauer=(ende - beginn) * _RAHMEN_SEKUNDEN,
        pause=(weiter - ende) * _RAHMEN_SEKUNDEN,
        danach=weiter < hoerbar.size,
    )


def _erste_ruhe(hoerbar: np.ndarray, ab: int, mindestens: int) -> tuple[int, int]:
    """Anfang und Ende der ersten Ruhe von mindestens ``mindestens`` Rahmen.

    Gibt ``(len, len)`` zurück, wenn im Fenster keine zu finden ist -- dann
    reicht das Hörbare bis ans Ende, und es gibt nichts zu trennen.
    """
    stelle = ab
    while stelle < hoerbar.size:
        if hoerbar[stelle]:
            stelle += 1
            continue
        ruhe_ende = stelle
        while ruhe_ende < hoerbar.size and not hoerbar[ruhe_ende]:
            ruhe_ende += 1
        if ruhe_ende - stelle >= mindestens:
            return stelle, ruhe_ende
        stelle = ruhe_ende
    return hoerbar.size, hoerbar.size


def first_word(text: str) -> str:
    """Das erste Wort der Sprechfassung, ohne Satzzeichen."""
    woerter = _wortliste(text)
    return woerter[0] if woerter else ""
