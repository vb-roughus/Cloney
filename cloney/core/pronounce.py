"""Aussprache dessen, was deutsche Regeln nicht hergeben.

Zwei getrennte Fälle, die von außen gleich aussehen:

1. **Buchstabierte Abkürzungen.** ``USB`` wird Buchstabe für Buchstabe
   gesprochen. Das ist keine Geschmacksfrage, sondern eine Tabelle: die
   deutschen Buchstabennamen stehen fest. Cloney kann das ausrechnen.

2. **Fremdwörter und Akronyme, die wie Wörter gesprochen werden.** ``SWIFT``,
   ``ACID``, ``Journal``. Wie sie klingen sollen, steht in keiner Regel -- es
   hängt an der Sprache, aus der sie stammen, und daran, wie geläufig sie im
   Deutschen sind. Das kann nur ein Mensch entscheiden, und deshalb steht es in
   einem Wörterbuch, das er führt.

Für den zweiten Fall liefert Cloney bewusst **keine** vorgefertigten
Aussprachen mit. Eine geratene Lautschrift wäre schlimmer als keine: sie sähe
aus wie eine Regel, wäre aber ein Vorschlag, den niemand belegt hat.
"""

from __future__ import annotations

import re

#: Die deutschen Buchstabennamen. Feststehend, keine Geschmacksfrage.
LETTER_NAMES: dict[str, str] = {
    "A": "A",
    "B": "Be",
    "C": "Ze",
    "D": "De",
    "E": "E",
    "F": "Ef",
    "G": "Ge",
    "H": "Ha",
    "I": "I",
    "J": "Jot",
    "K": "Ka",
    "L": "El",
    "M": "Em",
    "N": "En",
    "O": "O",
    "P": "Pe",
    "Q": "Ku",
    "R": "Er",
    "S": "Es",
    "T": "Te",
    "U": "U",
    "V": "Vau",
    "W": "We",
    "X": "Iks",
    "Y": "Ypsilon",
    "Z": "Zett",
    "Ä": "Ä",
    "Ö": "Ö",
    "Ü": "Ü",
    "0": "Null",
    "1": "Eins",
    "2": "Zwei",
    "3": "Drei",
    "4": "Vier",
    "5": "Fünf",
    "6": "Sechs",
    "7": "Sieben",
    "8": "Acht",
    "9": "Neun",
}

#: Ab dieser Länge gilt eine Kette aus Großbuchstaben als Abkürzung.
#: Ein einzelner Großbuchstabe ist ein Satzanfang oder eine Initiale.
MIN_ACRONYM_LETTERS = 2

#: Eine Kette aus Großbuchstaben und Ziffern, wie sie Abkürzungen bilden.
#: Der Wortrand hält 'GmbH' und 'iPhone' heraus -- dort steht Kleinschreibung
#: dazwischen, und beides ist keine buchstabierte Abkürzung.
_ACRONYM = re.compile(r"\b[A-ZÄÖÜ][A-ZÄÖÜ0-9]+\b")


def spell_out(word: str, trenner: str = "-") -> str:
    """``USB`` zu ``U-Es-Be``.

    Der Bindestrich hält die Buchstaben als ein Wort zusammen; mit Leerzeichen
    liest eine Engine drei einzelne Wörter, jedes mit eigener Betonung.
    """
    namen = [LETTER_NAMES.get(zeichen, zeichen) for zeichen in word.upper() if zeichen.strip()]
    return trenner.join(namen)


def acronyms(text: str) -> list[str]:
    """Ketten aus Großbuchstaben im Text, in der Reihenfolge des Auftretens.

    Kandidaten, nicht Befunde: ob ``ACID`` buchstabiert oder als Wort gesprochen
    wird, weiß der Text nicht. Die Liste sagt nur, worüber zu entscheiden ist.
    """
    gefunden: list[str] = []
    for treffer in _ACRONYM.finditer(text):
        wort = treffer.group(0)
        if sum(1 for c in wort if c.isalpha()) < MIN_ACRONYM_LETTERS:
            continue
        if wort not in gefunden:
            gefunden.append(wort)
    return gefunden
