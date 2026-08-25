"""Fehlerraten für die Qualitätskontrolle.

Die Rückschrift eines ASR-Modells wird nie zeichengleich mit der Vorlage sein.
Verglichen wird deshalb auf einer reduzierten Form: Kleinschreibung, ohne
Interpunktion, Umlaute vereinheitlicht -- damit die Kennzahl echte
Aussprachefehler misst und nicht die Schreibkonventionen des ASR-Modells.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def normalize_for_comparison(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.translate(_UMLAUTS)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def _levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (item_a != item_b),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. 0.0 = identisch, kann bei starker Abweichung > 1 werden."""
    ref = normalize_for_comparison(reference)
    hyp = normalize_for_comparison(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref = normalize_for_comparison(reference).split()
    hyp = normalize_for_comparison(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)
