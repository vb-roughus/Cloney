"""Satzbewusste Segmentierung deutscher Texte in Synthese-Chunks.

Der naive Split am Punkt scheitert im Deutschen an Abkürzungen ("z.B.", "Dr.")
und an Ordinalzahlen ("am 3. Mai") -- beides erzeugt Chunks, die mitten im Satz
abbrechen und hörbar falsch betont werden.

Segmentiert wird auf dem Rohtext, weil dort die Abkürzungen noch stehen und als
Signal dienen. Normalisiert wird satzweise unmittelbar danach, sodass Roh- und
Sprechfassung Satz für Satz zueinander ausgerichtet bleiben.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cloney.core.normalize import ABBREVIATION_TOKENS, MONTH_NAMES, normalize_german

#: Ab dieser Zahl gilt "<Zahl>." eher als Satzende denn als Ordinalzahl.
#: Tage und Auflagen liegen darunter, Kapitel- und Mengenangaben darüber.
_MAX_PLAUSIBLE_ORDINAL = 31

_INITIAL = re.compile(r"\b[A-ZÄÖÜ]\.$")
_TRAILING_NUMBER = re.compile(r"\b(\d{1,4})\.$")
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]?(?=\s|$)")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


@dataclass(frozen=True)
class Sentence:
    raw: str
    normalized: str
    ends_paragraph: bool


@dataclass(frozen=True)
class TextChunk:
    raw_text: str
    normalized_text: str
    ends_paragraph: bool

    def estimated_seconds(self, chars_per_second: float) -> float:
        return len(self.normalized_text) / chars_per_second


def _is_sentence_end(text: str, punct_end: int) -> bool:
    """Ist der Punkt an ``punct_end - 1`` ein echtes Satzende?"""
    prefix = text[:punct_end]
    if prefix.rstrip("\"')]").endswith(("!", "?")):
        return True

    stripped = prefix.rstrip("\"')]")
    if any(stripped.endswith(tok) for tok in ABBREVIATION_TOKENS):
        return False
    if _INITIAL.search(stripped):
        return False

    number = _TRAILING_NUMBER.search(stripped)
    if number:
        following = re.match(r"\s*([\wäöüÄÖÜß]+)", text[punct_end:])
        if following and following.group(1) in MONTH_NAMES:
            return False
        return int(number.group(1)) > _MAX_PLAUSIBLE_ORDINAL
    return True


def split_sentences(text: str) -> list[Sentence]:
    """Zerlegt Text in Sätze und normalisiert jeden Satz einzeln."""
    sentences: list[Sentence] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    for paragraph in paragraphs:
        body = " ".join(paragraph.split())
        pieces: list[str] = []
        start = 0
        for match in _BOUNDARY.finditer(body):
            if not _is_sentence_end(body, match.end()):
                continue
            pieces.append(body[start : match.end()].strip())
            start = match.end()
        tail = body[start:].strip()
        if tail:
            pieces.append(tail)

        for index, piece in enumerate(pieces):
            normalized = normalize_german(piece)
            if not normalized:
                continue
            sentences.append(
                Sentence(
                    raw=piece,
                    normalized=normalized,
                    ends_paragraph=index == len(pieces) - 1,
                )
            )

    if sentences:
        last = sentences[-1]
        sentences[-1] = Sentence(last.raw, last.normalized, ends_paragraph=True)
    return sentences


def _split_long_sentence(sentence: Sentence, max_chars: int) -> list[Sentence]:
    """Notfall-Trennung überlanger Sätze an Teilsatzgrenzen (Komma, Semikolon)."""
    parts = _CLAUSE_BREAK.split(sentence.raw)
    if len(parts) == 1:
        return [sentence]

    out: list[Sentence] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer} {part}".strip()
        if buffer and len(normalize_german(candidate)) > max_chars:
            out.append(Sentence(buffer, normalize_german(buffer), False))
            buffer = part
        else:
            buffer = candidate
    if buffer:
        out.append(Sentence(buffer, normalize_german(buffer), False))

    if out:
        out[-1] = Sentence(out[-1].raw, out[-1].normalized, sentence.ends_paragraph)
    return out


def build_chunks(
    text: str,
    chars_per_second: float = 14.0,
    target_seconds: float = 20.0,
    max_seconds: float = 25.0,
) -> list[TextChunk]:
    """Gruppiert Sätze zu Chunks von etwa ``target_seconds`` Sprechzeit.

    Absatzgrenzen werden nie überschritten -- ein Chunk endet immer spätestens
    am Absatzende, damit die Pausensteuerung beim Zusammenbau greifen kann.
    """
    target_chars = int(target_seconds * chars_per_second)
    max_chars = int(max_seconds * chars_per_second)

    sentences: list[Sentence] = []
    for sentence in split_sentences(text):
        if len(sentence.normalized) > max_chars:
            sentences.extend(_split_long_sentence(sentence, max_chars))
        else:
            sentences.append(sentence)

    chunks: list[TextChunk] = []
    raw_parts: list[str] = []
    norm_parts: list[str] = []

    def flush(ends_paragraph: bool) -> None:
        if not norm_parts:
            return
        chunks.append(
            TextChunk(
                raw_text=" ".join(raw_parts),
                normalized_text=" ".join(norm_parts),
                ends_paragraph=ends_paragraph,
            )
        )
        raw_parts.clear()
        norm_parts.clear()

    for sentence in sentences:
        pending = sum(len(p) + 1 for p in norm_parts)
        if norm_parts and pending + len(sentence.normalized) > target_chars:
            flush(False)
        raw_parts.append(sentence.raw)
        norm_parts.append(sentence.normalized)
        if sentence.ends_paragraph:
            flush(True)

    flush(True)
    return chunks
