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

from cloney.core.lexicon import Lexicon
from cloney.core.normalize import ABBREVIATION_TOKENS, MONTH_NAMES, normalize_german

#: Ab dieser Zahl gilt "<Zahl>." eher als Satzende denn als Ordinalzahl.
#: Tage und Auflagen liegen darunter, Kapitel- und Mengenangaben darüber.
_MAX_PLAUSIBLE_ORDINAL = 31

#: Höchstlänge einer Zeile, die noch als Überschrift durchgeht.
#: Darüber ist es ein Satz, auch ohne Punkt am Ende.
MAX_HEADING_CHARS = 70

#: Markdown-Auszeichnung. Wer sie setzt, meint es eindeutig.
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")

#: Womit ein gewöhnlicher Satz endet. Fehlt das, ist die Zeile ein Kandidat.
_SATZENDE = ".!?…:,;"

_INITIAL = re.compile(r"\b[A-ZÄÖÜ]\.$")
_TRAILING_NUMBER = re.compile(r"\b(\d{1,4})\.$")
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]?(?=\s|$)")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


@dataclass(frozen=True)
class Sentence:
    raw: str
    normalized: str
    ends_paragraph: bool
    is_heading: bool = False


@dataclass(frozen=True)
class TextChunk:
    raw_text: str
    normalized_text: str
    ends_paragraph: bool
    is_heading: bool = False

    def estimated_seconds(self, chars_per_second: float) -> float:
        return len(self.normalized_text) / chars_per_second


def heading_text(line: str, folgt: str = "") -> str | None:
    """Ist diese Zeile eine Überschrift? Dann ihr Wortlaut, sonst ``None``.

    Eine Überschrift ist im Fließtext kein Satz: sie ist kurz, steht auf einer
    eigenen Zeile und endet ohne Satzzeichen. Genau daran ist sie zu erkennen --
    und genau deshalb wird sie sonst falsch gesprochen, weil die Engine ohne
    Satzzeichen nicht absetzt und die Zeile in den folgenden Text hineinliest.

    Die Falle dabei ist der harte Zeilenumbruch: ein auf 72 Zeichen umbrochener
    Absatz besteht aus lauter Zeilen ohne Satzzeichen. Deshalb zählt die
    Fortsetzung mit: geht es klein weiter, war es ein umbrochener Satz. Eine
    Markdown-Auszeichnung sticht diese Prüfung -- wer ``##`` schreibt, meint es.
    """
    line = line.strip()
    if not line:
        return None

    markdown = _MARKDOWN_HEADING.match(line)
    if markdown:
        return markdown.group(2).strip() or None

    if len(line) > MAX_HEADING_CHARS or line[-1] in _SATZENDE:
        return None
    # Eine Zeile, die klein anfängt, ist die Fortsetzung von etwas.
    if line[0].islower():
        return None
    weiter = folgt.lstrip()
    if weiter and (weiter[0].islower() or weiter[0] in ",;)"):
        return None
    return line


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


def split_sentences(text: str, lexicon: Lexicon | None = None) -> list[Sentence]:
    """Zerlegt Text in Sätze und normalisiert jeden Satz einzeln."""
    sentences: list[Sentence] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    for paragraph in paragraphs:
        # Überschriften stehen auf eigener Zeile. Würde der Absatz erst zu einer
        # Zeile zusammengezogen, wäre die Zeile darin nicht mehr zu finden --
        # und der Titel spräche sich ohne Absetzen in den Text hinein.
        zeilen = paragraph.splitlines()
        while zeilen:
            titel = heading_text(zeilen[0], zeilen[1] if len(zeilen) > 1 else "")
            if titel is None:
                break
            sentences.append(_heading_sentence(titel, lexicon))
            zeilen = zeilen[1:]

        body = " ".join(" ".join(zeilen).split())
        if not body:
            continue
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
            normalized = normalize_german(piece, lexicon)
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
        sentences[-1] = Sentence(last.raw, last.normalized, True, last.is_heading)
    return sentences


def spoken_form(raw_text: str, is_heading: bool = False, lexicon: Lexicon | None = None) -> str:
    """Die Sprechfassung eines Satzes -- dieselbe Regel wie beim Segmentieren.

    Gebraucht, wo ein einzelner Satz nachträglich neu gefasst wird: beim
    Bearbeiten seines Textes und beim Auffrischen nach einer Änderung am
    Aussprache-Wörterbuch. Eine zweite Fassung dieser Regel würde
    auseinanderlaufen, und der Unterschied fiele erst beim Hören auf.
    """
    if is_heading:
        return _heading_sentence(raw_text, lexicon).normalized
    return normalize_german(raw_text, lexicon)


def _heading_sentence(titel: str, lexicon: Lexicon | None = None) -> Sentence:
    """Eine Überschrift als eigener Satz.

    Der Punkt in der Sprechfassung ist der Kern: ohne Satzzeichen setzt die
    Engine nicht ab und hetzt die Zeile herunter. Im Rohtext steht er nicht --
    dort steht, was dasteht.
    """
    normalisiert = normalize_german(titel, lexicon)
    if normalisiert and normalisiert[-1] not in ".!?":
        normalisiert += "."
    return Sentence(raw=titel, normalized=normalisiert, ends_paragraph=True, is_heading=True)


def _split_long_sentence(
    sentence: Sentence, max_chars: int, lexicon: Lexicon | None = None
) -> list[Sentence]:
    """Notfall-Trennung überlanger Sätze an Teilsatzgrenzen (Komma, Semikolon)."""
    parts = _CLAUSE_BREAK.split(sentence.raw)
    if len(parts) == 1:
        return [sentence]

    out: list[Sentence] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer} {part}".strip()
        if buffer and len(normalize_german(candidate, lexicon)) > max_chars:
            out.append(Sentence(buffer, normalize_german(buffer, lexicon), False))
            buffer = part
        else:
            buffer = candidate
    if buffer:
        out.append(Sentence(buffer, normalize_german(buffer, lexicon), False))

    if out:
        out[-1] = Sentence(out[-1].raw, out[-1].normalized, sentence.ends_paragraph)
    return out


def build_chunks(
    text: str,
    chars_per_second: float = 14.0,
    target_seconds: float = 20.0,
    max_seconds: float = 25.0,
    lexicon: Lexicon | None = None,
) -> list[TextChunk]:
    """Gruppiert Sätze zu Chunks von etwa ``target_seconds`` Sprechzeit.

    Absatzgrenzen werden nie überschritten -- ein Chunk endet immer spätestens
    am Absatzende, damit die Pausensteuerung beim Zusammenbau greifen kann.
    """
    target_chars = int(target_seconds * chars_per_second)
    max_chars = int(max_seconds * chars_per_second)

    sentences: list[Sentence] = []
    for sentence in split_sentences(text, lexicon):
        if len(sentence.normalized) > max_chars:
            sentences.extend(_split_long_sentence(sentence, max_chars, lexicon))
        else:
            sentences.append(sentence)

    chunks: list[TextChunk] = []
    raw_parts: list[str] = []
    norm_parts: list[str] = []

    def flush(ends_paragraph: bool, is_heading: bool = False) -> None:
        if not norm_parts:
            return
        chunks.append(
            TextChunk(
                raw_text=" ".join(raw_parts),
                normalized_text=" ".join(norm_parts),
                ends_paragraph=ends_paragraph,
                is_heading=is_heading,
            )
        )
        raw_parts.clear()
        norm_parts.clear()

    for sentence in sentences:
        if sentence.is_heading:
            # Eine Überschrift bleibt für sich. Mit Fließtext im selben Chunk
            # läse die Engine sie mit -- genau das klingt gehetzt.
            flush(True)
            raw_parts.append(sentence.raw)
            norm_parts.append(sentence.normalized)
            flush(True, is_heading=True)
            continue

        pending = sum(len(p) + 1 for p in norm_parts)
        if norm_parts and pending + len(sentence.normalized) > target_chars:
            flush(False)
        raw_parts.append(sentence.raw)
        norm_parts.append(sentence.normalized)
        if sentence.ends_paragraph:
            flush(True)

    flush(True)
    return chunks
