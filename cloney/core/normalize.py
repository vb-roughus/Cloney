"""Deterministische Normalisierung deutscher Texte für die Sprachsynthese.

Kein TTS-Modell spricht "Am 3. Mai 2024 kostete es 1.250,50 €" korrekt aus. Dieses
Modul wandelt alles, was kein Wort ist, in ausgeschriebene Wörter um -- regelbasiert,
damit das Ergebnis reproduzierbar und testbar ist.

Die Reihenfolge der Regeln ist bedeutsam: spezifische Muster (Datum, Uhrzeit, Währung)
laufen vor generischen (Dezimalzahl, Kardinalzahl), damit die spezifischen ihre Ziffern
zuerst konsumieren.
"""

from __future__ import annotations

import re

from num2words import num2words

# --------------------------------------------------------------------------
# Tabellen
# --------------------------------------------------------------------------

MONTHS = {
    1: "Januar",
    2: "Februar",
    3: "März",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember",
}

MONTH_NAMES = frozenset(MONTHS.values())

#: Abkürzung -> Ausschreibung. Punkt gehört zum Schlüssel, damit die
#: Satzsegmentierung dieselbe Tabelle nutzen kann.
ABBREVIATIONS: dict[str, str] = {
    "z.B.": "zum Beispiel",
    "z. B.": "zum Beispiel",
    "d.h.": "das heißt",
    "d. h.": "das heißt",
    "u.a.": "unter anderem",
    "u. a.": "unter anderem",
    "u.U.": "unter Umständen",
    "u. U.": "unter Umständen",
    "i.d.R.": "in der Regel",
    "i. d. R.": "in der Regel",
    "z.T.": "zum Teil",
    "z. T.": "zum Teil",
    "v.a.": "vor allem",
    "v. a.": "vor allem",
    "o.ä.": "oder ähnliches",
    "o. ä.": "oder ähnliches",
    "u.ä.": "und ähnliches",
    "u. ä.": "und ähnliches",
    "s.o.": "siehe oben",
    "s.u.": "siehe unten",
    "usw.": "und so weiter",
    "u.s.w.": "und so weiter",
    "etc.": "et cetera",
    "bzw.": "beziehungsweise",
    "ca.": "circa",
    "vgl.": "vergleiche",
    "ggf.": "gegebenenfalls",
    "evtl.": "eventuell",
    "sog.": "sogenannt",
    "bspw.": "beispielsweise",
    "insb.": "insbesondere",
    "inkl.": "inklusive",
    "exkl.": "exklusive",
    "zzgl.": "zuzüglich",
    "abzgl.": "abzüglich",
    "einschl.": "einschließlich",
    "urspr.": "ursprünglich",
    "eigtl.": "eigentlich",
    "allg.": "allgemein",
    "Dr.": "Doktor",
    "Prof.": "Professor",
    "Hrsg.": "Herausgeber",
    "Nr.": "Nummer",
    "Abb.": "Abbildung",
    "Tab.": "Tabelle",
    "Abs.": "Absatz",
    "Art.": "Artikel",
    "Ziff.": "Ziffer",
    "Kap.": "Kapitel",
    "Bd.": "Band",
    "Aufl.": "Auflage",
    "Jh.": "Jahrhundert",
    "Jhd.": "Jahrhundert",
    "Str.": "Straße",
    "St.": "Sankt",
    "S.": "Seite",
}

#: Nur die Token, die auf einen Punkt enden -- die Satzsegmentierung braucht sie,
#: um nicht mitten in "z.B." zu trennen.
ABBREVIATION_TOKENS = frozenset(k for k in ABBREVIATIONS if k.endswith("."))

#: Einheiten hinter einer Zahl. Singular/Plural getrennt, weil "1 Meter"/"2 Meter"
#: im Deutschen gleich, "1 Sekunde"/"2 Sekunden" aber verschieden ist.
UNITS: dict[str, tuple[str, str, str]] = {
    # Kürzel -> (Singular, Plural, unbestimmter Artikel)
    "km": ("Kilometer", "Kilometer", "ein"),
    "m": ("Meter", "Meter", "ein"),
    "cm": ("Zentimeter", "Zentimeter", "ein"),
    "mm": ("Millimeter", "Millimeter", "ein"),
    "kg": ("Kilogramm", "Kilogramm", "ein"),
    "g": ("Gramm", "Gramm", "ein"),
    "t": ("Tonne", "Tonnen", "eine"),
    "l": ("Liter", "Liter", "ein"),
    "ml": ("Milliliter", "Milliliter", "ein"),
    "h": ("Stunde", "Stunden", "eine"),
    "min": ("Minute", "Minuten", "eine"),
    "sek": ("Sekunde", "Sekunden", "eine"),
    "kW": ("Kilowatt", "Kilowatt", "ein"),
    "kWh": ("Kilowattstunde", "Kilowattstunden", "eine"),
    "MB": ("Megabyte", "Megabyte", "ein"),
    "GB": ("Gigabyte", "Gigabyte", "ein"),
    "TB": ("Terabyte", "Terabyte", "ein"),
}

#: Substantive, vor denen die Eins "eine" heißt. Bewusst klein gehalten: die
#: Liste deckt Zeit- und Mengenwörter ab, bei allem anderen fällt die Regel auf
#: "ein" zurück -- ein Genusfehler ist immer noch weit besser als "eins Minute".
FEMININE_NOUNS = frozenset(
    {
        "Minute", "Sekunde", "Stunde", "Woche", "Tonne", "Million", "Milliarde",
        "Person", "Seite", "Nacht", "Frage", "Antwort", "Möglichkeit",
        "Stimme", "Aufnahme", "Datei", "Zeile",
    }
)

CURRENCIES: dict[str, tuple[str, str, str]] = {
    # Symbol -> (Singular, Plural, Untereinheit-Plural)
    "€": ("Euro", "Euro", "Cent"),
    "EUR": ("Euro", "Euro", "Cent"),
    "CHF": ("Franken", "Franken", "Rappen"),
    "Fr.": ("Franken", "Franken", "Rappen"),
    "$": ("Dollar", "Dollar", "Cent"),
    "USD": ("Dollar", "Dollar", "Cent"),
    "£": ("Pfund", "Pfund", "Pence"),
}

SYMBOLS = {
    "%": "Prozent",
    "‰": "Promille",
    "§": "Paragraf",
    "&": "und",
    "°C": "Grad Celsius",
    "°": "Grad",
    "+": "plus",
    "×": "mal",
    "=": "gleich",
    "@": "at",
}

#: Ein Ordinalzahl-Punkt nach diesen Wörtern verlangt die Dativ-/Genitiv-Endung
#: ("am 3. Mai" -> "am dritten Mai").
_OBLIQUE_TRIGGERS = frozenset(
    [
        "am",
        "im",
        "vom",
        "zum",
        "beim",
        "dem",
        "des",
        "einem",
        "eines",
        "den",
        "einen",
        "zur",
        "seit",
        "nach",
        "mit",
        "aus",
        "bei",
        "ab",
        "bis",
        "ihrem",
        "seinem",
        "meinem",
        "unserem",
        "eurem",
        "ihres",
        "seines",
    ]
)

#: Signalwörter, die ein vierstelliges 11xx-19xx als Jahreszahl kennzeichnen.
#: Ohne solches Signal wird konservativ als Kardinalzahl gelesen -- eine etwas
#: steife Lesung ist besser als eine falsche.
_YEAR_TRIGGERS = frozenset(
    [
        "im",
        "in",
        "seit",
        "ab",
        "bis",
        "von",
        "vor",
        "nach",
        "um",
        "jahr",
        "jahre",
        "jahren",
        "jahrgang",
        "anno",
        "sommer",
        "winter",
        "herbst",
        "frühling",
        "frühjahr",
        "geboren",
        "gestorben",
        "gegründet",
        "erschienen",
        "veröffentlicht",
        "datiert",
    ]
) | {m.lower() for m in MONTH_NAMES}


# --------------------------------------------------------------------------
# Zahlwort-Helfer
# --------------------------------------------------------------------------


def cardinal(n: int) -> str:
    """Kardinalzahl als deutsches Zahlwort."""
    return num2words(n, lang="de")


def ordinal(n: int, oblique: bool = False) -> str:
    """Ordinalzahl. ``oblique=True`` liefert die -en-Form (Dativ/Genitiv/Akk. mask.)."""
    word = num2words(n, lang="de", to="ordinal")
    if oblique and word.endswith("e"):
        return word + "n"
    return word


def year(n: int) -> str:
    """Jahreszahl in der üblichen Hunderter-Lesung: 1984 -> neunzehnhundertvierundachtzig."""
    if 1100 <= n <= 1999:
        hundreds, rest = divmod(n, 100)
        head = f"{cardinal(hundreds)}hundert"
        return head if rest == 0 else f"{head}{cardinal(rest)}"
    return cardinal(n)


def _digits(s: str) -> str:
    """Ziffernfolge einzeln gelesen: '14' -> 'eins vier'."""
    return " ".join(cardinal(int(d)) for d in s)


def _int_from_german(s: str) -> int:
    """'1.250' -> 1250 (Punkt ist deutscher Tausendertrenner)."""
    return int(s.replace(".", "").replace(" ", "").replace(" ", ""))


def _amount(value: int, singular: str, plural: str, article: str = "ein") -> str:
    """Zahlwort + Einheit mit korrektem Numerus. 1 wird zum Artikel, nicht zu 'eins'."""
    if value == 1:
        return f"{article} {singular}"
    return f"{cardinal(value)} {plural}"


# --------------------------------------------------------------------------
# Einzelregeln
# --------------------------------------------------------------------------

_NUM = r"(?:\d{1,3}(?:\.\d{3})+|\d+)"


def _clean_typography(text: str) -> str:
    text = text.replace(" ", " ").replace(" ", " ")
    for dash in ("–", "—"):
        text = text.replace(f" {dash} ", ", ")
    text = re.sub(r"[„“”»«]", '"', text)
    text = re.sub(r"[‚‘’]", "'", text)
    text = text.replace("…", ".")
    return text


def _expand_urls_emails(text: str) -> str:
    text = re.sub(
        r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        lambda m: m.group(0).replace("@", " at ").replace(".", " Punkt "),
        text,
    )
    text = re.sub(r"\bhttps?://\S+", "", text)
    text = re.sub(r"\bwww\.\S+", "", text)
    return text


def _expand_dates(text: str) -> str:
    """3.5.2024 / 03.05.2024 / 3. Mai 2024 -> ausgeschrieben."""

    def full(m: re.Match[str]) -> str:
        day, month, yr = int(m.group("d")), int(m.group("m")), int(m.group("y"))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m.group(0)
        oblique = _preceding_word(text, m.start()) in _OBLIQUE_TRIGGERS
        if yr < 100:
            yr += 2000 if yr < 50 else 1900
        return f"{ordinal(day, oblique)} {MONTHS[month]} {year(yr)}"

    text = re.sub(r"\b(?P<d>\d{1,2})\.\s?(?P<m>\d{1,2})\.\s?(?P<y>\d{2,4})\b", full, text)

    def day_month(m: re.Match[str]) -> str:
        day, month = int(m.group("d")), int(m.group("m"))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m.group(0)
        oblique = _preceding_word(text, m.start()) in _OBLIQUE_TRIGGERS
        return f"{ordinal(day, oblique)} {MONTHS[month]}"

    return re.sub(r"\b(?P<d>\d{1,2})\.\s?(?P<m>\d{1,2})\.(?!\d)", day_month, text)


def _expand_times(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        h, minute = int(m.group(1)), int(m.group(2))
        if h > 23 or minute > 59:
            return m.group(0)
        if minute == 0:
            return f"{cardinal(h)} Uhr"
        return f"{cardinal(h)} Uhr {cardinal(minute)}"

    return re.sub(r"\b(\d{1,2}):(\d{2})\b(?:\s*Uhr)?", repl, text)


def _expand_currency(text: str) -> str:
    syms = "|".join(re.escape(s) for s in sorted(CURRENCIES, key=len, reverse=True))

    def repl(m: re.Match[str]) -> str:
        sym = m.group("sym")
        singular, plural, sub = CURRENCIES[sym]
        whole = _int_from_german(m.group("int"))
        head = _amount(whole, singular, plural)
        cents = m.group("frac")
        if cents and int(cents) > 0:
            return f"{head} {cardinal(int(cents))}"
        return head

    # Betrag vor dem Symbol: "1.250,50 €"
    text = re.sub(rf"(?P<int>{_NUM})(?:,(?P<frac>\d{{1,2}}))?\s*(?P<sym>{syms})", repl, text)
    # Symbol vor dem Betrag: "€ 1.250,50"
    text = re.sub(rf"(?P<sym>{syms})\s*(?P<int>{_NUM})(?:,(?P<frac>\d{{1,2}}))?", repl, text)
    return text


def _expand_large_number_words(text: str) -> str:
    """'3 Mio.' -> 'drei Millionen'. Läuft vor der Abkürzungstabelle."""
    mapping = {
        "Mio.": ("Million", "Millionen"),
        "Mrd.": ("Milliarde", "Milliarden"),
        "Tsd.": ("Tausend", "Tausend"),
    }
    for abbr, (singular, plural) in mapping.items():
        pattern = rf"({_NUM})(?:,(\d+))?\s*{re.escape(abbr)}"

        def repl(m: re.Match[str], s: str = singular, p: str = plural) -> str:
            whole = _int_from_german(m.group(1))
            if m.group(2):
                return f"{cardinal(whole)} Komma {_digits(m.group(2))} {p}"
            return _amount(whole, s, p)

        text = re.sub(pattern, repl, text)
    return text


def _expand_units(text: str) -> str:
    units = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))

    def repl(m: re.Match[str]) -> str:
        singular, plural, article = UNITS[m.group("u")]
        if m.group("frac"):
            whole = _int_from_german(m.group("int"))
            return f"{cardinal(whole)} Komma {_digits(m.group('frac'))} {plural}"
        return _amount(_int_from_german(m.group("int")), singular, plural, article)

    return re.sub(rf"\b(?P<int>{_NUM})(?:,(?P<frac>\d+))?\s?(?P<u>{units})\b", repl, text)


def _expand_symbols(text: str) -> str:
    for sym in sorted(SYMBOLS, key=len, reverse=True):
        text = re.sub(rf"(?<=\d)\s?{re.escape(sym)}", f" {SYMBOLS[sym]}", text)
    text = text.replace("°C", " Grad Celsius")
    text = re.sub(r"\s&\s", " und ", text)
    text = re.sub(r"§\s*(?=\d)", "Paragraf ", text)
    return text


def _expand_abbreviations(text: str) -> str:
    keys = sorted(ABBREVIATIONS, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in keys))

    def repl(m: re.Match[str]) -> str:
        expansion = ABBREVIATIONS[m.group(0)]
        # Satzanfang: Großschreibung der Abkürzung übernehmen.
        if m.group(0)[0].isupper() and not expansion[0].isupper():
            expansion = expansion[0].upper() + expansion[1:]
        return expansion

    return pattern.sub(repl, text)


def _preceding_word(text: str, pos: int) -> str:
    m = re.search(r"([\wäöüÄÖÜß]+)\W*$", text[:pos])
    return m.group(1).lower() if m else ""


def _following_word(text: str, pos: int) -> str:
    m = re.match(r"\W*([\wäöüÄÖÜß]+)", text[pos:])
    return m.group(1) if m else ""


def _expand_ordinals(text: str) -> str:
    """'am 3. Mai' -> 'am dritten Mai'.

    Ein Punkt nach einer Zahl ist im Deutschen mehrdeutig -- er kann eine
    Ordinalzahl markieren oder ein Satzende sein. Ersetzt wird nur bei einem
    klaren Signal: Ordinal-auslösendes Wort davor, Monatsname danach, oder ein
    kleingeschriebenes Wort danach (ein Satz beginnt nie klein).
    """
    out: list[str] = []
    last = 0
    for m in re.finditer(r"\b(\d{1,4})\.(?!\d)", text):
        num = int(m.group(1))
        prev = _preceding_word(text, m.start())
        nxt = _following_word(text, m.end())
        is_ordinal = (
            prev in _OBLIQUE_TRIGGERS
            or prev in {"der", "die", "das", "ein", "eine"}
            or nxt in MONTH_NAMES
            or (nxt != "" and nxt[0].islower())
        )
        if not is_ordinal or num == 0 or num > 999:
            continue
        out.append(text[last : m.start()])
        out.append(ordinal(num, oblique=prev in _OBLIQUE_TRIGGERS))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def _expand_years(text: str) -> str:
    out: list[str] = []
    last = 0
    for m in re.finditer(r"\b(1[1-9]\d{2})\b", text):
        prev = _preceding_word(text, m.start())
        nxt = _following_word(text, m.end()).lower()
        if prev not in _YEAR_TRIGGERS and nxt not in _YEAR_TRIGGERS:
            continue
        out.append(text[last : m.start()])
        out.append(year(int(m.group(1))))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def _expand_one_before_noun(text: str) -> str:
    """'1 Minute' -> 'eine Minute'. Die Ziffer 1 als 'eins' zu lesen ist vor einem
    Substantiv immer falsch; das Genus raten wir über eine kurze Liste."""

    def repl(m: re.Match[str]) -> str:
        noun = m.group(1)
        article = "eine" if noun in FEMININE_NOUNS else "ein"
        return f"{article} {noun}"

    return re.sub(r"\b1\s+([A-ZÄÖÜ][\wäöüß]*)", repl, text)


def _expand_decimals(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        whole = _int_from_german(m.group(1))
        return f"{cardinal(whole)} Komma {_digits(m.group(2))}"

    return re.sub(rf"\b({_NUM}),(\d+)\b", repl, text)


def _expand_cardinals(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        value = _int_from_german(raw)
        # Sehr lange Ziffernfolgen (Telefon, IBAN, Codes) einzeln lesen.
        if len(raw.replace(".", "")) > 6:
            return _digits(raw.replace(".", ""))
        return cardinal(value)

    return re.sub(rf"\b{_NUM}\b", repl, text)


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Öffentliche API
# --------------------------------------------------------------------------

#: Regeln in Anwendungsreihenfolge. Spezifisch vor generisch.
_PIPELINE = (
    _clean_typography,
    _expand_urls_emails,
    _expand_dates,
    _expand_times,
    _expand_currency,
    _expand_large_number_words,
    _expand_units,
    _expand_symbols,
    _expand_abbreviations,
    _expand_ordinals,
    _expand_years,
    _expand_one_before_noun,
    _expand_decimals,
    _expand_cardinals,
    _collapse_whitespace,
)


def normalize_german(text: str) -> str:
    """Wandelt Ziffern, Symbole und Abkürzungen in ausgeschriebene deutsche Wörter."""
    for rule in _PIPELINE:
        text = rule(text)
    return text
