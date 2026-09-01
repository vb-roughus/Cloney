"""Cloney über den Interpreter starten: ``python -m cloney``.

Der gewöhnliche Weg ist ``cloney``. Dahinter steckt keine eigene Anwendung,
sondern ein winziges Startprogramm, das pip beim Installieren erzeugt --
unter Windows eine ``cloney.exe`` von wenigen Kilobyte, die nichts weiter tut,
als den Interpreter mit ``cloney.cli:app`` aufzurufen. Sie ist unsigniert, und
genau daran scheitert sie auf Rechnern mit Smart App Control oder einer
WDAC-Richtlinie::

    Fehler beim Ausführen des Programms "cloney.exe":
    Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert

Blockiert ist dabei nur das Startprogramm, nicht Cloney: der Interpreter ist
signiert, und alles Übrige sind Python-Dateien, die keine Richtlinie ansieht.
Hierüber führt deshalb ein Weg daran vorbei, der nichts abschaltet und keine
Ausnahme in der Systemverwaltung braucht.
"""

from __future__ import annotations

from cloney.cli import app

if __name__ == "__main__":
    app()
