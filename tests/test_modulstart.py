"""``python -m cloney`` muss laufen -- es ist der Weg an einer Blockade vorbei.

Unter Windows blockieren Smart App Control und WDAC-Richtlinien die
``cloney.exe``, die pip beim Installieren erzeugt: ein unsigniertes
Startprogramm von wenigen Kilobyte. Es wird bei jeder neuen virtuellen Umgebung
frisch erzeugt, also hilft auch keine einmalige Freigabe.

Der Aufruf über den Interpreter umgeht das, weil dabei nichts Unsigniertes
ausgeführt wird. Damit ist er kein Nebenweg, sondern für manche Rechner der
einzige -- und gehört deshalb geprüft wie jeder andere Einstieg.

Geprüft wird in einem echten Unterprozess. Ein Import von ``cloney.__main__``
sagte nichts darüber aus, ob ``-m`` funktioniert: Python sucht dafür eine Datei
dieses Namens im Paket, und ob die da ist, merkt man nur beim Ausführen.
"""

from __future__ import annotations

import subprocess
import sys


def _cloney(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cloney", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_python_m_cloney_startet() -> None:
    ergebnis = _cloney("--help")

    assert ergebnis.returncode == 0, ergebnis.stderr
    assert "Voice Cloning" in ergebnis.stdout


def test_der_modulaufruf_fuehrt_wirklich_etwas_aus() -> None:
    """Eine Hilfe zeigt auch ein leeres Gerüst. Hier läuft ein Befehl."""
    ergebnis = _cloney("engines")

    assert ergebnis.returncode == 0, ergebnis.stderr
    assert "dummy" in ergebnis.stdout


def test_es_ist_dieselbe_anwendung_wie_hinter_dem_befehl() -> None:
    """Sonst liefe über den Interpreter etwas anderes als über 'cloney' --
    und die Anleitung für den blockierten Fall führte in die Irre."""
    from cloney.__main__ import app as ueber_modul
    from cloney.cli import app as ueber_befehl

    assert ueber_modul is ueber_befehl
