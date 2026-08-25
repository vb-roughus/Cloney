"""Das Einrichtungsskript darf keine Befehle empfehlen, die nicht laufen.

'cloney' allein liegt nur im Suchpfad, solange die virtuelle Umgebung aktiviert
ist. Ein Installer, der damit endet, schickt den Nutzer in eine Fehlermeldung.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _laden():
    spec = importlib.util.spec_from_file_location("cloney_setup", ROOT / "scripts" / "setup.py")
    assert spec and spec.loader
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


@pytest.fixture(scope="module")
def setup_modul():
    return _laden()


def test_empfohlener_befehl_ist_ohne_aktivierung_aufrufbar(setup_modul, monkeypatch) -> None:  # noqa: ANN001
    """Der genannte Weg muss den Interpreter der Umgebung mitbringen, nicht auf
    einen Suchpfad hoffen."""
    befehl = setup_modul.cloney_command()
    assert befehl != "cloney"
    assert "cloney" in befehl


def test_ohne_ausfuehrbare_datei_wird_das_modul_genannt(setup_modul, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Fehlt die erzeugte Datei -- etwa weil die Installation abbrach --, muss
    der Aufruf über das Modul genannt werden statt ein toter Pfad."""
    monkeypatch.setattr(setup_modul.sys, "executable", str(tmp_path / "python"))
    befehl = setup_modul.cloney_command()
    assert befehl.endswith("-m cloney.cli")


def test_aktivierungshinweis_passt_zur_plattform(setup_modul, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(setup_modul.platform, "system", lambda: "Windows")
    assert "Activate.ps1" in setup_modul.activation_hint()

    monkeypatch.setattr(setup_modul.platform, "system", lambda: "Linux")
    assert setup_modul.activation_hint() == "source .venv/bin/activate"


def test_windows_sucht_die_exe(setup_modul, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "cloney.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(setup_modul.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_modul.sys, "executable", str(scripts / "python.exe"))

    assert setup_modul.cloney_command().endswith("cloney.exe")


def test_modul_laesst_sich_ohne_nebenwirkung_laden() -> None:
    """Es darf beim Import nichts installieren -- sonst wäre schon dieser Test
    eine Installation."""
    vorher = list(sys.argv)
    _laden()
    assert sys.argv == vorher
