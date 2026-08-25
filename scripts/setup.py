"""Einrichtung von Cloney. Wird von install.ps1 und install.sh aufgerufen.

Die eigentliche Arbeit liegt hier und nicht in den Startskripten, weil sich
Python-Code plattformunabhängig prüfen lässt -- die Skripte drumherum tun nur
das, was vor dem Anlegen der Umgebung nötig ist.

Aufruf innerhalb der virtuellen Umgebung:

    .venv/bin/python scripts/setup.py            # Linux, macOS
    .venv\\Scripts\\python.exe scripts\\setup.py   # Windows
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: PyTorch von PyPI bringt nicht auf jeder Plattform einen Rechenkern für
#: Blackwell mit. Der cu128-Index tut es -- und schadet älteren Karten nicht.
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[90m", "\033[0m"


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def paint(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if supports_color() else text


def step(number: int, total: int, title: str) -> None:
    print(f"\n{paint(f'[{number}/{total}]', GREEN)} {title}")


def run(command: list[str], dry_run: bool = False) -> None:
    print(paint("      " + " ".join(command), DIM))
    if dry_run:
        return
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(
            paint(f"\nAbgebrochen: '{command[0]} ...' endete mit Code {result.returncode}", RED)
        )


def ensure_pip(dry_run: bool = False) -> None:
    """Mit uv angelegte Umgebungen bringen kein pip mit -- dann nachrüsten."""
    probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
    )
    if probe.returncode == 0:
        return
    print(paint("      pip fehlt in dieser Umgebung, wird nachgerüstet", DIM))
    run([sys.executable, "-m", "ensurepip", "--upgrade"], dry_run=dry_run)


def pip(*arguments: str, dry_run: bool = False) -> None:
    run([sys.executable, "-m", "pip", *arguments], dry_run=dry_run)


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def install_torch(index_url: str, dry_run: bool) -> None:
    pip("install", "--upgrade", "torch", "torchaudio", "--index-url", index_url, dry_run=dry_run)


def install_cloney(extras: str, dry_run: bool) -> None:
    target = f".[{extras}]" if extras else "."
    pip("install", "-e", target, dry_run=dry_run)


def create_env_file(dry_run: bool) -> None:
    target, template = ROOT / ".env", ROOT / ".env.example"
    if target.exists():
        print(paint("      .env ist bereits vorhanden und bleibt unverändert", DIM))
        return
    if not template.exists():
        print(paint("      .env.example fehlt -- übersprungen", DIM))
        return
    print(paint(f"      {template.name} -> .env", DIM))
    if not dry_run:
        shutil.copyfile(template, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cloney einrichten.")
    parser.add_argument("--torch-index", default=DEFAULT_TORCH_INDEX)
    parser.add_argument("--skip-torch", action="store_true", help="PyTorch nicht anfassen.")
    parser.add_argument("--extras", default="asr,f5", help="Extras, kommagetrennt. Leer = keine.")
    parser.add_argument("--dry-run", action="store_true", help="Nur zeigen, was liefe.")
    args = parser.parse_args()

    print(paint("Cloney einrichten", GREEN))
    print(paint(f"Python {sys.version.split()[0]} aus {sys.prefix}", DIM))
    if not in_virtualenv():
        print(
            paint(
                "\nWarnung: keine virtuelle Umgebung aktiv. Die Pakete landen systemweit.\n"
                "Besser über install.sh beziehungsweise install.ps1 starten.",
                YELLOW,
            )
        )

    ensure_pip(args.dry_run)

    total = 4 if args.skip_torch else 5
    number = 0

    if not args.skip_torch:
        number += 1
        step(number, total, "PyTorch installieren")
        print(
            paint(
                "      Aus dem CUDA-12.8-Index, weil PyTorch von PyPI nicht überall\n"
                "      einen Rechenkern für RTX-50-Karten (sm_120) mitbringt.",
                DIM,
            )
        )
        install_torch(args.torch_index, args.dry_run)

    number += 1
    step(number, total, f"Cloney installieren (Extras: {args.extras or 'keine'})")
    install_cloney(args.extras, args.dry_run)

    number += 1
    step(number, total, "Konfiguration anlegen")
    create_env_file(args.dry_run)

    number += 1
    step(number, total, "Umgebung prüfen")
    if args.dry_run:
        print(paint("      cloney doctor", DIM))
        diagnosis = 0
    else:
        diagnosis = subprocess.run(
            [sys.executable, "-m", "cloney.cli", "doctor"], cwd=ROOT
        ).returncode

    number += 1
    step(number, total, "Nächster Schritt")
    if diagnosis != 0:
        print(
            paint(
                "      Die Diagnose meldet offene Punkte. Sie stehen oben mit dem\n"
                "      jeweils passenden Befehl -- danach erneut: cloney doctor",
                YELLOW,
            )
        )
        return 1

    print(
        "      Einmal den ganzen Weg gehen, mit einer eigenen Aufnahme:\n"
        f"        {paint('cloney demo --audio meine_stimme.wav', GREEN)}\n\n"
        "      Danach die Oberfläche:\n"
        f"        {paint('cloney web', GREEN)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
