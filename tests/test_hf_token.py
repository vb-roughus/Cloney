"""Das Hub-Token: gelesen, wo man es hinschreibt; wirksam, wo es gebraucht wird.

Nötig ist es für nichts -- die Modelle, die Cloney holt, sind öffentlich. Es
hebt nur die Ratengrenze für unangemeldete Zugriffe an und lässt die Warnung
verstummen, die huggingface_hub bei jedem Lauf ausgibt.

Die Falle steckt woanders: pydantic-settings liest die .env in das
Settings-Objekt und **nicht** in die Umgebung. huggingface_hub liest
ausschließlich die Umgebung. Wer also ``HF_TOKEN`` in die .env schreibt -- das
Naheliegende, weil die Warnung genau diesen Namen nennt -- hätte ohne
``apply_hf_token`` weiterhin die Warnung und keine Wirkung.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from cloney.config import Settings, apply_hf_token


@pytest.fixture(autouse=True)
def _ohne_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CLONEY_HF_TOKEN", raising=False)


def _mit_env(tmp_path: Path, inhalt: str, monkeypatch: pytest.MonkeyPatch) -> Settings:
    (tmp_path / ".env").write_text(inhalt, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return Settings()


def test_ohne_token_bleibt_es_leer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    einstellungen = _mit_env(tmp_path, "CLONEY_ENGINE=dummy\n", monkeypatch)

    assert einstellungen.hf_token == ""
    assert apply_hf_token(einstellungen) is False
    assert "HF_TOKEN" not in os.environ


def test_der_name_aus_der_warnung_wird_gelesen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """huggingface_hub nennt in seiner Warnung 'HF_TOKEN'. Genau das schreibt
    man dann in die .env -- und genau das muss ankommen."""
    einstellungen = _mit_env(tmp_path, "HF_TOKEN=hf_aus_der_env\n", monkeypatch)

    assert einstellungen.hf_token == "hf_aus_der_env"


def test_auch_der_cloney_eigene_name_wird_gelesen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    einstellungen = _mit_env(tmp_path, "CLONEY_HF_TOKEN=hf_mit_praefix\n", monkeypatch)

    assert einstellungen.hf_token == "hf_mit_praefix"


def test_das_token_landet_in_der_umgebung(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Der eigentliche Punkt: ohne diesen Schritt stünde es nur im Objekt, und
    huggingface_hub sieht ausschließlich die Umgebung."""
    einstellungen = _mit_env(tmp_path, "HF_TOKEN=hf_aus_der_env\n", monkeypatch)
    assert "HF_TOKEN" not in os.environ

    assert apply_hf_token(einstellungen) is True
    assert os.environ["HF_TOKEN"] == "hf_aus_der_env"


def test_die_umgebung_schlaegt_die_datei(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wer sich mit 'huggingface-cli login' angemeldet oder die Variable im
    Terminal gesetzt hat, meinte das so."""
    einstellungen = _mit_env(tmp_path, "HF_TOKEN=hf_aus_der_env\n", monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_aus_dem_terminal")

    assert apply_hf_token(einstellungen) is True
    assert os.environ["HF_TOKEN"] == "hf_aus_dem_terminal"


def test_leerraum_zaehlt_nicht(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    einstellungen = _mit_env(tmp_path, "HF_TOKEN=   \n", monkeypatch)

    assert apply_hf_token(einstellungen) is False
    assert "HF_TOKEN" not in os.environ


def test_der_trainingsprozess_erbt_das_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Das Training läuft in einem eigenen Prozess, und F5 lädt dort selbst
    nach. Ein durchgereichtes Argument erreichte ihn nicht -- die Umgebung
    schon."""
    einstellungen = _mit_env(tmp_path, "HF_TOKEN=hf_aus_der_env\n", monkeypatch)
    apply_hf_token(einstellungen)

    kind = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('HF_TOKEN', ''))"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert kind.stdout.strip() == "hf_aus_der_env"
