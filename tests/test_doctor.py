"""Prüfungen der Umgebungsdiagnose.

Der Wert von ``doctor`` liegt nicht darin, dass es läuft, sondern darin, dass es
zu jedem Befund einen Befehl nennt und dass es prüft statt Versionen zu
vergleichen. Genau das wird hier festgehalten.
"""

from __future__ import annotations

import sys
import types

import pytest

from cloney.config import Settings
from cloney.doctor import Report, check_pipeline, check_torch, run_checks


def test_jeder_befund_nennt_einen_weg(settings: Settings) -> None:
    report = run_checks(settings)
    for check in report.results:
        if check.status != "ok":
            assert check.remedy, f"'{check.name}' meldet ein Problem ohne Ausweg"


def test_durchstich_belegt_die_verkettung(settings: Settings) -> None:
    """Der Selbsttest muss echtes Audio erzeugen, nicht nur Importe prüfen."""
    report = Report()
    check_pipeline(report)
    result = report.results[0]
    assert result.status == "ok"
    assert "Chunks" in result.detail
    assert "erzeugt" in result.detail


def test_fehlendes_torch_ist_kein_fehler(settings: Settings, monkeypatch) -> None:  # noqa: ANN001
    """Ohne PyTorch läuft der Kern weiterhin -- das ist ein Hinweis, kein Fehler."""
    monkeypatch.setitem(sys.modules, "torch", None)
    report = Report()
    check_torch(report)
    assert report.results[0].status == "warn"


def _fake_torch(arch_list: list[str], capability: tuple[int, int]) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.__version__ = "2.7.0"
    module.version = types.SimpleNamespace(cuda="12.8")
    module.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _i: "NVIDIA GeForce RTX 5080",
        get_device_capability=lambda _i: capability,
        get_arch_list=lambda: arch_list,
        get_device_properties=lambda _i: types.SimpleNamespace(total_memory=16 * 1024**3),
    )
    return module


def test_torch_ohne_rechenkern_fuer_die_karte_ist_ein_fehler(monkeypatch) -> None:  # noqa: ANN001
    """Der eigentliche Blackwell-Fallstrick: PyTorch startet, kennt die Karte
    aber nicht und rechnet still auf der CPU weiter."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["sm_80", "sm_90"], (12, 0)))
    report = Report()
    check_torch(report)

    result = report.results[0]
    assert result.status == "fail"
    assert "sm_120" in result.detail
    assert "cu128" in result.remedy


def test_torch_mit_passendem_rechenkern_ist_in_ordnung(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["sm_90", "sm_120"], (12, 0)))
    report = Report()
    check_torch(report)

    result = report.results[0]
    assert result.status == "ok"
    assert "sm_120" in result.detail
    assert "RTX 5080" in result.detail


@pytest.mark.parametrize(
    ("files", "expected_ckpt"),
    [
        (
            ["vocab.txt", "F5TTS_Base/model_1200000.safetensors"],
            "F5TTS_Base/model_1200000.safetensors",
        ),
        (
            ["vocab.txt", "model_last.safetensors", "model_500000.safetensors"],
            "model_last.safetensors",
        ),
        (
            ["vocab.txt", "model_100000.safetensors", "model_2400000.safetensors"],
            "model_2400000.safetensors",
        ),
        (["vocab.txt", "ckpts/model_900000.pt"], "ckpts/model_900000.pt"),
    ],
)
def test_checkpoint_wird_gefunden_statt_geraten(files: list[str], expected_ckpt: str) -> None:
    """Die deutschen Finetunes legen ihre Dateien unterschiedlich ab. Einen
    Namen zu raten scheitert je nach Repo -- also wird gewählt."""
    from cloney.engines.f5_german import choose_model_files

    ckpt, vocab = choose_model_files(files)
    assert ckpt == expected_ckpt
    assert vocab == "vocab.txt"


def test_safetensors_schlaegt_pt_und_bigvgan_bleibt_aussen_vor() -> None:
    from cloney.engines.f5_german import choose_model_files

    files = [
        "vocab.txt",
        "F5TTS_Base/model_1200000.safetensors",
        "F5TTS_Base_bigvgan/model_1250000.pt",
    ]
    assert choose_model_files(files)[0] == "F5TTS_Base/model_1200000.safetensors"
    assert (
        choose_model_files(files, prefer_bigvgan=True)[0] == "F5TTS_Base_bigvgan/model_1250000.pt"
    )


def test_repo_ohne_checkpoint_wird_gemeldet() -> None:
    from cloney.engines.base import EngineError
    from cloney.engines.f5_german import choose_model_files

    with pytest.raises(EngineError, match="keinen Checkpoint"):
        choose_model_files(["vocab.txt", "README.md"])
