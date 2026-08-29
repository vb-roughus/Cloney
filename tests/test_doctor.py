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


# -- Fehlermeldungen von torchcodec ----------------------------------------

#: Gekürzte, aber wortgetreue Fassung des Fehlers, den torchcodec unter Windows
#: wirft, wenn nur der statische FFmpeg-Build installiert ist.
_TORCHCODEC_MISSING_DLL = (
    r"Failed to create AudioDecoder for C:\Temp\probe.wav: "
    "Could not load libtorchcodec. Likely causes:\n"
    r"""
  1. FFmpeg is not properly installed in your environment. We support
     versions 4, 5, 6, 7, 8, and 9. On Windows, ensure you've installed the
     "full-shared" version which ships DLLs.
  2. The PyTorch version (2.11.0+cu128) is not compatible with
     this version of TorchCodec.

[start of libtorchcodec loading traceback]
FFmpeg version 9:
FileNotFoundError: Could not find module 'libtorchcodec_core9.dll' (or one of its dependencies).
FFmpeg version 8:
FileNotFoundError: Could not find module 'libtorchcodec_core8.dll' (or one of its dependencies).
[end of libtorchcodec loading traceback]."""
)


def test_fehlerlawine_wird_auf_eine_zeile_eingedampft() -> None:
    """torchcodec probiert sechs FFmpeg-Versionen durch und legt jeden Fehlschlag
    offen. Ungekürzt macht das den Diagnosebericht unlesbar."""
    from cloney.doctor import summarise_decoder_error

    summary, _ = summarise_decoder_error(_TORCHCODEC_MISSING_DLL)
    assert "\n" not in summary
    assert len(summary) <= 200
    assert "Likely causes" not in summary
    assert "AudioDecoder" in summary


def test_fehlende_bibliotheken_verweisen_auf_den_shared_build() -> None:
    """Der eigentliche Fallstrick: 'winget install Gyan.FFmpeg' liefert den
    statischen Build. Der legt ffmpeg.exe ab und sonst nichts -- torchcodec
    braucht die DLLs."""
    from cloney.doctor import summarise_decoder_error

    _, remedy = summarise_decoder_error(_TORCHCODEC_MISSING_DLL)
    assert "Gyan.FFmpeg.Shared" in remedy
    assert "Konsole neu öffnen" in remedy


def test_abi_fehler_verweist_auf_die_versionen() -> None:
    from cloney.doctor import summarise_decoder_error

    _, remedy = summarise_decoder_error("Could not load: undefined symbol: _ZN3c10abc")
    assert "torchcodec" in remedy
    assert "PyTorch-Version" in remedy
    assert "Gyan.FFmpeg.Shared" not in remedy


def test_leerer_fehler_bricht_nicht_ab() -> None:
    from cloney.doctor import summarise_decoder_error

    summary, remedy = summarise_decoder_error("")
    assert summary
    assert remedy


def test_nur_ffmpeg_exe_ist_kein_gruenes_licht(monkeypatch) -> None:  # noqa: ANN001
    """Der Fehler in der ersten Fassung: ffmpeg.exe im Suchpfad wurde als 'in
    Ordnung' gemeldet, während jedes Laden einer Audiodatei scheiterte."""
    from cloney.doctor import Report, check_ffmpeg

    monkeypatch.setattr("cloney.doctor.platform.system", lambda: "Windows")
    monkeypatch.setattr("cloney.doctor.shutil.which", lambda _name: r"C:\ffmpeg\bin\ffmpeg.exe")
    monkeypatch.setattr("cloney.doctor.find_ffmpeg_shared_libraries", list)

    report = Report()
    check_ffmpeg(report)
    result = report.results[0]
    assert result.status == "warn"
    assert "statische Build" in result.detail
    assert "Gyan.FFmpeg.Shared" in result.remedy


def test_vorhandene_bibliotheken_sind_in_ordnung(monkeypatch) -> None:  # noqa: ANN001
    from cloney.doctor import Report, check_ffmpeg

    monkeypatch.setattr(
        "cloney.doctor.find_ffmpeg_shared_libraries",
        lambda: [r"C:\ffmpeg\bin\avcodec-61.dll"],
    )
    report = Report()
    check_ffmpeg(report)
    assert report.results[0].status == "ok"


# -- Higgs hinter einem Server ----------------------------------------------


def test_higgs_meldet_wenn_der_modellname_nicht_passt(monkeypatch) -> None:  # noqa: ANN001
    """Ein OpenAI-kompatibler Server lehnt unbekannte Modellnamen ab. Ohne diese
    Prüfung taucht die Meldung erst mitten in einem Renderlauf auf."""
    from cloney.doctor import Report, check_higgs

    monkeypatch.setattr(
        "cloney.doctor.served_models", lambda *_a, **_k: ["bosonai/higgs-audio-v3-tts-4b"]
    )
    report = Report()
    check_higgs(report, Settings(higgs_model="higgs-audio-v3-tts"))

    eintrag = next(e for e in report.results if e.name == "Engine higgs")
    assert eintrag.status == "fail"
    assert "bosonai/higgs-audio-v3-tts-4b" in eintrag.remedy


def test_higgs_ist_unter_windows_kein_ausschlusskriterium(monkeypatch) -> None:  # noqa: ANN001
    """Mit WSL läuft der Server auch dort. Früher hat die Prüfung hier
    grundsätzlich abgewunken."""
    from cloney.doctor import Report, check_higgs

    monkeypatch.setattr("cloney.doctor.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "cloney.doctor.served_models", lambda *_a, **_k: ["bosonai/higgs-audio-v3-tts-4b"]
    )
    report = Report()
    check_higgs(report, Settings())

    assert next(e for e in report.results if e.name == "Engine higgs").status == "ok"


def test_higgs_nennt_den_serverparameter_fuer_den_pfadweg(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Der Pfadweg braucht --allowed-local-media-path. Fehlt er, lehnt der
    Server die Referenz ab -- und zwar erst beim ersten Satz."""
    from cloney.doctor import Report, check_higgs

    monkeypatch.setattr("cloney.doctor.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "cloney.doctor.served_models", lambda *_a, **_k: ["bosonai/higgs-audio-v3-tts-4b"]
    )
    report = Report()
    check_higgs(report, Settings(data_dir=tmp_path / "data", higgs_reference_mode="wsl"))

    eintrag = next(e for e in report.results if e.name == "Higgs-Referenz")
    assert eintrag.status == "warn"
    assert "--allowed-local-media-path" in eintrag.remedy


def test_higgs_mit_base64_braucht_keinen_serverparameter(monkeypatch) -> None:  # noqa: ANN001
    from cloney.doctor import Report, check_higgs

    monkeypatch.setattr(
        "cloney.doctor.served_models", lambda *_a, **_k: ["bosonai/higgs-audio-v3-tts-4b"]
    )
    report = Report()
    check_higgs(report, Settings())

    eintrag = next(e for e in report.results if e.name == "Higgs-Referenz")
    assert eintrag.status == "ok"
    assert "Data-URL" in eintrag.detail


def test_higgs_ohne_server_nennt_den_startbefehl(monkeypatch) -> None:  # noqa: ANN001
    from cloney.doctor import Report, check_higgs

    monkeypatch.setattr("cloney.doctor.served_models", lambda *_a, **_k: None)
    report = Report()
    check_higgs(report, Settings())

    eintrag = next(e for e in report.results if e.name == "Engine higgs")
    assert eintrag.status == "warn"
    assert "sgl-omni serve" in eintrag.remedy
