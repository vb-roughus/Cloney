"""Anbindung an F5-TTS, geprüft gegen ein eingeschleustes Ersatzmodul.

Das echte Modell lässt sich hier nicht laden -- es bringt Torch und einen
Modelldownload mit. Prüfbar ist aber das, was Cloney selbst verantwortet:
welche Parameter an ``infer`` gehen, ob der Seed durchgereicht wird, ob der
fehlende Referenztext früh und verständlich abbricht, und ob die Ausgabe des
Modells nicht in unsere Ausgabe durchschlägt.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cloney.engines.base import EngineError, VoiceRef
from cloney.engines.f5_german import F5_INFO, F5GermanEngine


@pytest.fixture
def fake_f5(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Ersetzt f5_tts.api durch eine Attrappe, die ihre Aufrufe mitschreibt."""
    calls: list[tuple[str, dict]] = []

    class FakeF5TTS:
        target_sample_rate = 24000

        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))

        def infer(self, **kwargs: object) -> tuple[np.ndarray, int, None]:
            # F5-TTS schreibt unaufgefordert nach stdout -- genau das soll die
            # Engine abfangen.
            print("gen_text 0 Ein Satz")
            calls.append(("infer", kwargs))
            return np.zeros(24000, dtype=np.float32), 24000, None

    api = types.ModuleType("f5_tts.api")
    api.F5TTS = FakeF5TTS
    monkeypatch.setitem(sys.modules, "f5_tts", types.ModuleType("f5_tts"))
    monkeypatch.setitem(sys.modules, "f5_tts.api", api)
    return calls


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[str, str]:
    ckpt = tmp_path / "model.safetensors"
    vocab = tmp_path / "vocab.txt"
    ckpt.write_bytes(b"nicht wirklich ein Modell")
    vocab.write_text("a\nb\n", encoding="utf-8")
    return str(ckpt), str(vocab)


def _engine(model_files: tuple[str, str], **kwargs: object) -> F5GermanEngine:
    ckpt, vocab = model_files
    return F5GermanEngine(ckpt_path=ckpt, vocab_path=vocab, **kwargs)  # type: ignore[arg-type]


def _voice(transcript: str = "Wortlaut der Referenzaufnahme.") -> VoiceRef:
    return VoiceRef("test", Path("referenz.wav"), transcript, duration_s=9.0)


# -- Budgetrechnung ---------------------------------------------------------


@pytest.mark.parametrize(
    ("reference_seconds", "expected"),
    [(6.0, 16.0), (9.0, 13.0), (12.0, 10.0), (30.0, 10.0)],
)
def test_chunk_budget_folgt_der_referenzlaenge(reference_seconds: float, expected: float) -> None:
    """F5-TTS erzeugt rund 22 s am Stück, Referenz eingerechnet. Je länger die
    Referenz, desto weniger bleibt für den Text."""
    assert F5_INFO.chunk_budget_seconds(reference_seconds, 20.0) == expected


def test_budget_ueberschreitet_den_wunschwert_nie() -> None:
    assert F5_INFO.chunk_budget_seconds(2.0, 8.0) == 8.0


# -- Aufrufübersetzung ------------------------------------------------------


def test_parameter_gehen_vollstaendig_an_infer(
    fake_f5: list[tuple[str, dict]], model_files: tuple[str, str]
) -> None:
    engine = _engine(model_files, nfe_step=16, speed=1.1, cfg_strength=2.5)
    engine.synthesize("Ein deutscher Satz.", _voice(), seed=4711)

    kwargs = next(payload for kind, payload in fake_f5 if kind == "infer")
    assert kwargs["gen_text"] == "Ein deutscher Satz."
    assert kwargs["ref_text"] == "Wortlaut der Referenzaufnahme."
    assert kwargs["seed"] == 4711
    assert kwargs["nfe_step"] == 16
    assert kwargs["speed"] == 1.1
    assert kwargs["cfg_strength"] == 2.5


def test_checkpoint_und_vokabular_werden_durchgereicht(
    fake_f5: list[tuple[str, dict]], model_files: tuple[str, str]
) -> None:
    ckpt, vocab = model_files
    _engine(model_files, model_config="F5TTS_Base")

    kwargs = next(payload for kind, payload in fake_f5 if kind == "init")
    assert kwargs["ckpt_file"] == ckpt
    assert kwargs["vocab_file"] == vocab
    assert kwargs["model"] == "F5TTS_Base"


def test_geraetewahl_bleibt_dem_modell_ueberlassen(
    fake_f5: list[tuple[str, dict]], model_files: tuple[str, str]
) -> None:
    _engine(model_files, device="auto")
    assert next(p for k, p in fake_f5 if k == "init")["device"] is None
    fake_f5.clear()
    _engine(model_files, device="cpu")
    assert next(p for k, p in fake_f5 if k == "init")["device"] == "cpu"


def test_ausgabe_ist_mono_float32(
    fake_f5: list[tuple[str, dict]], model_files: tuple[str, str]
) -> None:
    audio = _engine(model_files).synthesize("Text", _voice(), seed=1)
    assert audio.ndim == 1
    assert audio.dtype == np.float32


# -- Fehlerfälle ------------------------------------------------------------


def test_fehlender_referenztext_bricht_verstaendlich_ab(
    fake_f5: list[tuple[str, dict]], model_files: tuple[str, str]
) -> None:
    engine = _engine(model_files)
    with pytest.raises(EngineError) as exc:
        engine.synthesize("Text", _voice(transcript="  "), seed=1)
    assert "braucht" in str(exc.value)
    assert "--auto-transcript" in str(exc.value)
    assert not any(kind == "infer" for kind, _ in fake_f5)


def test_fehlendes_modell_nennt_die_installation(model_files: tuple[str, str]) -> None:
    with pytest.raises(EngineError, match=r"pip install -e"):
        _engine(model_files)


def test_fehlender_checkpoint_wird_gemeldet(fake_f5: list[tuple[str, dict]]) -> None:
    with pytest.raises(EngineError, match="Checkpoint nicht gefunden"):
        F5GermanEngine(ckpt_path="/gibt/es/nicht.safetensors", vocab_path="/auch/nicht.txt")


def test_ausgabe_des_modells_landet_in_der_fehlermeldung(
    fake_f5: list[tuple[str, dict]], model_files: tuple[str, str], capsys: pytest.CaptureFixture
) -> None:
    """Die Wortmeldungen von F5-TTS gehören nicht in unsere Ausgabe -- im
    Fehlerfall sind sie aber die nützlichste Spur."""
    engine = _engine(model_files)

    def explode(**_kwargs: object) -> None:
        print("gen_text 0 Ein Satz")
        raise ValueError("CUDA out of memory")

    engine._model.infer = explode  # type: ignore[method-assign]
    with pytest.raises(EngineError) as exc:
        engine.synthesize("Text", _voice(), seed=1)

    assert "CUDA out of memory" in str(exc.value)
    assert "gen_text 0" in str(exc.value)
    assert "gen_text" not in capsys.readouterr().out


def test_leeres_ergebnis_wird_nicht_stillschweigend_akzeptiert(
    fake_f5: list[tuple[str, dict]], model_files: tuple[str, str]
) -> None:
    engine = _engine(model_files)
    engine._model.infer = lambda **_k: (None, 24000, None)  # type: ignore[method-assign]
    with pytest.raises(EngineError, match="kein Audio"):
        engine.synthesize("Text", _voice(), seed=1)


# -- Auflösung der Modelldateien -------------------------------------------


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Ein Modell-Repo, das nur bekannte Dateinamen herausgibt."""
    available = ["F5TTS_Base/model_420000.safetensors", "vocab.txt"]
    calls: dict[str, list[str]] = {"download": [], "discover": []}

    def fake_download(repo_id: str, filename: str) -> str:
        calls["download"].append(filename)
        if filename not in available:
            raise EngineError(f"404 Entry Not Found for {filename}")
        return f"/cache/{filename}"

    def fake_discover(repo_id: str, prefer_bigvgan: bool = False) -> tuple[str, str]:
        calls["discover"].append(repo_id)
        return available[0], available[1]

    monkeypatch.setattr("cloney.engines.f5_german._download", fake_download)
    monkeypatch.setattr("cloney.engines.f5_german.discover_model_files", fake_discover)
    return calls


def test_ohne_vorgabe_wird_im_repo_nachgesehen(repo: dict[str, list[str]]) -> None:
    from cloney.engines.f5_german import resolve_model_files

    ckpt, vocab = resolve_model_files("aihpi/F5-TTS-German")
    assert ckpt == "/cache/F5TTS_Base/model_420000.safetensors"
    assert vocab == "/cache/vocab.txt"
    assert repo["discover"] == ["aihpi/F5-TTS-German"]


def test_gueltige_vorgabe_wird_genommen(repo: dict[str, list[str]]) -> None:
    from cloney.engines.f5_german import resolve_model_files

    ckpt, _ = resolve_model_files(
        "aihpi/F5-TTS-German", "F5TTS_Base/model_420000.safetensors", "vocab.txt"
    )
    assert ckpt == "/cache/F5TTS_Base/model_420000.safetensors"
    # Bei gültiger Vorgabe wird gar nicht erst gesucht.
    assert repo["discover"] == []


def test_veraltete_vorgabe_faellt_auf_die_suche_zurueck(repo: dict[str, list[str]]) -> None:
    """Der gemeldete Fall: eine alte Konfiguration nannte model_last.safetensors,
    das es im Repo nie gab. Ein 404 darf den Lauf nicht beenden, solange das
    gewünschte Modell erreichbar ist."""
    from cloney.engines.f5_german import resolve_model_files

    ckpt, vocab = resolve_model_files(
        "aihpi/F5-TTS-German", "F5TTS_Base/model_last.safetensors", "vocab.txt"
    )
    assert ckpt == "/cache/F5TTS_Base/model_420000.safetensors"
    assert vocab == "/cache/vocab.txt"
    assert repo["discover"] == ["aihpi/F5-TTS-German"]
    assert "F5TTS_Base/model_last.safetensors" in repo["download"]


def test_unerreichbares_repo_meldet_den_fehler(monkeypatch: pytest.MonkeyPatch) -> None:
    from cloney.engines.f5_german import resolve_model_files

    def explode(*_args: object, **_kwargs: object) -> tuple[str, str]:
        raise EngineError("Dateiliste von 'tippfehler/modell' nicht abrufbar: 404")

    monkeypatch.setattr("cloney.engines.f5_german.discover_model_files", explode)
    with pytest.raises(EngineError, match="nicht abrufbar"):
        resolve_model_files("tippfehler/modell")
