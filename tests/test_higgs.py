"""Higgs v3 gegen einen nachgebildeten Server.

Die Engine spricht HTTP mit einem fremden Prozess. Genau das lässt sich ohne
Modell prüfen: httpx nimmt einen eigenen Transport entgegen, und damit ist
belegbar, welche Felder tatsächlich über die Leitung gehen -- die Stelle, an der
diese Engine bisher falsch lag.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from cloney.engines.base import EngineError, VoiceRef
from cloney.engines.higgs import HIGGS_INFO, HiggsEngine, server_pfad

SR = 24000


def _wav(sekunden: float = 1.0, rate: int = SR) -> bytes:
    t = np.arange(int(sekunden * rate), dtype=np.float32) / rate
    puffer = io.BytesIO()
    sf.write(puffer, 0.2 * np.sin(2 * np.pi * 220 * t), rate, format="WAV", subtype="PCM_16")
    return puffer.getvalue()


def _voice(tmp_path: Path, transcript: str = "Wortlaut der Aufnahme.") -> VoiceRef:
    pfad = tmp_path / "reference.wav"
    pfad.write_bytes(_wav(3.0))
    return VoiceRef(name="anna", audio_path=pfad, transcript=transcript, duration_s=3.0)


class _Server:
    """Nimmt eine Anfrage entgegen und merkt sich, was ankam."""

    def __init__(self, status: int = 200, body: bytes | None = None) -> None:
        self.status = status
        self.body = _wav() if body is None else body
        self.payload: dict | None = None
        self.url: str | None = None

    def transport(self) -> httpx.MockTransport:
        def antworte(request: httpx.Request) -> httpx.Response:
            self.url = str(request.url)
            self.payload = json.loads(request.content)
            return httpx.Response(self.status, content=self.body)

        return httpx.MockTransport(antworte)


def _engine(server: _Server, **kw) -> HiggsEngine:  # noqa: ANN003
    return HiggsEngine(transport=server.transport(), **kw)


# -- Das Anfrageschema ------------------------------------------------------


def test_anfrage_entspricht_dem_kochbuch(tmp_path: Path) -> None:
    """Die Felder stammen aus dem Beispiel zum Klonen: references als Liste mit
    audio_path und text, dazu voice und die Abtastparameter."""
    server = _Server()
    _engine(server, reference_mode="path").synthesize("Hallo Welt.", _voice(tmp_path), seed=7)

    assert server.url.endswith("/v1/audio/speech")
    payload = server.payload
    assert payload["model"] == "bosonai/higgs-audio-v3-tts-4b"
    assert payload["voice"] == "default"
    assert payload["input"] == "Hallo Welt."
    assert payload["references"] == [
        {
            "audio_path": str((tmp_path / "reference.wav").resolve()),
            "text": "Wortlaut der Aufnahme.",
        }
    ]
    assert (payload["temperature"], payload["top_k"], payload["max_new_tokens"]) == (0.8, 50, 1024)


def test_kein_seed_im_payload(tmp_path: Path) -> None:
    """Die Schnittstelle kennt kein solches Feld. Ihn trotzdem mitzuschicken
    hieße, auf gut Glück ein unbekanntes Feld an einen fremden Server zu geben."""
    server = _Server()
    _engine(server).synthesize("Hallo.", _voice(tmp_path), seed=12345)
    assert "seed" not in server.payload


def test_engine_verspricht_keine_reproduzierbarkeit() -> None:
    """Ohne Seed in der Schnittstelle würfelt jeder Aufruf neu -- das gehört als
    Datum in EngineInfo, nicht als Fußnote in die Dokumentation."""
    assert HIGGS_INFO.reproducible_seed is False


def test_referenz_ohne_wortlaut_laesst_das_feld_weg(tmp_path: Path) -> None:
    server = _Server()
    _engine(server, reference_mode="path").synthesize("Hallo.", _voice(tmp_path, ""), seed=1)
    assert server.payload["references"] == [
        {"audio_path": str((tmp_path / "reference.wav").resolve())}
    ]


def test_base64_schickt_eine_data_url(tmp_path: Path) -> None:
    server = _Server()
    _engine(server, reference_mode="base64").synthesize("Hallo.", _voice(tmp_path), seed=1)
    referenz = server.payload["references"][0]
    assert "audio_path" not in referenz
    assert referenz["audio"].startswith("data:audio/wav;base64,")


# -- Der Pfad, den der Server sieht -----------------------------------------


def test_windows_pfad_wird_fuer_wsl_uebersetzt() -> None:
    """Cloney läuft unter Windows, der Server in WSL. 'C:\\...' ist für ihn kein
    gültiger Pfad -- dieselbe Datei liegt dort unter /mnt/c/..."""
    assert (
        server_pfad(r"C:\Users\rolf\Cloney\data\voices\anna\reference.wav", "wsl")
        == "/mnt/c/Users/rolf/Cloney/data/voices/anna/reference.wav"
    )


def test_auto_uebersetzt_nur_unter_windows() -> None:
    windows_pfad = r"D:\ton\reference.wav"
    assert server_pfad(windows_pfad, "auto", windows=True) == "/mnt/d/ton/reference.wav"
    assert server_pfad(windows_pfad, "auto", windows=False) == windows_pfad


def test_path_reicht_unveraendert_weiter() -> None:
    """Cloney und Server auf demselben Linux-System: nichts zu übersetzen."""
    assert server_pfad(r"C:\ton\reference.wav", "path") == r"C:\ton\reference.wav"
    assert server_pfad("/daten/reference.wav", "auto", windows=True) == "/daten/reference.wav"


# -- Was der Server zurückgibt ----------------------------------------------


def test_antwort_wird_zu_mono_audio(tmp_path: Path) -> None:
    server = _Server(body=_wav(2.0))
    audio = _engine(server).synthesize("Hallo.", _voice(tmp_path), seed=1)
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(2 * SR, rel=0.01)


def test_abweichende_samplerate_des_servers_gilt(tmp_path: Path) -> None:
    """Nicht unsere Annahme, sondern was tatsächlich ankommt."""
    server = _Server(body=_wav(1.0, rate=44100))
    engine = _engine(server)
    engine.synthesize("Hallo.", _voice(tmp_path), seed=1)
    assert engine.info.sample_rate == 44100


def test_fehlerantwort_nennt_den_grund_des_servers(tmp_path: Path) -> None:
    """Der Text der Gegenstelle sagt, welches Feld nicht passt -- er darf nicht
    hinter einer eigenen Meldung verschwinden."""
    server = _Server(status=400, body=b'{"error":"unknown field: references"}')
    with pytest.raises(EngineError, match="unknown field: references"):
        _engine(server).synthesize("Hallo.", _voice(tmp_path), seed=1)


def test_unlesbare_antwort_wird_erklaert(tmp_path: Path) -> None:
    server = _Server(body=b"<html>Gateway Timeout</html>")
    with pytest.raises(EngineError, match="kein lesbarer Ton"):
        _engine(server).synthesize("Hallo.", _voice(tmp_path), seed=1)


def test_nicht_erreichbarer_server_nennt_den_startbefehl(tmp_path: Path) -> None:
    def verweigert(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    engine = HiggsEngine(transport=httpx.MockTransport(verweigert))
    with pytest.raises(EngineError, match="sgl-omni serve"):
        engine.synthesize("Hallo.", _voice(tmp_path), seed=1)
