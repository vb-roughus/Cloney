"""Starten und Beenden der Oberfläche.

Sofort nach dem Serverstart den Browser zu öffnen führt zuverlässig auf eine
Fehlerseite. Geprüft wird deshalb gegen einen echten HTTP-Server statt gegen
eine Attrappe -- und das Beenden ebenso: dass eine offene Verbindung den Server
nicht festhält, zeigt sich nur an einem laufenden.
"""

from __future__ import annotations

import contextlib
import http.server
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from cloney.web.launch import (
    GNADENFRIST,
    SHUTDOWN_SECONDS,
    build_server,
    open_browser_when_ready,
    wait_until_ready,
)


@pytest.fixture
def live_server():
    """Ein echter HTTP-Server auf einem freien Port."""
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_erkennt_laufenden_server(live_server: str) -> None:
    assert wait_until_ready(live_server, timeout=5.0) is True


def test_gibt_bei_totem_port_auf() -> None:
    """Ohne Server darf nicht endlos gewartet werden."""
    started = time.monotonic()
    assert wait_until_ready("http://127.0.0.1:9", timeout=1.0, interval=0.05) is False
    assert time.monotonic() - started < 3.0


def test_wartet_auf_einen_spaeter_startenden_server() -> None:
    """Der eigentliche Fall: der Browser soll erst öffnen, wenn der Server steht."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Timer(
        0.4, lambda: threading.Thread(target=server.serve_forever, daemon=True).start()
    ).start()
    try:
        assert wait_until_ready(url, timeout=6.0, interval=0.1) is True
    finally:
        server.shutdown()
        server.server_close()


def test_browser_wird_erst_nach_bereitschaft_geoeffnet(
    live_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr("cloney.web.launch.webbrowser.open", lambda url: opened.append(url))

    open_browser_when_ready(live_server, timeout=5.0).join(timeout=6.0)
    assert opened == [live_server]


def test_ohne_server_wird_nichts_geoeffnet(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("cloney.web.launch.webbrowser.open", lambda url: opened.append(url))

    open_browser_when_ready("http://127.0.0.1:9", timeout=0.5).join(timeout=5.0)
    assert opened == []


# -- Beenden ----------------------------------------------------------------

#: Ein Server mit einer Anfrage, die nicht zurückkommt -- wie eine laufende
#: Synthese. Entscheidend ist die gewöhnliche ``def``-Route: FastAPI führt sie
#: in einem Thread des Standard-Executors aus, und auf den wartet ``asyncio.run``
#: am Ende, ohne ihn abbrechen zu können. Genau daran hing der Server.
HAENGENDER_SERVER = """
import time
from fastapi import FastAPI
from cloney.web.launch import build_server

app = FastAPI()

@app.get("/")
def bereit():
    return {"ok": True}

@app.get("/langsam")
def langsam():
    time.sleep(120)
    return {"fertig": True}

build_server(app, "127.0.0.1", PORT).run()
"""


def _freier_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _app(scope, receive, send) -> None:  # noqa: ANN001
    await receive()
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"da"})


def test_frist_statt_endlosem_warten() -> None:
    """Ohne Frist wartet uvicorn unbegrenzt auf offene Verbindungen."""
    assert build_server(_app, "127.0.0.1", 8080).config.timeout_graceful_shutdown == (
        SHUTDOWN_SECONDS
    )


def test_zweites_signal_beendet_ohne_umweg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uvicorns eigene Notbremse bricht nur die Warteschleifen ab, nicht das
    wait_closed() dahinter. Das zweite Strg+C muss deshalb härter sein."""
    beendet: list[int] = []
    monkeypatch.setattr("cloney.web.launch.os._exit", lambda code: beendet.append(code))
    server = build_server(_app, "127.0.0.1", 8080)

    server.handle_exit(signal.SIGINT, None)
    server.abpfiff()  # sonst zöge der Wächter mitten im nächsten Test
    assert server.should_exit is True
    assert beendet == []

    server.handle_exit(signal.SIGINT, None)
    assert beendet == [0]


def test_der_waechter_ist_kein_eigener_grund_zu_bleiben() -> None:
    """Zweierlei: der Wächter-Thread darf den Prozess nicht selbst am Leben
    halten, und er muss abbestellbar sein -- ist der Server von selbst durch,
    träfe die Notbremse sonst, was danach kommt."""
    server = build_server(_app, "127.0.0.1", 8080)
    server.handle_exit(signal.SIGINT, None)

    assert server.wache is not None
    assert server.wache.daemon

    server.abpfiff()
    assert server.wache is None


@pytest.mark.skipif(os.name == "nt", reason="SIGINT an einen Subprozess ist hier anders")
def test_haengende_anfrage_haelt_den_server_nicht_fest(tmp_path: Path) -> None:
    """Der Fall aus dem Terminal: eine Anfrage rechnet noch, Strg+C kommt --
    und der Server war danach nur noch mit dem Taskmanager loszuwerden.

    Geprüft im Subprozess, weil das Ende hier ein echtes Prozessende ist.
    """
    port = _freier_port()
    skript = tmp_path / "server.py"
    skript.write_text(HAENGENDER_SERVER.replace("PORT", str(port)), encoding="utf-8")

    prozess = subprocess.Popen(
        [sys.executable, str(skript)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        assert wait_until_ready(f"http://127.0.0.1:{port}", timeout=20.0)

        def haengende_anfrage() -> None:
            with contextlib.suppress(Exception):
                httpx.get(f"http://127.0.0.1:{port}/langsam", timeout=200, trust_env=False)

        threading.Thread(target=haengende_anfrage, daemon=True).start()
        time.sleep(1.5)

        begonnen = time.monotonic()
        prozess.send_signal(signal.SIGINT)
        prozess.wait(timeout=SHUTDOWN_SECONDS + GNADENFRIST + 10.0)

        assert time.monotonic() - begonnen < SHUTDOWN_SECONDS + GNADENFRIST + 5.0
    finally:
        prozess.kill()
        prozess.wait(timeout=10)
