"""Bereitschaftsprüfung vor dem Öffnen des Browsers.

Sofort nach dem Serverstart zu öffnen führt zuverlässig auf eine Fehlerseite.
Geprüft wird deshalb gegen einen echten HTTP-Server statt gegen eine Attrappe.
"""

from __future__ import annotations

import http.server
import threading
import time

import pytest

from cloney.web.launch import open_browser_when_ready, wait_until_ready


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
