"""Browser öffnen, sobald die Oberfläche tatsächlich antwortet.

Sofort nach dem Start zu öffnen führt zuverlässig auf eine Fehlerseite: der
Server braucht einen Moment, bis er den Port bedient. Deshalb wird gewartet, bis
er wirklich antwortet -- und wenn er das nicht tut, wird eben nichts geöffnet.
"""

from __future__ import annotations

import contextlib
import threading
import time
import webbrowser

import httpx


def wait_until_ready(url: str, timeout: float = 20.0, interval: float = 0.25) -> bool:
    """Wartet, bis unter ``url`` jemand antwortet. False bei Zeitüberschreitung."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # trust_env=False: der Server läuft lokal, ein HTTP-Proxy der
            # Umgebung hat hier nichts verloren.
            httpx.get(url, timeout=1.0, trust_env=False)
        except httpx.RequestError:
            time.sleep(interval)
        else:
            return True
    return False


def open_browser_when_ready(url: str, timeout: float = 20.0) -> threading.Thread:
    """Startet einen Hintergrund-Thread, der den Browser öffnet, sobald es geht."""

    def work() -> None:
        if wait_until_ready(url, timeout):
            # Ohne Desktop gibt es keinen Browser. Kein Grund abzubrechen --
            # die Adresse steht ohnehin im Terminal.
            with contextlib.suppress(Exception):
                webbrowser.open(url)

    thread = threading.Thread(target=work, daemon=True, name="browser-open")
    thread.start()
    return thread
