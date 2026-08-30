"""Server starten, Browser öffnen -- und sich wieder beenden lassen.

Sofort nach dem Start zu öffnen führt zuverlässig auf eine Fehlerseite: der
Server braucht einen Moment, bis er den Port bedient. Deshalb wird gewartet, bis
er wirklich antwortet -- und wenn er das nicht tut, wird eben nichts geöffnet.

Das Beenden ist der zweite Teil, und er ist der unangenehmere: uvicorn wartet
von sich aus unbegrenzt darauf, dass offene Verbindungen schließen.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import webbrowser
from types import FrameType

import httpx
import uvicorn

#: Wie lange beim Beenden auf offene Verbindungen und laufende Anfragen
#: gewartet wird, bevor der Prozess endet.
SHUTDOWN_SECONDS = 2.0

#: Zugabe, bevor die Notbremse zieht. Deckt den Weg vom abgelaufenen Warten bis
#: zum tatsächlichen Ende ab.
GNADENFRIST = 1.5


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


class FastShutdownServer(uvicorn.Server):
    """Ein Server, der sich beenden lässt, wenn man ihn darum bittet.

    Uvicorn wartet beim Beenden von sich aus unbegrenzt: erst auf offene
    Verbindungen, dann auf laufende Anfragen. Eine Anfrage, die gerade einen
    Satz synthetisiert, dauert eine Minute -- und sie läuft in einem Thread des
    Standard-Executors, den niemand abbrechen kann. ``asyncio.run`` wartet am
    Ende noch einmal genau darauf.

    Uvicorns eigene Notbremse (zweites Strg+C setzt ``force_exit``) greift dabei
    nicht durch: sie bricht nur die beiden Warteschleifen ab, nicht das
    ``wait_closed()`` dahinter und nicht den Executor. Genau so entsteht das
    Bild, das man im Terminal sieht -- 'Waiting for connections to close', und
    kein weiteres Strg+C hilft noch.

    Deshalb hier drei Dinge:

    * eine Frist für das geordnete Beenden (``timeout_graceful_shutdown``),
    * ein Wächter, der den Prozess beendet, wenn diese Frist verstreicht,
    * ein zweites Strg+C, das sofort wirkt.

    Was dabei verloren geht, ist wenig: Manifeste werden atomar geschrieben, und
    ein abgebrochener Renderlauf ist genau dafür gebaut, fortgesetzt zu werden.
    """

    #: Der Wächter, sobald das Beenden begonnen hat.
    wache: threading.Timer | None = None

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        if self.should_exit:
            self._sofort("Beendet.")
            return
        super().handle_exit(sig, frame)
        self.wache = threading.Timer(
            SHUTDOWN_SECONDS + GNADENFRIST,
            self._sofort,
            args=("Beendet -- offene Verbindungen wurden abgeschnitten.",),
        )
        # Ohne daemon hielte der Wächter den Prozess selbst am Leben.
        self.wache.daemon = True
        self.wache.start()

    def abpfiff(self) -> None:
        """Den Wächter abbestellen. Ist der Server von selbst durch, hat eine
        Notbremse nichts mehr zu suchen -- sie träfe sonst, was danach kommt."""
        if self.wache is not None:
            self.wache.cancel()
            self.wache = None

    def _sofort(self, meldung: str) -> None:
        print(f"\n{meldung}", flush=True)
        os._exit(0)


def build_server(app: object, host: str, port: int) -> uvicorn.Server:
    """Der Server, mit dem 'cloney web' läuft. Getrennt vom Starten, damit sich
    die Einstellungen prüfen lassen, ohne einen Port zu belegen."""
    return FastShutdownServer(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            timeout_graceful_shutdown=SHUTDOWN_SECONDS,
        )
    )


def serve(app: object, host: str, port: int) -> None:
    server = build_server(app, host, port)
    try:
        server.run()
    finally:
        server.abpfiff()
