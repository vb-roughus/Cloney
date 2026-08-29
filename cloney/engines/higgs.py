"""Higgs Audio v3 über einen lokalen SGLang-Omni-Server.

Das Modell läuft als eigener Prozess::

    sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000

und spricht die OpenAI-kompatible ``/v1/audio/speech``-Schnittstelle. Für Cloney
ist das der bequemste Weg: die Modellverwaltung liegt vollständig beim Server,
und der VRAM wird freigegeben, indem man diesen Prozess beendet.

Das Anfrageschema folgt dem Kochbuch von SGLang-Omni: ``references`` ist eine
Liste aus ``{"audio_path": ..., "text": ...}``, dazu ``voice``, ``temperature``,
``top_k`` und ``max_new_tokens``. Zwei Dinge sind daran wichtig:

* ``voice`` steht in jedem dokumentierten Beispiel, auch im Beispiel zum
  Klonen. Hier fehlte es bisher. Ob der Server ohne dieses Feld ablehnt, ist
  nicht geprüft -- mitzuschicken kostet nichts, es wegzulassen wäre geraten.
* ``audio_path`` liest der **Server**, nicht Cloney. Läuft Cloney unter Windows
  und der Server in WSL, ist ``C:\\...`` für ihn kein gültiger Pfad; er sieht
  dieselbe Datei unter ``/mnt/c/...``. Genau das übersetzt ``server_pfad``.

Einen Seed nimmt die Schnittstelle nicht entgegen. Higgs würfelt bei jedem Aufruf
neu -- das ist als ``reproducible_seed=False`` in ``EngineInfo`` vermerkt, damit
Pipeline und Oberfläche keine Reproduzierbarkeit versprechen, die es hier nicht
gibt.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import replace

import httpx
import numpy as np
import soundfile as sf

from cloney.core.audio import media_type, to_mono
from cloney.engines.base import EngineError, EngineInfo, EngineOption, VoiceRef

#: Inline-Steuertoken von Higgs v3. Tags außerhalb dieser Menge entfernt die
#: Pipeline vor der Synthese.
HIGGS_TAGS = frozenset(
    {
        # Emotion
        "neutral",
        "happy",
        "sad",
        "angry",
        "excited",
        "calm",
        "fearful",
        "surprised",
        "disgusted",
        # Stil und Prosodie
        "whisper",
        "shout",
        "soft",
        "loud",
        "fast",
        "slow",
        "high",
        "low",
        "emphasis",
        # Geräusche
        "laugh",
        "sigh",
        "breath",
        "cough",
        "gasp",
        "pause",
        "hum",
        "cry",
        "sing",
    }
)

HIGGS_INFO = EngineInfo(
    name="higgs",
    license="Boson Higgs Audio v3 Research & Non-Commercial (Gewichte); Code Apache-2.0",
    vram_gb=11.0,
    languages=("de", "en", "fr", "es", "it", "zh", "ja"),
    sample_rate=24000,
    requires_ref_text=False,
    supported_tags=HIGGS_TAGS,
    # Die Schnittstelle nimmt keinen Seed entgegen; jeder Aufruf würfelt neu.
    reproducible_seed=False,
    description=(
        "Higgs Audio v3 (4B) über lokalen SGLang-Omni-Server. Versteht Inline-Tags. "
        "Braucht in bf16 rund 11 GB VRAM -- auf 8-GB-Karten nicht lauffähig."
    ),
    options=(
        EngineOption(
            key="temperature",
            label="Temperatur",
            minimum=0.1,
            maximum=1.5,
            step=0.05,
            default=0.8,
            help="Niedriger klingt gleichmäßiger, höher lebendiger und unruhiger.",
        ),
        EngineOption(
            key="top_k",
            label="Top-K",
            minimum=1,
            maximum=100,
            step=1,
            default=50,
            integer=True,
            help="Wie viele Kandidaten je Schritt in Frage kommen.",
        ),
    ),
)

_WINDOWS_LAUFWERK = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def server_pfad(pfad: str, modus: str = "auto", *, windows: bool | None = None) -> str:
    """Pfad so schreiben, wie der Server ihn sieht.

    ``auto`` übersetzt unter Windows nach ``/mnt/<laufwerk>/...``, weil ein
    lokaler Higgs-Server dort in aller Regel in WSL läuft und Windows-Pfade
    nicht auflösen kann. ``wsl`` erzwingt die Übersetzung, ``path`` unterlässt
    sie -- etwa wenn Cloney und Server auf demselben Linux-System laufen.
    """
    import platform

    if modus == "path":
        return pfad
    if modus == "auto":
        ist_windows = platform.system() == "Windows" if windows is None else windows
        if not ist_windows:
            return pfad

    treffer = _WINDOWS_LAUFWERK.match(pfad)
    if treffer is None:
        return pfad
    laufwerk, rest = treffer.groups()
    return f"/mnt/{laufwerk.lower()}/{rest.replace(chr(92), '/')}"


class HiggsEngine:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "bosonai/higgs-audio-v3-tts-4b",
        timeout_s: float = 300.0,
        reference_mode: str = "auto",
        voice: str = "default",
        temperature: float = 0.8,
        top_k: int = 50,
        max_new_tokens: int = 1024,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.info = HIGGS_INFO
        self.model = model
        self.reference_mode = reference_mode
        self.voice = voice
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        # trust_env=False: der Server läuft lokal, der HTTP-Proxy der Umgebung
        # darf hier nicht dazwischenfunken.
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            trust_env=False,
            transport=transport,
        )

    def _reference(self, voice: VoiceRef) -> dict[str, str]:
        reference: dict[str, str] = {}
        if self.reference_mode == "base64":
            # Nicht im Kochbuch belegt. Nur nehmen, wenn der Server die Datei
            # nachweislich nicht selbst lesen kann.
            roh = base64.b64encode(voice.audio_path.read_bytes()).decode("ascii")
            reference["audio"] = f"data:{media_type(voice.audio_path)};base64,{roh}"
        else:
            reference["audio_path"] = server_pfad(
                str(voice.audio_path.resolve()), self.reference_mode
            )
        if voice.transcript:
            # Der Referenztext verbessert die Klonqualität deutlich.
            reference["text"] = voice.transcript
        return reference

    def synthesize(self, text: str, voice: VoiceRef, seed: int) -> np.ndarray:
        # seed wird bewusst nicht mitgeschickt: die Schnittstelle kennt kein
        # solches Feld. Siehe reproducible_seed in HIGGS_INFO.
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "references": [self._reference(voice)],
            "response_format": "wav",
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
        }
        try:
            response = self._client.post("/audio/speech", json=payload)
        except httpx.RequestError as exc:
            raise EngineError(
                f"Kein Kontakt zum TTS-Server unter {self._client.base_url}. "
                f"Läuft 'sgl-omni serve' bereits? ({exc})"
            ) from exc

        if response.status_code != 200:
            raise EngineError(
                f"TTS-Server antwortete mit HTTP {response.status_code}: {response.text[:800]}"
            )

        try:
            audio, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        except Exception as exc:  # noqa: BLE001 - der Inhalt ist fremd, die Ursache soll lesbar sein
            kopf = response.text[:200] if response.text else f"{len(response.content)} Bytes"
            raise EngineError(
                f"Antwort des TTS-Servers ist kein lesbarer Ton ({exc}). Anfang: {kopf}"
            ) from exc

        if sample_rate != self.info.sample_rate:
            # Die tatsächliche Rate des Servers gilt, nicht unsere Annahme.
            self.info = replace(self.info, sample_rate=int(sample_rate))
        return to_mono(audio)

    def close(self) -> None:
        self._client.close()
