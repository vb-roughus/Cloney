"""Higgs Audio v3 über einen lokalen SGLang-Omni-Server.

Das Modell läuft als eigener Prozess (``sgl-omni serve --model-path
bosonai/higgs-audio-v3-tts-4b --port 8000``) und spricht die OpenAI-kompatible
``/v1/audio/speech``-Schnittstelle. Für Cloney ist das der bequemste Weg: die
Modellverwaltung liegt vollständig beim Server, und der VRAM wird freigegeben,
indem man diesen Prozess beendet.

Hinweis zum Schema: die ``references``-Struktur ist gegen die öffentliche
Dokumentation gebaut, konnte hier aber nicht gegen einen laufenden Server
verifiziert werden. Antwortet der Server mit einem Feldfehler, gibt
``EngineError`` den Text der Serverantwort unverändert weiter -- damit ist
erkennbar, welches Feld anzupassen ist.
"""

from __future__ import annotations

import base64
import io
from dataclasses import replace

import httpx
import numpy as np
import soundfile as sf

from cloney.core.audio import to_mono
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


class HiggsEngine:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "higgs-audio-v3-tts",
        timeout_s: float = 300.0,
        reference_mode: str = "path",
        temperature: float = 0.8,
        top_k: int = 50,
        max_new_tokens: int = 1024,
    ) -> None:
        self.info = HIGGS_INFO
        self.model = model
        self.reference_mode = reference_mode
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        # trust_env=False: der Server läuft lokal, der HTTP-Proxy der Umgebung
        # darf hier nicht dazwischenfunken.
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s, trust_env=False
        )

    def _reference(self, voice: VoiceRef) -> dict[str, str]:
        reference: dict[str, str] = {}
        if self.reference_mode == "base64":
            data = voice.audio_path.read_bytes()
            reference["audio"] = base64.b64encode(data).decode("ascii")
        else:
            reference["audio_path"] = str(voice.audio_path.resolve())
        if voice.transcript:
            # Der Referenztext verbessert die Klonqualität deutlich.
            reference["text"] = voice.transcript
        return reference

    def synthesize(self, text: str, voice: VoiceRef, seed: int) -> np.ndarray:
        payload = {
            "model": self.model,
            "input": text,
            "references": [self._reference(voice)],
            "response_format": "wav",
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "seed": seed,
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

        audio, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        if sample_rate != self.info.sample_rate:
            # Die tatsächliche Rate des Servers gilt, nicht unsere Annahme.
            self.info = replace(self.info, sample_rate=int(sample_rate))
        return to_mono(audio)

    def close(self) -> None:
        self._client.close()
