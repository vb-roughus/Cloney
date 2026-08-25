"""Modellfreie Engine für Tests, CI und Entwicklung ohne GPU.

Sie erzeugt aus (Text, Seed) deterministisch ein sprachähnliches Signal. Damit
lassen sich Segmentierung, Manifest, Resume, Metriken, Zusammenbau und sämtliche
Web-Routen vollständig prüfen, ohne ein einziges Modell zu laden.

Damit ``DummyASR`` den zugrunde liegenden Text zurückliefern kann -- und damit
auch die Retry-Schleife testbar wird -- trägt jede Ausgabe eine kurze Kennung in
den ersten Samples. Ein Hash über die Samples genügt dafür nicht: die Umwandlung
nach PCM16 verschiebt die Werte minimal, sodass der Hash den Weg über die
WAV-Datei nicht übersteht. Die Kennung ist deshalb in Amplitudenstufen kodiert,
die um Größenordnungen gröber sind als der Quantisierungsfehler.
"""

from __future__ import annotations

import hashlib

import numpy as np

from cloney.engines.base import EngineInfo, VoiceRef

#: Kennung -> (Text, Seed). Prozessweit, damit ASR und Engine sich nicht kennen müssen.
_REGISTRY: dict[str, tuple[str, int]] = {}

_CHARS_PER_SECOND = 14.0
_LEAD_SILENCE_S = 0.12
_KEY_NIBBLES = 40  # SHA1 in Hex
_SAMPLES_PER_NIBBLE = 8
_KEY_STEP = 0.0125  # ~400x über dem PCM16-Rundungsfehler von ~3e-5
_KEY_LENGTH = _KEY_NIBBLES * _SAMPLES_PER_NIBBLE


def _encode_key(key: str) -> np.ndarray:
    nibbles = np.array([int(c, 16) for c in key], dtype=np.float32)
    levels = (nibbles - 7.5) * _KEY_STEP
    return np.repeat(levels, _SAMPLES_PER_NIBBLE).astype(np.float32)


def _decode_key(audio: np.ndarray) -> str | None:
    if audio.size < _KEY_LENGTH:
        return None
    block = audio[:_KEY_LENGTH].reshape(_KEY_NIBBLES, _SAMPLES_PER_NIBBLE).mean(axis=1)
    nibbles = np.rint(block / _KEY_STEP + 7.5).astype(int)
    if np.any(nibbles < 0) or np.any(nibbles > 15):
        return None
    return "".join(f"{n:x}" for n in nibbles)


def lookup(audio: np.ndarray) -> tuple[str, int] | None:
    key = _decode_key(audio)
    return _REGISTRY.get(key) if key else None


def reset_registry() -> None:
    _REGISTRY.clear()


class DummyEngine:
    info = EngineInfo(
        name="dummy",
        license="n/a (kein Modell)",
        vram_gb=0.0,
        languages=("de", "en"),
        sample_rate=24000,
        requires_ref_text=False,
        supported_tags=frozenset({"freundlich", "traurig", "schnell", "langsam"}),
        description="Synthetisches Testsignal ohne Modell. Für CI und Entwicklung ohne GPU.",
    )

    def synthesize(self, text: str, voice: VoiceRef, seed: int) -> np.ndarray:
        sample_rate = self.info.sample_rate
        seconds = max(0.5, len(text) / _CHARS_PER_SECOND)
        n = int(seconds * sample_rate)

        key = hashlib.sha1(f"{text}|{seed}|{voice.name}".encode()).hexdigest()
        rng = np.random.default_rng(int(key[:16], 16))

        t = np.arange(n, dtype=np.float32) / sample_rate
        base = 110.0 + rng.uniform(0, 60)
        signal = np.zeros(n, dtype=np.float32)
        for harmonic, weight in enumerate((1.0, 0.5, 0.25), start=1):
            signal += weight * np.sin(2 * np.pi * base * harmonic * t)

        # Silbenartige Amplitudenmodulation, damit Trimmen und Lautheitsmessung
        # auf einem Signal mit echter Hüllkurve arbeiten.
        syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
        signal *= (0.25 * syllables).astype(np.float32)

        pad = np.zeros(int(_LEAD_SILENCE_S * sample_rate), dtype=np.float32)
        audio = np.concatenate([_encode_key(key), pad, signal, pad]).astype(np.float32)

        _REGISTRY[key] = (text, seed)
        return audio

    def close(self) -> None:
        return None
