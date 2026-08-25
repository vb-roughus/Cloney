"""Spracherkennung über faster-whisper.

Zwei Aufgaben: den Referenzclip transkribieren (manche Engines brauchen den Text,
und für die Eingangsprüfung ist er ohnehin nützlich) und in der
Qualitätskontrolle jeden erzeugten Chunk zurückschreiben.
"""

from __future__ import annotations

import numpy as np

#: faster-whisper erwartet 16 kHz Mono.
WHISPER_SAMPLE_RATE = 16000


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Einfache lineare Umtastung. Für ASR-Eingaben ausreichend, nicht für Ausgaben."""
    if source_rate == target_rate or audio.size == 0:
        return audio
    duration = audio.size / source_rate
    target_length = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, audio.size, endpoint=False)
    target_x = np.linspace(0.0, duration, target_length, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


class WhisperASR:
    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper ist nicht installiert. Installation: uv pip install -e '.[asr]'"
            ) from exc
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "de") -> str:
        samples = resample_linear(audio, sample_rate, WHISPER_SAMPLE_RATE)
        segments, _ = self._model.transcribe(
            samples,
            language=language,
            beam_size=5,
            # Ohne diese Abschaltung reicht Whisper den vorherigen Text als
            # Kontext weiter und halluziniert bei kurzen Chunks Fortsetzungen.
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def close(self) -> None:
        self._model = None
