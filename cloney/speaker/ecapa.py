"""Stimmeinbettung über ECAPA-TDNN aus SpeechBrain.

Das Modell ist mit rund 20 MB klein genug, um nach der Spracherkennung in einem
eigenen Slot zu laufen, ohne das Phasenmodell zu verletzen.

Es ist auf 16 kHz trainiert; abweichende Raten werden vorher umgetastet.
"""

from __future__ import annotations

import numpy as np

#: ECAPA-TDNN ist auf einkanaligem Ton mit dieser Rate trainiert.
ECAPA_SAMPLE_RATE = 16000

DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class EcapaEmbedder:
    def __init__(self, source: str = DEFAULT_SOURCE, savedir: str = "", device: str = "") -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise RuntimeError(
                'speechbrain ist nicht installiert. Installation: pip install -e ".[similarity]"'
            ) from exc

        optionen: dict[str, object] = {}
        if savedir:
            optionen["savedir"] = savedir
        if device:
            optionen["run_opts"] = {"device": device}
        self._model = EncoderClassifier.from_hparams(source=source, **optionen)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        from cloney.asr.whisper import resample_linear

        samples = resample_linear(audio, sample_rate, ECAPA_SAMPLE_RATE)
        with torch.no_grad():
            vektor = self._model.encode_batch(torch.from_numpy(samples).unsqueeze(0))
        return vektor.squeeze().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self._model = None
