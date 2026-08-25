from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cloney.config import Settings
from cloney.core.audio import write_wav
from cloney.core.voices import VoiceStore
from cloney.engines.dummy import reset_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", engine="dummy")


@pytest.fixture
def reference_wav(tmp_path: Path) -> Path:
    """Acht Sekunden sprachähnliches Signal als Referenzaufnahme."""
    sample_rate = 24000
    t = np.arange(8 * sample_rate, dtype=np.float32) / sample_rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    audio = (0.3 * envelope * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    path = tmp_path / "referenz.wav"
    write_wav(path, audio, sample_rate)
    return path


@pytest.fixture
def voice_store(settings: Settings, reference_wav: Path) -> VoiceStore:
    store = VoiceStore(settings.voices_dir)
    store.add("test-stimme", reference_wav, transcript="Dies ist die Referenzaufnahme.")
    return store
