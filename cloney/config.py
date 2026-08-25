"""Zentrale Konfiguration. Alle Werte über CLONEY_*-Umgebungsvariablen oder .env überschreibbar."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLONEY_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("./data")
    engine: str = "dummy"

    # --- Segmentierung ---------------------------------------------------
    # Deutsche Sprechgeschwindigkeit, konservativ geschätzt. Dient nur der
    # Chunk-Größenplanung, nicht der Ausgabe.
    chars_per_second: float = 14.0
    target_chunk_seconds: float = 20.0
    max_chunk_seconds: float = 25.0

    # --- Qualitätskontrolle ----------------------------------------------
    cer_threshold: float = 0.10
    max_retries: int = 2

    # --- Assembly ---------------------------------------------------------
    target_lufs: float = -16.0
    pause_sentence_ms: int = 350
    pause_paragraph_ms: int = 800
    edge_fade_ms: int = 12
    trim_threshold_db: float = -45.0

    # --- Referenz-Audio-Gate ---------------------------------------------
    ref_min_seconds: float = 5.0
    ref_max_seconds: float = 20.0

    # --- Higgs Audio v3 über sgl-omni ------------------------------------
    higgs_base_url: str = "http://localhost:8000/v1"
    higgs_model: str = "higgs-audio-v3-tts"
    higgs_timeout_s: float = 300.0

    # --- ASR (faster-whisper) --------------------------------------------
    asr_model: str = "large-v3-turbo"
    asr_device: str = "auto"
    asr_compute_type: str = "int8"
    asr_language: str = "de"

    # --- Text-LLM (Ollama / llama.cpp, OpenAI-kompatibel) ----------------
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_timeout_s: float = 120.0

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    def ensure_dirs(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Prozessweite Settings-Instanz (in Tests via set_settings ersetzbar)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings
