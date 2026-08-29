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
    # F5-TTS lässt gelegentlich ein Stück der Referenz am Anfang stehen. Wird es
    # erkannt und ist es länger als dieser Wert, schneidet Cloney es weg.
    trim_reference_bleed: bool = True
    min_bleed_seconds: float = 0.15

    # --- Stimmähnlichkeit --------------------------------------------------
    # Die Fehlerrate prüft die Wörter, nicht die Stimme. Dieser Vergleich
    # schließt die Lücke.
    check_speaker_similarity: bool = True
    # Bewusst 0: gemessen und angezeigt wird immer, markiert wird erst, wenn
    # hier ein Wert steht. Welche Ähnlichkeit ein guter Klon erreicht, hängt am
    # Modell und an der Aufnahme -- eine ungeprüfte Schwelle würde Fehlalarme
    # erzeugen und vom Wesentlichen ablenken. Nach ein paar Läufen den
    # niedrigsten Wert guter Sätze ablesen und knapp darunter setzen.
    similarity_threshold: float = 0.0
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"

    # --- Assembly ---------------------------------------------------------
    target_lufs: float = -16.0
    pause_sentence_ms: int = 350
    pause_paragraph_ms: int = 800
    edge_fade_ms: int = 12
    trim_threshold_db: float = -45.0

    # --- Referenz-Audio-Gate ---------------------------------------------
    ref_min_seconds: float = 5.0
    ref_max_seconds: float = 12.0

    # --- Higgs Audio v3 über sgl-omni ------------------------------------
    higgs_base_url: str = "http://localhost:8000/v1"
    # Muss dem --model-path des Servers entsprechen; sonst lehnt er die Anfrage ab.
    higgs_model: str = "bosonai/higgs-audio-v3-tts-4b"
    higgs_timeout_s: float = 300.0
    # Steht in jedem Beispiel der Schnittstelle, auch beim Klonen.
    higgs_voice: str = "default"
    # Wie der Server an die Referenzaufnahme kommt.
    # "base64": als Data-URL in der Anfrage. Voreinstellung, weil sie ohne
    #   Pfadübersetzung und ohne --allowed-local-media-path am Server auskommt.
    # "auto": Dateipfad, unter Windows nach /mnt/<laufwerk>/... übersetzt (WSL).
    # "wsl": Übersetzung erzwingen. "path": Pfad unverändert weiterreichen.
    # Die Pfadwege setzen voraus, dass der Server mit
    #   --allowed-local-media-path <ordner der stimmen> gestartet wurde.
    higgs_reference_mode: str = "base64"
    higgs_temperature: float = 0.8
    higgs_top_k: int = 50
    higgs_max_new_tokens: int = 1024

    # --- F5-TTS mit deutschem Finetune -----------------------------------
    # Die Dateinamen unterscheiden sich zwischen den Finetunes. Passen sie nicht,
    # nennt die Fehlermeldung der Engine die zu setzenden Variablen.
    f5_repo_id: str = "aihpi/F5-TTS-German"
    f5_model_config: str = "F5TTS_Base"
    # Leer = im Repo nachsehen und selbst wählen.
    f5_ckpt_filename: str = ""
    f5_vocab_filename: str = ""
    # Lokale Dateien haben Vorrang vor dem Download.
    f5_ckpt_path: str = ""
    f5_vocab_path: str = ""
    f5_device: str = "auto"
    # nfe_step steuert Qualität gegen Rechenzeit: 32 ist der Standard, 16 ist
    # spürbar schneller und für Korrekturläufe meist ausreichend.
    f5_nfe_step: int = 32
    f5_cfg_strength: float = 2.0
    f5_speed: float = 1.0

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

    @property
    def comparisons_dir(self) -> Path:
        return self.data_dir / "comparisons"

    def ensure_dirs(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)


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
