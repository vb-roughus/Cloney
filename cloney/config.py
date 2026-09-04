"""Zentrale Konfiguration. Alle Werte über CLONEY_*-Umgebungsvariablen oder .env überschreibbar."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field
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
    #: Nach einer Überschrift. Sie trennt den Titel vom Text und ist
    #: deshalb die längste der drei.
    pause_heading_ms: int = 1200
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
    # --- Hugging Face Hub ---------------------------------------------------
    #: Zugangstoken für den Hub. Leer heißt: unangemeldet laden.
    #:
    #: Die Modelle, die Cloney holt, sind öffentlich -- ein Token ist für keines
    #: davon nötig. Es hebt nur die Ratengrenze an, die für unangemeldete
    #: Zugriffe je IP-Adresse gilt, und lässt die Warnung verstummen, die
    #: huggingface_hub sonst bei jedem Lauf ausgibt.
    #:
    #: Angenommen wird beides: CLONEY_HF_TOKEN wie jede andere Einstellung, und
    #: HF_TOKEN unter dem Namen, den die Warnung selbst nennt. Ohne den zweiten
    #: schriebe man das Naheliegende in die .env und wunderte sich, dass nichts
    #: geschieht -- siehe apply_hf_token.
    hf_token: str = Field(default="", validation_alias=AliasChoices("CLONEY_HF_TOKEN", "HF_TOKEN"))

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

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def apply_hf_token(settings: Settings) -> bool:
    """Das Token aus der Konfiguration in die Umgebung heben. True, wenn gesetzt.

    Nötig, weil pydantic-settings die .env in dieses Objekt liest und **nicht**
    in die Umgebung exportiert. Nachgemessen: mit ``HF_TOKEN=...`` in der .env
    steht der Wert im Settings-Objekt, ``os.environ`` bleibt leer, und
    ``huggingface_hub.get_token()`` gibt nichts zurück. Wer das Naheliegende
    tut, hätte also weiterhin die Warnung und keine Wirkung.

    huggingface_hub liest ausschließlich die Umgebung und
    ``~/.cache/huggingface/token``. Ein durchgereichtes Argument genügte
    ohnehin nicht: das Training läuft in einem eigenen Prozess, und F5 lädt dort
    selbst nach. Der Kindprozess erbt die Umgebung -- und damit auch das hier
    Gesetzte.

    Ein bereits gesetztes HF_TOKEN gewinnt: wer sich mit ``huggingface-cli
    login`` angemeldet oder die Variable im Terminal gesetzt hat, meinte das so.
    """
    if os.environ.get("HF_TOKEN"):
        return True
    if not settings.hf_token.strip():
        return False
    os.environ["HF_TOKEN"] = settings.hf_token.strip()
    return True


_settings: Settings | None = None


def get_settings() -> Settings:
    """Prozessweite Settings-Instanz (in Tests via set_settings ersetzbar).

    Hier wird zugleich das Hub-Token wirksam gemacht. Die Stelle ist bewusst
    diese: alles -- Kommandozeile wie Weboberfläche -- kommt hier vorbei, und
    eine Einstellung, die nur im Objekt steht und nirgends wirkt, wäre keine.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        apply_hf_token(_settings)
    return _settings


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings
