"""F5-TTS mit einem deutschen Finetune, in-process.

Anders als Higgs läuft dieses Modell im eigenen Prozess. Es ist damit die erste
Engine, bei der ``vram.py`` tatsächlich Speicher freigeben muss statt nur eine
HTTP-Verbindung zu schließen.

Warum ein deutschsprachiger Finetune: Higgs v3 ist ein generalistisches
Multilingual-Modell, dessen deutsche Prosodie sein schwächster Teil ist. Ein auf
Deutsch nachtrainiertes Modell trifft Betonung und Satzmelodie in aller Regel
besser -- und mit rund 2 GB passt es auch auf Karten, auf denen Higgs nicht läuft.

Zwei Eigenheiten des Modells prägen die Anbindung:

1. **Der Referenztext ist Pflicht.** F5-TTS leitet aus ihm die Sprechgeschwindigkeit
   ab. Fehlt er, wird das Ergebnis unbrauchbar -- deshalb bricht die Engine
   lieber mit einer klaren Meldung ab.
2. **Eine Generierung umfasst höchstens rund 22 Sekunden**, Referenz eingerechnet;
   längere Eingaben teilt das Modell selbst auf und blendet die Teile ineinander.
   ``EngineInfo.chunk_budget_seconds`` schneidet die Chunks deshalb vorher so zu,
   dass es nie dazu kommt.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from cloney.core.audio import to_mono
from cloney.engines.base import EngineError, EngineInfo, VoiceRef

#: F5-TTS zielt auf dieses Gesamtbudget je Generierung (siehe infer_process:
#: max_chars leitet sich aus 22 minus Referenzlänge ab).
MAX_GENERATION_SECONDS = 22.0
#: Auf diese Länge kürzt F5-TTS die Referenzaufnahme selbst zurecht.
MAX_REFERENCE_SECONDS = 12.0

F5_INFO = EngineInfo(
    name="f5-de",
    license="CC-BY-NC-4.0 (Finetune erbt von SWivid/F5-TTS); Code MIT",
    vram_gb=2.0,
    languages=("de",),
    sample_rate=24000,
    requires_ref_text=True,
    supported_tags=frozenset(),
    description=(
        "F5-TTS mit deutschem Finetune, läuft im eigenen Prozess. Rund 2 GB VRAM, "
        "damit auch auf 8-GB-Karten nutzbar. Kennt keine Inline-Tags."
    ),
    max_generation_seconds=MAX_GENERATION_SECONDS,
    max_reference_seconds=MAX_REFERENCE_SECONDS,
)


def _resolve_device(device: str) -> str | None:
    if device != "auto":
        return device
    return None  # F5-TTS wählt dann selbst cuda / mps / cpu


def choose_model_files(files: list[str], prefer_bigvgan: bool = False) -> tuple[str, str]:
    """Sucht Checkpoint und Vokabular in der Dateiliste eines Modell-Repos.

    Die deutschen Finetunes legen ihre Dateien unterschiedlich ab -- mal flach,
    mal in Unterordnern, mal als .pt statt .safetensors, und die Schrittzahl im
    Namen ist beliebig. Statt einen Namen zu raten wird hier gewählt.
    """
    vocabs = [f for f in files if pathlib.PurePosixPath(f).name == "vocab.txt"]
    if not vocabs:
        raise EngineError("Im Modell-Repo gibt es keine vocab.txt")
    # Flach liegende Vokabulare gelten für das ganze Repo.
    vocab = min(vocabs, key=lambda f: (f.count("/"), len(f)))

    candidates = [f for f in files if f.endswith((".safetensors", ".pt"))]
    if not candidates:
        raise EngineError("Im Modell-Repo gibt es keinen Checkpoint (.safetensors oder .pt)")

    def rank(name: str) -> tuple:
        lower = name.lower()
        digits = re.findall(r"\d+", pathlib.PurePosixPath(name).stem)
        return (
            # bigvgan braucht zusätzliche Abhängigkeiten -- nur auf Wunsch.
            ("bigvgan" in lower) != prefer_bigvgan,
            not name.endswith(".safetensors"),
            # Bei mehreren Ständen den höchsten nehmen; "last" schlägt alles.
            0 if "last" in lower else 1,
            -max((int(d) for d in digits), default=0),
            len(name),
        )

    return min(candidates, key=rank), vocab


def discover_model_files(repo_id: str, prefer_bigvgan: bool = False) -> tuple[str, str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise EngineError('huggingface_hub fehlt. Installation: pip install -e ".[f5]"') from exc
    try:
        files = list(list_repo_files(repo_id))
    except Exception as exc:
        raise EngineError(f"Dateiliste von '{repo_id}' nicht abrufbar: {exc}") from exc
    return choose_model_files(files, prefer_bigvgan)


def _download(repo_id: str, filename: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise EngineError('huggingface_hub fehlt. Installation: pip install -e ".[f5]"') from exc

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as exc:
        raise EngineError(
            f"'{filename}' konnte nicht aus '{repo_id}' geladen werden: {exc}\n"
            "Mit CLONEY_F5_CKPT_FILENAME und CLONEY_F5_VOCAB_FILENAME lässt sich die "
            "Auswahl überschreiben, mit CLONEY_F5_CKPT_PATH und CLONEY_F5_VOCAB_PATH "
            "direkt auf lokale Dateien zeigen."
        ) from exc


class F5GermanEngine:
    """Anbindung an ``f5_tts.api.F5TTS``.

    Das Modell wird im Konstruktor geladen, weil ``vram.py`` die Engine über eine
    Fabrik erzeugt und direkt danach wieder freigibt -- Erzeugen und Laden fallen
    hier also bewusst zusammen.
    """

    def __init__(
        self,
        model_config: str = "F5TTS_Base",
        repo_id: str = "aihpi/F5-TTS-German",
        ckpt_filename: str = "",
        vocab_filename: str = "",
        ckpt_path: str = "",
        vocab_path: str = "",
        device: str = "auto",
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        speed: float = 1.0,
        cross_fade_seconds: float = 0.15,
    ) -> None:
        self.info = F5_INFO
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.speed = speed
        self.cross_fade_seconds = cross_fade_seconds

        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise EngineError(
                'f5-tts ist nicht installiert. Installation: pip install -e ".[f5]"'
            ) from exc

        if not (ckpt_path and vocab_path) and not (ckpt_filename and vocab_filename):
            # Nichts vorgegeben: im Repo nachsehen, statt Namen zu raten.
            ckpt_filename, vocab_filename = discover_model_files(repo_id)
        checkpoint = ckpt_path or _download(repo_id, ckpt_filename)
        vocabulary = vocab_path or _download(repo_id, vocab_filename)
        for label, path in (("Checkpoint", checkpoint), ("Vokabular", vocabulary)):
            if not Path(path).exists():
                raise EngineError(f"{label} nicht gefunden: {path}")

        self._model = F5TTS(
            model=model_config,
            ckpt_file=str(checkpoint),
            vocab_file=str(vocabulary),
            device=_resolve_device(device),
        )
        rate = getattr(self._model, "target_sample_rate", None)
        if rate:
            self.info = replace(F5_INFO, sample_rate=int(rate))

    def synthesize(self, text: str, voice: VoiceRef, seed: int) -> np.ndarray:
        if not voice.transcript.strip():
            raise EngineError(
                f"Die Stimme '{voice.name}' hat keinen Referenztext. F5-TTS leitet daraus "
                "die Sprechgeschwindigkeit ab und braucht ihn zwingend. Nachtragen mit: "
                f"cloney voices add --audio <datei> --name {voice.name} --auto-transcript"
            )

        # F5-TTS schreibt Fortschritt und Textbatches unaufgefordert nach stdout.
        # Das gehört nicht in unsere Ausgabe -- im Fehlerfall ist es aber die
        # nützlichste Spur, deshalb wird es aufgehoben statt verworfen.
        chatter = io.StringIO()
        try:
            with contextlib.redirect_stdout(chatter):
                wave, _sample_rate, _spectrogram = self._model.infer(
                    ref_file=str(voice.audio_path),
                    ref_text=voice.transcript,
                    gen_text=text,
                    seed=seed,
                    nfe_step=self.nfe_step,
                    cfg_strength=self.cfg_strength,
                    speed=self.speed,
                    cross_fade_duration=self.cross_fade_seconds,
                    show_info=lambda *_args, **_kwargs: None,
                )
        except Exception as exc:
            noise = chatter.getvalue().strip()
            detail = f"\nAusgabe des Modells:\n{noise[-800:]}" if noise else ""
            raise EngineError(f"F5-TTS konnte nichts erzeugen: {exc}{detail}") from exc

        if wave is None:
            raise EngineError("F5-TTS lieferte kein Audio zurück (leerer Textbatch?)")
        return to_mono(np.asarray(wave, dtype=np.float32))

    def close(self) -> None:
        self._model = None
