"""Umgebungsdiagnose.

Die Fehler, die beim ersten Aufsetzen auftreten, sehen alle gleich aus: irgendwo
bricht es ab. Die Ursachen liegen aber weit auseinander -- ein PyTorch ohne
Unterstützung für die eigene Karte, ein fehlender Modell-Download, ein nicht
gestarteter Server. Dieses Modul prüft jede dieser Stellen einzeln und nennt zu
jedem Befund den Befehl, der ihn behebt.

Geprüft wird durch Ausführen, nicht durch Nachschlagen von Versionsnummern:
ob ``torchaudio`` eine WAV-Datei öffnen kann, entscheidet ein echter Ladeversuch.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from cloney.config import Settings

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    remedy: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str, remedy: str = "") -> None:
        self.results.append(CheckResult(name, status, detail, remedy))

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "warn"]

    @property
    def ok(self) -> bool:
        return not self.failures


# --------------------------------------------------------------------------
# Einzelprüfungen
# --------------------------------------------------------------------------


def check_python(report: Report) -> None:
    # Kein Versionsvergleich: unter Python 3.11 scheitert bereits der Import von
    # cloney, diese Prüfung käme also gar nicht erst zum Zug.
    version = ".".join(str(v) for v in sys.version_info[:3])
    report.add("Python", "ok", f"{version} auf {platform.system()} {platform.machine()}")


def check_torch(report: Report) -> None:
    """PyTorch muss die eigene Karte auch wirklich bedienen können.

    Ein PyTorch ohne passenden Rechenkern startet anstandslos und rechnet dann
    still auf der CPU oder bricht mitten im Lauf ab -- deshalb wird hier nicht
    die Version verglichen, sondern die Architekturliste des Builds gegen die
    Architektur der eingebauten Karte.
    """
    try:
        import torch
    except ImportError:
        report.add(
            "PyTorch",
            "warn",
            "nicht installiert",
            "Nur nötig für die Engine f5-de und die Qualitätskontrolle. "
            "Installation siehe README (cu128-Index für RTX-50-Karten).",
        )
        return

    if not torch.cuda.is_available():
        report.add(
            "PyTorch",
            "warn",
            f"{torch.__version__}, aber keine CUDA-Karte sichtbar -- alles läuft auf der CPU",
            "Bei vorhandener NVIDIA-Karte: PyTorch mit CUDA installieren, "
            "--index-url https://download.pytorch.org/whl/cu128",
        )
        return

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = list(torch.cuda.get_arch_list())
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if arch not in supported:
        report.add(
            "PyTorch",
            "fail",
            f"{torch.__version__} kennt {arch} nicht ({name}). Unterstützt: {', '.join(supported)}",
            "Dieses PyTorch bringt keinen Rechenkern für deine Karte mit. "
            "Neu installieren mit: pip install --force-reinstall torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128",
        )
        return

    report.add(
        "PyTorch",
        "ok",
        f"{torch.__version__} (CUDA {torch.version.cuda}) auf {name}, "
        f"{arch}, {total_gb:.0f} GB VRAM",
    )


#: winget-Paket mit den FFmpeg-Bibliotheken. Nicht "Gyan.FFmpeg" -- das ist der
#: statische Build, der nur ffmpeg.exe mitbringt und torchcodec nichts nützt.
FFMPEG_WINGET = "winget install --id Gyan.FFmpeg.Shared"

#: Dateien, an denen ein Shared-Build erkennbar ist.
_FFMPEG_LIBS = ("avcodec-*.dll", "avformat-*.dll", "avutil-*.dll")


def find_ffmpeg_shared_libraries() -> list[str]:
    """Sucht die FFmpeg-DLLs im Suchpfad. Leer = kein Shared-Build vorhanden."""
    found: list[str] = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for pattern in _FFMPEG_LIBS:
            found.extend(glob.glob(os.path.join(directory, pattern)))
    return found


def summarise_decoder_error(text: str) -> tuple[str, str]:
    """Verdichtet die Fehlerlawine von torchcodec auf Ursache und Abhilfe.

    torchcodec probiert sechs FFmpeg-Versionen durch und legt jeden Fehlschlag
    einzeln offen -- rund hundert Zeilen. Ungekürzt in einen Diagnosebericht
    gekippt macht das den Bericht unlesbar und verdeckt die anderen Befunde.
    """
    first = text.strip().splitlines()[0].strip() if text.strip() else "unbekannter Fehler"
    # Der Anreißer endet oft mit "Likely causes:" -- die Ursachen stehen danach
    # ohnehin ausführlich da und werden hier durch die Abhilfe ersetzt.
    first = re.split(r"\s*Likely causes:?", first)[0]
    first = re.sub(r"\s+", " ", first).strip()[:200]

    if "could not find module" in text.lower() or "cannot open shared object" in text.lower():
        cause = (
            "Die FFmpeg-Bibliotheken fehlen. Der statische FFmpeg-Build bringt nur "
            "ffmpeg.exe mit; torchcodec braucht die DLLs des Shared-Builds. "
            f"Installieren mit: {FFMPEG_WINGET} -- danach die Konsole neu öffnen, "
            "damit der Suchpfad übernommen wird."
        )
    elif "undefined symbol" in text.lower() or "not compatible" in text.lower():
        cause = (
            "torchcodec passt nicht zur installierten PyTorch-Version. "
            "Passende Fassung wählen: pip install --upgrade torchcodec "
            "(Kompatibilitätstabelle: github.com/pytorch/torchcodec)."
        )
    else:
        cause = (
            f"Ursache unklar. Häufigster Fall sind fehlende FFmpeg-Bibliotheken: {FFMPEG_WINGET}"
        )
    return first, cause


def check_audio_loading(report: Report) -> None:
    """Kann torchaudio eine WAV-Datei öffnen?

    Neuere torchaudio-Versionen laden über torchcodec, das FFmpeg-Bibliotheken
    voraussetzt. Fehlen sie, scheitert F5-TTS erst beim Einlesen der Referenz --
    also spät und mit unverständlicher Meldung. Ein echter Ladeversuch klärt das
    vorab.
    """
    try:
        import torchaudio
    except ImportError:
        report.add(
            "Audio-Laden",
            "warn",
            "torchaudio nicht installiert (nur für f5-de nötig)",
            "Kommt mit PyTorch: pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128",
        )
        return

    from cloney.core.audio import write_wav

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.wav"
        tone = 0.2 * np.sin(2 * np.pi * 220 * np.arange(24000, dtype=np.float32) / 24000)
        write_wav(path, tone.astype(np.float32), 24000)
        try:
            waveform, rate = torchaudio.load(str(path))
        except Exception as exc:
            summary, remedy = summarise_decoder_error(str(exc))
            report.add(
                "Audio-Laden", "fail", f"torchaudio öffnet keine WAV-Datei: {summary}", remedy
            )
            return

    detail = f"torchaudio liest WAV ({rate} Hz, {waveform.shape[-1]} Samples)"
    report.add("Audio-Laden", "ok", detail)


def check_ffmpeg(report: Report) -> None:
    """Nicht die ausführbare Datei zählt, sondern die Bibliotheken.

    ffmpeg.exe im Suchpfad sagt nichts darüber, ob torchcodec arbeiten kann:
    der statische Build bringt genau diese Datei mit und sonst nichts. Wer nur
    darauf prüft, meldet 'in Ordnung', während das Laden jeder Audiodatei
    scheitert.
    """
    executable = shutil.which("ffmpeg")
    libraries = find_ffmpeg_shared_libraries()

    if libraries:
        where = Path(libraries[0]).parent
        report.add("FFmpeg", "ok", f"Bibliotheken gefunden in {where}")
    elif executable and platform.system() == "Windows":
        report.add(
            "FFmpeg",
            "warn",
            "nur ffmpeg.exe gefunden, keine Bibliotheken -- das ist der statische Build",
            f"Für torchcodec wird der Shared-Build gebraucht: {FFMPEG_WINGET} "
            "-- danach die Konsole neu öffnen.",
        )
    elif executable:
        report.add("FFmpeg", "ok", f"ffmpeg gefunden unter {executable}")
    else:
        report.add(
            "FFmpeg",
            "warn",
            "nicht im Suchpfad",
            "Für WAV-Referenzen nicht nötig. Scheitert das Audio-Laden oben, ist dies "
            f"die Ursache: {FFMPEG_WINGET}",
        )


def check_asr(report: Report, settings: Settings) -> None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        report.add(
            "Spracherkennung",
            "warn",
            "faster-whisper nicht installiert -- ohne sie entfällt die Qualitätsmessung",
            'pip install -e ".[asr]"',
        )
        return
    report.add("Spracherkennung", "ok", f"faster-whisper vorhanden, Modell {settings.asr_model}")


def check_similarity(report: Report, settings: Settings) -> None:
    """Die zweite Kennzahl neben der Fehlerrate.

    Fehlt das Paket, wird der Vergleich übersprungen und die Spur trotzdem
    fertig -- deshalb ein Hinweis und keine Fehlermeldung.
    """
    if not settings.check_speaker_similarity:
        report.add("Stimmähnlichkeit", "ok", "abgeschaltet (CLONEY_CHECK_SPEAKER_SIMILARITY)")
        return
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        report.add(
            "Stimmähnlichkeit",
            "warn",
            "speechbrain nicht installiert -- gerendert wird trotzdem, nur ohne die "
            "Ähnlichkeit zur Referenzstimme",
            f'"{sys.executable}" -m pip install -e ".[similarity]"',
        )
        return
    schwelle = settings.similarity_threshold
    hinweis = (
        f"Schwelle {schwelle:.2f}"
        if schwelle > 0
        else "ohne Schwelle -- es wird gemessen und angezeigt, aber nichts markiert"
    )
    report.add("Stimmähnlichkeit", "ok", f"speechbrain vorhanden, {hinweis}")


def check_f5(report: Report, settings: Settings) -> None:
    try:
        import f5_tts  # noqa: F401
    except ImportError:
        report.add(
            "Engine f5-de",
            "warn",
            "f5-tts nicht installiert",
            'pip install -e ".[f5]"',
        )
        return

    if settings.f5_ckpt_path and settings.f5_vocab_path:
        configured = (settings.f5_ckpt_path, settings.f5_vocab_path)
        missing = [p for p in configured if not Path(p).exists()]
        if missing:
            report.add("Engine f5-de", "fail", f"Datei fehlt: {missing[0]}", "Pfad in .env prüfen")
        else:
            report.add("Engine f5-de", "ok", "lokale Modelldateien vorhanden")
        return

    from cloney.engines.base import EngineError
    from cloney.engines.f5_german import discover_model_files

    try:
        ckpt, vocab = discover_model_files(settings.f5_repo_id)
    except EngineError as exc:
        report.add(
            "Engine f5-de",
            "fail",
            str(exc).splitlines()[0],
            f"Repo '{settings.f5_repo_id}' erreichbar? Sonst CLONEY_F5_REPO_ID anpassen.",
        )
        return
    report.add("Engine f5-de", "ok", f"{settings.f5_repo_id}: {ckpt} + {vocab}")


def served_models(base_url: str, timeout: float = 3.0) -> list[str] | None:
    """Welche Modelle der Server anbietet. None heißt: nicht erreichbar."""
    import httpx

    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout, trust_env=False)
        if response.status_code >= 500:
            return None
        daten = response.json().get("data") or []
    except Exception:  # noqa: BLE001 - jede Störung heißt hier schlicht "nicht erreichbar"
        return None
    return [str(eintrag.get("id", "")) for eintrag in daten if eintrag.get("id")]


def check_higgs(report: Report, settings: Settings) -> None:
    """Server erreichbar, und heißt das Modell dort so wie hier?

    Der zweite Teil ist der Grund für diese Prüfung. Ein OpenAI-kompatibler
    Server lehnt eine Anfrage mit unbekanntem Modellnamen ab, und die Meldung
    dazu taucht sonst erst mitten in einem Renderlauf auf.
    """
    modelle = served_models(settings.higgs_base_url)
    if modelle is None:
        report.add(
            "Engine higgs",
            "warn",
            f"kein Server unter {settings.higgs_base_url}",
            "In WSL starten mit: "
            "sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000",
        )
        return

    if modelle and settings.higgs_model not in modelle:
        report.add(
            "Engine higgs",
            "fail",
            f"Server kennt '{settings.higgs_model}' nicht. Angeboten: {', '.join(modelle)}",
            f"CLONEY_HIGGS_MODEL={modelle[0]}",
        )
        return

    report.add("Engine higgs", "ok", f"Server erreichbar, Modell '{settings.higgs_model}'")

    # Wie die Referenzaufnahme zum Server kommt, ist der zweite Stolperstein
    # nach dem Modellnamen -- und der einzige, der einen Startparameter braucht.
    if settings.higgs_reference_mode == "base64":
        report.add("Higgs-Referenz", "ok", "geht als Data-URL mit, kein Serverparameter nötig")
        return

    stimmen = settings.voices_dir.resolve()
    if platform.system() == "Windows":
        from cloney.engines.higgs import server_pfad

        stimmen_fuer_server = server_pfad(str(stimmen), settings.higgs_reference_mode)
    else:
        stimmen_fuer_server = str(stimmen)
    report.add(
        "Higgs-Referenz",
        "warn",
        f"als Dateipfad ({settings.higgs_reference_mode}) -- der Server muss ihn lesen dürfen",
        f"sgl-omni serve ... --allowed-local-media-path {stimmen_fuer_server} "
        "(oder CLONEY_HIGGS_REFERENCE_MODE=base64)",
    )


def check_data_dir(report: Report, settings: Settings) -> None:
    try:
        settings.ensure_dirs()
        probe = settings.data_dir / ".schreibprobe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        report.add("Datenverzeichnis", "fail", f"{settings.data_dir} nicht beschreibbar: {exc}")
        return
    voices = len(list(settings.voices_dir.iterdir())) if settings.voices_dir.exists() else 0
    projects = len(list(settings.projects_dir.iterdir())) if settings.projects_dir.exists() else 0
    report.add(
        "Datenverzeichnis",
        "ok",
        f"{settings.data_dir.resolve()} ({voices} Stimmen, {projects} Projekte)",
    )


def check_pipeline(report: Report) -> None:
    """Durchstich mit der Dummy-Engine -- prüft die Verkettung ohne jedes Modell."""
    from cloney.asr.dummy import DummyASR
    from cloney.core.audio import duration_seconds, read_wav, write_wav
    from cloney.core.project import Project
    from cloney.core.voices import VoiceStore
    from cloney.engines.dummy import DummyEngine
    from cloney.pipeline import run_project

    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = VoiceStore(root / "voices")
            tone = 0.3 * np.sin(2 * np.pi * 150 * np.arange(8 * 24000, dtype=np.float32) / 24000)
            reference = root / "ref.wav"
            write_wav(reference, tone.astype(np.float32), 24000)
            store.add("probe", reference, transcript="Probeaufnahme.")

            project = Project.create(
                name="Selbsttest",
                text="Am 3. Mai 2024 kostete es 1.250,50 Euro.",
                voice="probe",
                engine=DummyEngine.info,
                projects_dir=root / "projects",
            )
            run_project(project, Settings(data_dir=root), store, DummyEngine, DummyASR)
            audio, rate = read_wav(project.output_path)
    except Exception as exc:
        report.add("Durchstich", "fail", f"Selbsttest fehlgeschlagen: {exc}")
        return

    report.add(
        "Durchstich",
        "ok" if project.is_complete else "fail",
        f"{len(project.chunks)} Chunks, {duration_seconds(audio, rate):.1f}s erzeugt, "
        f"Fehlerrate {project.median_cer():.0%}",
    )


def run_checks(settings: Settings) -> Report:
    report = Report()
    check_python(report)
    check_torch(report)
    check_audio_loading(report)
    check_ffmpeg(report)
    check_asr(report, settings)
    check_similarity(report, settings)
    check_f5(report, settings)
    check_higgs(report, settings)
    check_data_dir(report, settings)
    check_pipeline(report)
    return report
