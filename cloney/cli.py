"""Kommandozeile. Dünn -- sie ruft dieselbe Pipeline wie die Web-UI."""

from __future__ import annotations

from pathlib import Path

import typer

from cloney.config import Settings, get_settings
from cloney.core.project import Project
from cloney.core.voices import VoiceStore
from cloney.engines.registry import available_engines, create_engine, engine_info
from cloney.pipeline import ProgressEvent, run_project

app = typer.Typer(help="Lokales Voice Cloning für deutsche Langform-Texte.", no_args_is_help=True)
voices_app = typer.Typer(help="Referenzstimmen verwalten.", no_args_is_help=True)
app.add_typer(voices_app, name="voices")


def _asr_factory(settings: Settings, enabled: bool):  # noqa: ANN202
    if not enabled:
        return None

    def make():  # noqa: ANN202
        from cloney.asr.whisper import WhisperASR

        return WhisperASR(settings.asr_model, settings.asr_device, settings.asr_compute_type)

    return make


def _echo(event: ProgressEvent) -> None:
    progress = f" [{event.done}/{event.total}]" if event.total else ""
    typer.echo(f"  {event.phase}{progress}: {event.message}")


@app.command("engines")
def list_engines() -> None:
    """Verfügbare Engines mit VRAM-Bedarf und Lizenz der Gewichte."""
    for info in available_engines():
        typer.echo(f"{info.name:8} {info.vram_gb:>5.1f} GB  {info.license}")
        typer.echo(f"{'':8} {'':>5}     {info.description}")


@voices_app.command("add")
def voice_add(
    audio: Path = typer.Option(..., exists=True, help="Referenzaufnahme (WAV)."),
    name: str = typer.Option(..., help="Name der Stimme."),
    transcript: str = typer.Option("", help="Wortlaut der Aufnahme."),
    auto_transcript: bool = typer.Option(
        False, "--auto-transcript", help="Wortlaut per Spracherkennung ermitteln."
    ),
) -> None:
    """Stimme anlegen und die Referenzaufnahme prüfen."""
    settings = get_settings()
    settings.ensure_dirs()
    store = VoiceStore(settings.voices_dir)

    if auto_transcript and not transcript:
        from cloney.asr.whisper import WhisperASR
        from cloney.core.audio import read_wav

        samples, rate = read_wav(audio)
        asr = WhisperASR(settings.asr_model, settings.asr_device, settings.asr_compute_type)
        transcript = asr.transcribe(samples, rate, settings.asr_language)
        asr.close()
        typer.echo(f"Erkannt: {transcript}")

    _, check = store.add(
        name, audio, transcript, settings.ref_min_seconds, settings.ref_max_seconds
    )
    typer.echo(
        f"'{name}' angelegt: {check.duration_s:.1f}s bei {check.sample_rate} Hz, "
        f"Spitze {check.peak_dbfs:.1f} dBFS, Sprachanteil {check.speech_ratio:.0%}"
    )
    for warning in check.warnings:
        typer.secho(f"  Achtung: {warning}", fg=typer.colors.YELLOW)
    if not check.ok:
        raise typer.Exit(1)


@voices_app.command("list")
def voice_list() -> None:
    """Angelegte Stimmen auflisten."""
    settings = get_settings()
    found = VoiceStore(settings.voices_dir).list_all()
    if not found:
        typer.echo("Noch keine Stimmen angelegt.")
        return
    for voice in found:
        note = voice.transcript or "kein Transkript hinterlegt"
        typer.echo(f"{voice.name:20} {note}")


@app.command()
def render(
    text: Path = typer.Option(
        ..., exists=True, help="Textdatei, Absätze durch Leerzeile getrennt."
    ),
    voice: str = typer.Option(..., help="Name der Referenzstimme."),
    name: str = typer.Option("", help="Projektname. Standard: Dateiname."),
    engine: str = typer.Option("", help="Engine. Standard: aus der Konfiguration."),
    qc: bool = typer.Option(True, help="Qualitätskontrolle per Spracherkennung."),
) -> None:
    """Text zu einer fertigen Spur rendern."""
    settings = get_settings()
    settings.ensure_dirs()
    engine_name = engine or settings.engine
    info = engine_info(engine_name)

    store = VoiceStore(settings.voices_dir)
    if not store.exists(voice):
        typer.secho(f"Stimme '{voice}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)
    reference = store.get(voice)
    if info.requires_ref_text and not reference.transcript.strip():
        # Lieber jetzt abbrechen als nach der halben Synthese.
        typer.secho(
            f"Die Engine '{engine_name}' braucht den Wortlaut der Referenzaufnahme, "
            f"'{voice}' hat aber keinen. Nachtragen mit:\n"
            f"  cloney voices add --audio <datei> --name {voice} --auto-transcript",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    project = Project.create(
        name=name or text.stem,
        text=text.read_text(encoding="utf-8"),
        voice=voice,
        engine=info,
        projects_dir=settings.projects_dir,
        reference_seconds=reference.duration_s,
        chars_per_second=settings.chars_per_second,
        target_seconds=settings.target_chunk_seconds,
        max_seconds=settings.max_chunk_seconds,
    )
    typer.echo(
        f"Projekt {project.id} mit {len(project.chunks)} Chunks angelegt "
        f"(bis {project.target_chunk_seconds:.0f}s je Chunk)."
    )
    _run(project, settings, engine_name, qc)


@app.command()
def resume(
    project_id: str = typer.Argument(..., help="Projekt-Kennung."),
    qc: bool = typer.Option(True, help="Qualitätskontrolle per Spracherkennung."),
) -> None:
    """Einen unterbrochenen Lauf fortsetzen. Fertige Chunks bleiben unberührt."""
    settings = get_settings()
    root = settings.projects_dir / project_id
    if not (root / "project.json").exists():
        typer.secho(f"Projekt '{project_id}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)

    project = Project.load(root)
    offen = len(project.pending_synthesis())
    typer.echo(f"{offen} von {len(project.chunks)} Chunks noch offen.")
    _run(project, settings, project.engine, qc)


def _run(project: Project, settings: Settings, engine_name: str, qc: bool) -> None:
    store = VoiceStore(settings.voices_dir)
    try:
        run_project(
            project,
            settings,
            store,
            lambda: create_engine(engine_name, settings),
            _asr_factory(settings, qc),
            _echo,
        )
    except (RuntimeError, ValueError) as exc:
        # Fehlendes Modell, nicht erreichbarer Server, unbekannte Engine: alles
        # vorhersehbare, behebbare Zustände. Ein Traceback hilft dabei niemandem.
        typer.secho(f"\nAbgebrochen: {exc}", fg=typer.colors.RED)
        typer.echo(
            f"Bereits erzeugte Chunks bleiben erhalten. Fortsetzen mit: cloney resume {project.id}"
        )
        raise typer.Exit(1) from None

    median = project.median_cer()
    typer.echo("")
    typer.echo(f"Fertig: {project.output_path}")
    if median is not None:
        typer.echo(f"Median-Fehlerrate: {median:.1%}")
    flagged = project.flagged()
    if flagged:
        indices = ", ".join(str(c.index + 1) for c in flagged)
        typer.secho(
            f"{len(flagged)} Chunks zur Durchsicht markiert (Nr. {indices}). "
            "In der Web-Oberfläche einzeln nachbessern: cloney web",
            fg=typer.colors.YELLOW,
        )


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Adresse."),
    port: int = typer.Option(8080, help="Port."),
    qc: bool = typer.Option(True, help="Qualitätskontrolle per Spracherkennung."),
) -> None:
    """Web-Oberfläche starten."""
    import uvicorn

    from cloney.web.app import create_app

    settings = get_settings()
    typer.echo(f"Cloney läuft auf http://{host}:{port}")
    uvicorn.run(create_app(settings, _asr_factory(settings, qc)), host=host, port=port)


if __name__ == "__main__":
    app()
