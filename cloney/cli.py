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


def _parse_options(rohwerte: list[str] | None, info) -> dict[str, float]:  # noqa: ANN001
    """'-o speed=0.85' zu Zahlen. Unbekannte Namen brechen ab, statt zu wirken,
    als hätten sie etwas bewirkt."""
    if not rohwerte:
        return {}
    gesammelt: dict[str, float] = {}
    for eintrag in rohwerte:
        name, _, wert = eintrag.partition("=")
        name = name.strip()
        if info.option(name) is None:
            erlaubt = ", ".join(o.key for o in info.options) or "keine"
            typer.secho(
                f"'{name}' ist kein Regler von '{info.name}'. Verfügbar: {erlaubt}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        try:
            gesammelt[name] = float(wert)
        except ValueError:
            typer.secho(f"'{wert}' ist keine Zahl (bei --option {eintrag})", fg=typer.colors.RED)
            raise typer.Exit(1) from None
    return info.clean_options(gesammelt)


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
    option: list[str] = typer.Option(
        None, "--option", "-o", help="Regler der Engine, etwa -o speed=0.85. Mehrfach möglich."
    ),
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

    options = _parse_options(option, info)

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
    if options:
        project.engine_options = options
        project.save()
        gesetzt = ", ".join(f"{k}={v:g}" for k, v in sorted(options.items()))
        typer.echo(f"Regler: {gesetzt}")
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
            lambda: create_engine(engine_name, settings, project.engine_options),
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


#: Kurzer Text für den Selbsttest. Bewusst voller Ziffern, Symbole und
#: Abkürzungen -- man soll hören, dass die Normalisierung greift.
DEMO_TEXT = (
    "Am 3. Mai 2024 um 14:30 Uhr kostete es 1.250,50 €. "
    "Dr. Meier sagte z.B., das seien ca. 50 % zu viel."
)

#: Wortlaut des Beispielclips, den F5-TTS mitbringt.
_F5_EXAMPLE_TRANSCRIPT = "Some call me nature, others call me mother nature."


@app.command()
def doctor() -> None:
    """Umgebung prüfen und zu jedem Befund den Befehl nennen, der ihn behebt."""
    from cloney.doctor import run_checks

    settings = get_settings()
    report = run_checks(settings)

    colors = {
        "ok": typer.colors.GREEN,
        "warn": typer.colors.YELLOW,
        "fail": typer.colors.RED,
    }
    labels = {"ok": " OK ", "warn": "WARN", "fail": "FEHL"}
    for check in report.results:
        typer.secho(f"[{labels[check.status]}] ", fg=colors[check.status], nl=False)
        typer.echo(f"{check.name:18} {check.detail}")
        if check.remedy:
            typer.secho(f"{'':25} -> {check.remedy}", fg=typer.colors.BRIGHT_BLACK)

    typer.echo("")
    if report.failures:
        typer.secho(
            f"{len(report.failures)} Punkte müssen behoben werden, "
            f"{len(report.warnings)} Hinweise.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    if report.warnings:
        typer.secho(
            f"Einsatzbereit, mit {len(report.warnings)} Hinweisen. "
            "Nicht jeder davon muss dich betreffen.",
            fg=typer.colors.YELLOW,
        )
        return
    typer.secho("Alles bereit.", fg=typer.colors.GREEN)


def _demo_voice(store: VoiceStore, audio: Path | None, voice: str, settings: Settings) -> str:
    """Referenzstimme für den Selbsttest bestimmen."""
    if audio is not None:
        store.add(
            "demo",
            audio,
            transcript="",
            min_seconds=settings.ref_min_seconds,
            max_seconds=settings.ref_max_seconds,
        )
        return "demo"
    if voice:
        return voice
    existing = store.list_all()
    if existing:
        return existing[0].name

    # Letzte Möglichkeit: der Beispielclip aus F5-TTS. Er ist englisch, taugt
    # also nicht zur Beurteilung der Stimme -- wohl aber zum Nachweis, dass die
    # Kette insgesamt läuft.
    example: Path | None = None
    try:
        from importlib.resources import files

        candidate = Path(str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")))
        example = candidate if candidate.is_file() else None
    except (ImportError, ModuleNotFoundError):
        example = None
    if example is None:
        typer.secho(
            "Keine Referenzstimme vorhanden. Eine Aufnahme mitgeben:\n"
            "  cloney demo --audio meine_stimme.wav",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho(
        "Keine eigene Stimme hinterlegt -- der englische Beispielclip von F5-TTS springt ein.\n"
        "Er belegt, dass die Kette läuft, sagt über die Klangqualität aber nichts aus.",
        fg=typer.colors.YELLOW,
    )
    store.add("demo", example, transcript=_F5_EXAMPLE_TRANSCRIPT, min_seconds=0.0, max_seconds=99.0)
    return "demo"


@app.command()
def demo(
    audio: Path | None = typer.Option(
        None, exists=True, help="Referenzaufnahme für den Selbsttest."
    ),
    voice: str = typer.Option("", help="Bereits angelegte Stimme verwenden."),
    engine: str = typer.Option("", help="Engine. Standard: aus der Konfiguration."),
    qc: bool = typer.Option(False, help="Qualitätskontrolle per Spracherkennung."),
) -> None:
    """Kurzen Text rendern, um die gesamte Kette einmal zu belegen."""
    settings = get_settings()
    settings.ensure_dirs()
    engine_name = engine or settings.engine
    info = engine_info(engine_name)

    store = VoiceStore(settings.voices_dir)
    name = _demo_voice(store, audio, voice, settings)
    reference = store.get(name)

    if info.requires_ref_text and not reference.transcript.strip():
        typer.echo("Die Engine braucht den Wortlaut der Referenz -- wird ermittelt ...")
        try:
            from cloney.asr.whisper import WhisperASR
            from cloney.core.audio import read_wav

            samples, rate = read_wav(reference.audio_path)
            asr = WhisperASR(settings.asr_model, settings.asr_device, settings.asr_compute_type)
            transcript = asr.transcribe(samples, rate, settings.asr_language)
            asr.close()
        except RuntimeError as exc:
            typer.secho(
                f"{exc}\nAlternativ den Wortlaut von Hand hinterlegen:\n"
                f'  cloney voices add --audio <datei> --name {name} --transcript "..."',
                fg=typer.colors.RED,
            )
            raise typer.Exit(1) from None
        store.set_transcript(name, transcript)
        reference = store.get(name)
        typer.echo(f"Erkannt: {transcript}")

    typer.echo(f"Engine {engine_name}, Stimme '{name}' ({reference.duration_s:.0f}s Referenz)")
    typer.echo(f"Text: {DEMO_TEXT}")
    typer.echo("")

    project = Project.create(
        name="Selbsttest",
        text=DEMO_TEXT,
        voice=name,
        engine=info,
        projects_dir=settings.projects_dir,
        reference_seconds=reference.duration_s,
        chars_per_second=settings.chars_per_second,
        target_seconds=settings.target_chunk_seconds,
        max_seconds=settings.max_chunk_seconds,
    )
    typer.echo(f"Gesprochen wird: {project.chunks[0].normalized_text}")
    typer.echo("")
    _run(project, settings, engine_name, qc)
    typer.echo("")
    typer.secho(f"Fertig. Anhören: {project.output_path.resolve()}", fg=typer.colors.GREEN)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Adresse."),
    port: int = typer.Option(8080, help="Port."),
    qc: bool = typer.Option(True, help="Qualitätskontrolle per Spracherkennung."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Browser öffnen, sobald der Server antwortet."
    ),
) -> None:
    """Web-Oberfläche starten."""
    import uvicorn

    from cloney.web.app import create_app
    from cloney.web.launch import open_browser_when_ready

    settings = get_settings()
    url = f"http://{host}:{port}"
    typer.secho(f"Cloney läuft auf {url}", fg=typer.colors.GREEN)
    typer.secho("Beenden mit Strg+C", fg=typer.colors.BRIGHT_BLACK)
    if open_browser:
        open_browser_when_ready(url)
    try:
        uvicorn.run(create_app(settings, _asr_factory(settings, qc)), host=host, port=port)
    except KeyboardInterrupt:
        typer.echo("")


if __name__ == "__main__":
    app()
