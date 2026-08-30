"""Kommandozeile. Dünn -- sie ruft dieselbe Pipeline wie die Web-UI."""

from __future__ import annotations

from pathlib import Path

import typer

from cloney.config import Settings, get_settings
from cloney.core.compare import Comparison
from cloney.core.project import Project
from cloney.core.voices import VoiceStore
from cloney.engines.registry import available_engines, create_engine, engine_info
from cloney.pipeline import ProgressEvent, run_comparison, run_project

app = typer.Typer(help="Lokales Voice Cloning für deutsche Langform-Texte.", no_args_is_help=True)
voices_app = typer.Typer(help="Referenzstimmen verwalten.", no_args_is_help=True)
app.add_typer(voices_app, name="voices")
projects_app = typer.Typer(help="Projekte verwalten.", no_args_is_help=True)
app.add_typer(projects_app, name="projects")


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


def _embedder_factory(settings: Settings, enabled: bool):  # noqa: ANN202
    """Fabrik für den Stimmvergleich, oder None wenn er nicht gewünscht ist."""
    if not enabled or not settings.check_speaker_similarity:
        return None

    def make():  # noqa: ANN202
        from cloney.speaker.ecapa import EcapaEmbedder

        return EcapaEmbedder(settings.speaker_model)

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
    audio: Path = typer.Option(
        ..., exists=True, help="Referenzaufnahme. Wird unverändert übernommen."
    ),
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
    kanaele = {1: "Mono", 2: "Stereo"}.get(check.channels, f"{check.channels} Kanäle")
    typer.echo(
        f"'{name}' angelegt: {check.duration_s:.1f}s, "
        f"Spitze {check.peak_dbfs:.1f} dBFS, Sprachanteil {check.speech_ratio:.0%}"
    )
    typer.echo(
        f"  Unverändert abgelegt: {check.sample_rate} Hz, {kanaele}"
        f"{', ' + check.subtype if check.subtype else ''}"
    )
    if check.chars_per_second:
        from cloney.core.voices import TYPICAL_CHARS_PER_SECOND, suggested_speed

        unten, oben = TYPICAL_CHARS_PER_SECOND
        typer.echo(
            f"  Sprechtempo {check.chars_per_second:.1f} Zeichen/s "
            f"(üblich sind {unten:.0f} bis {oben:.0f})"
        )
        vorschlag = suggested_speed(check.chars_per_second)
        if vorschlag:
            typer.echo(
                "  Engines wie F5-TTS übernehmen dieses Tempo. "
                f"Für ruhigeres Zuhören: -o speed={vorschlag:g}"
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
    from cloney.core.audio import describe_audio

    for voice in found:
        note = voice.transcript or "kein Transkript hinterlegt"
        typer.echo(f"{voice.name:20} {note}")
        try:
            typer.echo(f"{'':20} {describe_audio(voice.audio_path).beschreibung()}")
        except Exception:  # noqa: BLE001 - eine unlesbare Datei darf die Liste nicht kippen
            typer.secho(f"{'':20} Aufnahme nicht lesbar", fg=typer.colors.YELLOW)


@voices_app.command("remove")
def voice_remove(
    name: str = typer.Argument(..., help="Name der Stimme."),
    force: bool = typer.Option(False, "--force", help="Auch löschen, wenn Projekte sie nutzen."),
) -> None:
    """Stimme löschen."""
    settings = get_settings()
    store = VoiceStore(settings.voices_dir)
    if not store.exists(name):
        typer.secho(f"Stimme '{name}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)

    benutzt = [p.name for p in Project.list_all(settings.projects_dir) if p.voice == name]
    if benutzt and not force:
        # Ohne diesen Halt bliebe ein Projekt mit einem Verweis ins Leere zurück.
        typer.secho(
            f"'{name}' wird noch verwendet von: {', '.join(benutzt)}.\n"
            "Diese Projekte zuerst löschen -- oder --force verwenden.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)

    store.delete(name)
    typer.echo(f"'{name}' gelöscht.")


@voices_app.command("transcript")
def voice_transcript(
    name: str = typer.Argument(..., help="Name der Stimme."),
    text: str = typer.Option("", help="Neuer Wortlaut. Ohne Angabe: aktuellen zeigen."),
) -> None:
    """Wortlaut der Referenzaufnahme zeigen oder ändern."""
    settings = get_settings()
    store = VoiceStore(settings.voices_dir)
    if not store.exists(name):
        typer.secho(f"Stimme '{name}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)

    if not text:
        typer.echo(store.get(name).transcript or "(kein Wortlaut hinterlegt)")
        return

    store.set_transcript(name, text)
    pruefung = store.recheck(name)
    if pruefung.chars_per_second:
        typer.echo(f"Gespeichert. {pruefung.chars_per_second:.1f} Zeichen/s (Deutsch: etwa 14)")
    for warnung in pruefung.warnings:
        typer.secho(f"  Achtung: {warnung}", fg=typer.colors.YELLOW)


@projects_app.command("list")
def project_list() -> None:
    """Projekte auflisten."""
    settings = get_settings()
    gefunden = Project.list_all(settings.projects_dir)
    if not gefunden:
        typer.echo("Noch keine Projekte.")
        return
    for project in gefunden:
        fertig, gesamt = project.progress
        typer.echo(f"{project.id}  {fertig}/{gesamt}  {project.engine:7} {project.name}")


@projects_app.command("remove")
def project_remove(
    project_id: str = typer.Argument(..., help="Projekt-Kennung."),
) -> None:
    """Projekt samt erzeugtem Ton löschen."""
    settings = get_settings()
    try:
        root = Project.resolve(settings.projects_dir, project_id)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from None
    if not (root / "project.json").exists():
        typer.secho(f"Projekt '{project_id}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)

    project = Project.load(root)
    project.delete()
    typer.echo(f"'{project.name}' gelöscht.")


@projects_app.command("discard")
def project_discard(
    project_id: str = typer.Argument(..., help="Projekt-Kennung."),
) -> None:
    """Erzeugten Ton verwerfen, Seeds behalten."""
    settings = get_settings()
    root = settings.projects_dir / project_id
    if not (root / "project.json").exists():
        typer.secho(f"Projekt '{project_id}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)

    project = Project.load(root)
    betroffen = project.discard_all_audio()
    typer.echo(f"{betroffen} Sätze zurückgesetzt. Die Seeds bleiben, nur der Ton ist weg.")


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
    model: str = typer.Option("", help="Trainierter Stand. Standard: der Pretrain."),
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
        model=model,
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
    if model:
        typer.echo(f"Modell: {model}")
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
    modell_settings = _modell_einstellungen(settings, project.model)
    try:
        run_project(
            project,
            settings,
            store,
            lambda: create_engine(engine_name, modell_settings, project.engine_options),
            _asr_factory(settings, qc),
            _echo,
            _embedder_factory(settings, qc),
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
    aehnlichkeit = project.median_similarity()
    if aehnlichkeit is not None:
        typer.echo(f"Median-Stimmähnlichkeit: {aehnlichkeit:.2f} (1.00 = identisch)")
    elif project.similarity_note:
        typer.secho(
            f"Ohne Stimmähnlichkeit gerendert: {project.similarity_note}", fg=typer.colors.YELLOW
        )
    flagged = project.flagged()
    if flagged:
        indices = ", ".join(str(c.index + 1) for c in flagged)
        typer.secho(
            f"{len(flagged)} Chunks zur Durchsicht markiert (Nr. {indices}). "
            "In der Web-Oberfläche einzeln nachbessern: cloney web",
            fg=typer.colors.YELLOW,
        )


datasets_app = typer.Typer(help="Trainingsmaterial für ein Finetune.", no_args_is_help=True)
app.add_typer(datasets_app, name="dataset")


@datasets_app.command("build")
def dataset_build(
    audio: list[Path] = typer.Option(
        ..., "--audio", "-a", exists=True, help="Aufnahme oder Ordner. Mehrfach möglich."
    ),
    name: str = typer.Option(..., help="Name des Datensatzes."),
    min_seconds: float = typer.Option(3.0, help="Kürzeste Segmentlänge."),
    max_seconds: float = typer.Option(15.0, help="Längste Segmentlänge."),
    force_split: bool = typer.Option(
        False,
        "--force-split",
        help="Zu lange Bereiche notfalls an der leisesten Stelle trennen, "
        "auch ohne echte Pause. Rettet Material, schneidet aber womöglich im Wort.",
    ),
) -> None:
    """Aus langen Aufnahmen einen Trainingsdatensatz im Format von F5-TTS.

    Geschnitten wird an Pausen, transkribiert mit Whisper, und der Text
    durchläuft dieselbe Normalisierung wie bei der Synthese -- trainiert werden
    muss auf der Form, die später auch hineingeht.
    """
    settings = get_settings()
    settings.ensure_dirs()

    quellen = sorted(_sammle_aufnahmen(audio))
    if not quellen:
        typer.secho("Keine lesbaren Aufnahmen gefunden.", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(f"{len(quellen)} Aufnahme(n) werden zerlegt.")

    from cloney.asr.whisper import WhisperASR
    from cloney.core.dataset import build_dataset

    asr = WhisperASR(settings.asr_model, settings.asr_device, settings.asr_compute_type)
    try:
        dataset = build_dataset(
            name,
            quellen,
            asr,
            settings.datasets_dir,
            language=settings.asr_language,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            force_split=force_split,
            on_event=lambda text: typer.echo(f"  {text}"),
        )
    except ValueError as exc:
        typer.secho(f"\nAbgebrochen: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from None
    finally:
        asr.close()

    _print_dataset(dataset)


@datasets_app.command("probe")
def dataset_probe(
    audio: list[Path] = typer.Option(
        ..., "--audio", "-a", exists=True, help="Aufnahme oder Ordner. Mehrfach möglich."
    ),
) -> None:
    """Nachsehen, statt zu raten: Pegel messen und Schwellen durchspielen.

    Fällt eine Lesung durch, sind zwei Ursachen möglich -- eine Schwelle, die
    nicht zur Aufnahme passt, oder eine Leseweise ohne Pausen. Von außen sehen
    beide gleich aus. Diese Tabelle trennt sie.
    """
    from cloney.core.audio import read_wav
    from cloney.core.dataset import probe_audio

    for quelle in sorted(_sammle_aufnahmen(audio)):
        samples, rate = read_wav(quelle)
        befund = probe_audio(samples, rate)

        typer.echo("")
        typer.secho(f"{quelle.name}", bold=True)
        typer.echo(f"  {befund.duration_s:.1f}s bei {rate} Hz")
        typer.echo(
            f"  Grundpegel   {befund.levels.floor_db:6.0f} dBFS   (leiseste anhaltende Stelle)"
        )
        typer.echo(f"  Sprechpegel  {befund.levels.speech_db:6.0f} dBFS   (95. Perzentil)")
        typer.echo(f"  Exakte Stille {befund.digital_silence_share:6.1%} der Aufnahme")
        typer.echo("")
        typer.echo("  Schwelle   Pausen ab 180ms   ab 320ms   längste Stille   still")
        for zeile in befund.rows:
            marken = []
            if zeile.threshold_db == befund.threshold_db:
                marken.append("verwendet")
            if zeile.silence_share > 0.5:
                marken.append("über dem Sprechpegel")
            marke = "  <- " + ", ".join(marken) if marken else ""
            typer.echo(
                f"  {zeile.threshold_db:7.0f}   {zeile.pauses_split:>13}   "
                f"{zeile.pauses_utterance:>8}   {zeile.longest_pause_s:>11.2f}s   "
                f"{zeile.silence_share:>5.0%}{marke}"
            )
        typer.echo("")
        if befund.hoffnungslos:
            typer.secho(
                "  Keine Schwelle findet Pausen. Es liegt nicht an der Einstellung, "
                "sondern an der Aufnahme:\n"
                "  entweder wird durchgehend gesprochen, oder die Pausen sind mit "
                "Atem oder Raumgeräusch gefüllt.\n"
                "  Beim nächsten Take zwischen den Sätzen bewusst absetzen. Für "
                "vorhandenes Material hilft\n"
                "  'cloney dataset build --force-split' -- das trennt an der leisesten "
                "Stelle, notfalls im Wort.",
                fg=typer.colors.YELLOW,
            )
        elif befund.genug_pausen():
            typer.echo(
                f"  {befund.gefundene_pausen()} Pausen auf {befund.duration_s:.0f}s "
                "-- das reicht für die Zerlegung."
            )
        elif (beste := befund.beste_zeile()) and beste.pauses_split >= befund.benoetigte_pausen():
            typer.secho(
                f"  Die verwendete Schwelle ({befund.threshold_db:.0f} dBFS) findet nur "
                f"{befund.gefundene_pausen()} Pause(n), bei {beste.threshold_db:.0f} dBFS "
                f"wären es {beste.pauses_split}.\n"
                "  Die Pausen liegen also höher als erwartet -- vermutlich wird in ihnen "
                "geatmet. Bitte melden:\n"
                "  daraus gehört eine bessere Regel, keine Handeinstellung.",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(
                f"  Nur {befund.gefundene_pausen()} Pause(n) auf {befund.duration_s:.0f}s. "
                f"Für Segmente von höchstens 15s bräuchte es mindestens "
                f"{befund.benoetigte_pausen()}.\n"
                "  Die Schwelle ist nicht das Problem -- es wird zu lang am Stück "
                "gesprochen. Beim nächsten Take\n"
                "  zwischen den Sätzen bewusst absetzen. Für vorhandenes Material: "
                "'cloney dataset build --force-split'.",
                fg=typer.colors.YELLOW,
            )


@datasets_app.command("list")
def dataset_list() -> None:
    """Angelegte Datensätze auflisten."""
    from cloney.core.dataset import Dataset

    settings = get_settings()
    gefunden = Dataset.list_all(settings.datasets_dir)
    if not gefunden:
        typer.echo("Noch keine Datensätze angelegt.")
        return
    for dataset in gefunden:
        werte = dataset.statistik()
        typer.echo(
            f"{dataset.name:20} {werte['segmente']:>5} Segmente, "
            f"{werte['minuten']:>6.1f} min, {werte['verworfen']} verworfen"
        )


@datasets_app.command("show")
def dataset_show(name: str = typer.Argument(..., help="Name des Datensatzes.")) -> None:
    """Kennzahlen und die Gründe für Verworfenes."""
    from cloney.core.dataset import Dataset

    settings = get_settings()
    root = Dataset.resolve(settings.datasets_dir, name)
    if not (root / "dataset.json").exists():
        typer.secho(f"Datensatz '{name}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)
    _print_dataset(Dataset.load(root), ausfuehrlich=True)


def _sammle_aufnahmen(pfade: list[Path]) -> list[Path]:
    """Dateien und Ordner zu einer Liste von Aufnahmen."""
    endungen = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    gesammelt: list[Path] = []
    for pfad in pfade:
        if pfad.is_dir():
            gesammelt.extend(p for p in sorted(pfad.iterdir()) if p.suffix.lower() in endungen)
        else:
            gesammelt.append(pfad)
    return gesammelt


def _print_dataset(dataset, ausfuehrlich: bool = False) -> None:  # noqa: ANN001
    werte = dataset.statistik()
    typer.echo("")
    typer.echo(f"Datensatz '{dataset.name}' in {dataset.root}")
    typer.echo(
        f"  {werte['segmente']} Segmente, {werte['minuten']:.1f} Minuten, {dataset.sample_rate} Hz"
    )
    typer.echo(
        f"  Median: {werte['median_laenge_s']:.1f}s je Segment, "
        f"{werte['median_zeichen_pro_s']:.1f} Zeichen/s"
    )
    if werte["verworfen"]:
        typer.secho(
            f"  Verworfen: {werte['verworfen']} Abschnitte "
            f"({werte['verworfene_minuten']:.1f} Minuten)",
            fg=typer.colors.YELLOW,
        )
        # Nach Art gruppiert, nicht nach Wortlaut: zwölf Zeilen mit je einer
        # anderen Sekundenzahl sagen weniger als eine Zeile mit der Spannweite.
        gruende: dict[str, list[float]] = {}
        for eintrag in dataset.rejected:
            art = eintrag.reason.split(" -- ")[0]
            gruende.setdefault(art, []).append(eintrag.duration_s)
        for grund, dauern in sorted(gruende.items(), key=lambda p: -len(p[1])):
            spanne = (
                f"{min(dauern):.1f}s"
                if len(dauern) == 1 or min(dauern) == max(dauern)
                else f"{min(dauern):.1f}-{max(dauern):.1f}s"
            )
            typer.echo(f"    {len(dauern):>4}x {grund} ({spanne})")
        if any(r.reason.startswith("am Stück zu lang") for r in dataset.rejected):
            typer.echo(
                "  Zu lange Abschnitte lassen sich notfalls trennen: "
                "cloney dataset build --force-split"
            )
        if any(r.reason.startswith("zu kurz") for r in dataset.rejected):
            typer.echo(
                "  Zu kurze Abschnitte werden mit ihrem Nachbarn zusammengefasst, "
                "solange die Lücke\n  dazwischen unter einer Sekunde bleibt. "
                "Steht daneben ein größerer Abstand, lag es daran."
            )
        if ausfuehrlich:
            typer.echo("")
            for eintrag in dataset.rejected:
                typer.echo(
                    f"    {eintrag.source} bei {eintrag.start_s:7.1f}s "
                    f"({eintrag.duration_s:5.1f}s): {eintrag.reason}"
                )


models_app = typer.Typer(help="Trainierte Modelle verwalten.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("add")
def model_add(
    name: str = typer.Option(..., help="Name des Modells."),
    ckpt: Path = typer.Option(..., exists=True, help="Checkpoint aus dem Training."),
    vocab: Path = typer.Option(
        None, help="Vokabular. Standard: das des Pretrains aus der Konfiguration."
    ),
    note: str = typer.Option("", help="Notiz, etwa Datensatz und Schrittzahl."),
) -> None:
    """Einen trainierten Stand eintragen, damit er auswählbar wird."""
    from cloney.core.models import ModelError, ModelStore

    settings = get_settings()
    settings.ensure_dirs()
    if vocab is None:
        _, vocab = _pretrain_dateien()

    try:
        modell = ModelStore(settings.models_dir).add(name, ckpt, vocab, note)
    except ModelError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from None
    typer.secho(f"'{modell.name}' eingetragen.", fg=typer.colors.GREEN)
    typer.echo(f"  Checkpoint: {modell.ckpt_path}")
    typer.echo(f"  Vokabular:  {modell.vocab_path}")
    typer.echo("")
    typer.echo(f"Rendern damit:  cloney render --text kapitel.txt --voice <stimme> --model {name}")
    typer.echo(f"Vergleichen:    cloney compare --text probe.txt --voice <stimme> --model {name}")


@models_app.command("list")
def model_list() -> None:
    """Eingetragene Modelle."""
    from cloney.core.models import ModelStore

    settings = get_settings()
    gefunden = ModelStore(settings.models_dir).list_all()
    if not gefunden:
        typer.echo("Noch keine Modelle eingetragen.")
        return
    for modell in gefunden:
        zustand = "" if modell.exists else "  (Datei fehlt)"
        typer.echo(f"{modell.name:24} {modell.note or Path(modell.ckpt_path).name}{zustand}")


@models_app.command("remove")
def model_remove(name: str = typer.Argument(..., help="Name des Modells.")) -> None:
    """Eintrag entfernen. Der Checkpoint selbst bleibt liegen."""
    from cloney.core.models import ModelStore

    settings = get_settings()
    store = ModelStore(settings.models_dir)
    if not store.exists(name):
        typer.secho(f"Modell '{name}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)
    store.delete(name)
    typer.echo(f"'{name}' entfernt. Der Checkpoint selbst bleibt liegen.")


def _modell_einstellungen(settings: Settings, name: str) -> Settings:
    """Einstellungen, die auf einen trainierten Stand zeigen."""
    from cloney.core.models import ModelError, ModelStore, settings_for

    if not name:
        return settings
    try:
        return settings_for(ModelStore(settings.models_dir).get(name), settings)
    except ModelError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from None


finetune_app = typer.Typer(
    help="Ein eigenes Modell auf eine Stimme trainieren.", no_args_is_help=True
)
app.add_typer(finetune_app, name="finetune")


def _lade_datensatz(name: str):  # noqa: ANN202
    from cloney.core.dataset import Dataset

    settings = get_settings()
    root = Dataset.resolve(settings.datasets_dir, name)
    if not (root / "dataset.json").exists():
        typer.secho(f"Datensatz '{name}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)
    return Dataset.load(root)


def _pretrain_dateien() -> tuple[Path, Path]:
    """Checkpoint und Vokabular des deutschen Pretrains."""
    from cloney.engines.base import EngineError
    from cloney.engines.f5_german import resolve_model_files

    settings = get_settings()
    if settings.f5_ckpt_path and settings.f5_vocab_path:
        return Path(settings.f5_ckpt_path), Path(settings.f5_vocab_path)
    try:
        ckpt, vocab = resolve_model_files(
            settings.f5_repo_id, settings.f5_ckpt_filename, settings.f5_vocab_filename
        )
    except EngineError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from None
    return Path(ckpt), Path(vocab)


@finetune_app.command("prepare")
def finetune_prepare(
    name: str = typer.Argument(..., help="Name des Datensatzes."),
    f5_dir: Path = typer.Option(
        None, "--f5-dir", help="Wurzel von F5-TTS. Standard: aus dem installierten Paket."
    ),
) -> None:
    """Datensatz in das Format bringen, das F5-TTS zum Training einliest.

    Zwei Dinge passieren dabei, die man leicht übersieht: die Pfadliste bekommt
    eine Kopfzeile und absolute Pfade, und das Vokabular wird durch das des
    deutschen Pretrains ersetzt -- F5 legt sonst sein eigenes hin, das zum
    englisch-chinesischen Basismodell gehört.
    """
    import subprocess

    from cloney.core.finetune import (
        FinetuneError,
        check_prepared,
        data_dir_for,
        install_vocab,
        prepare_command,
        write_f5_metadata,
    )

    dataset = _lade_datensatz(name)
    if not dataset.utterances:
        typer.secho("Der Datensatz ist leer.", fg=typer.colors.RED)
        raise typer.Exit(1)

    _, vocab = _pretrain_dateien()
    try:
        ziel = data_dir_for(dataset.root.name, root=f5_dir)
    except FinetuneError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from None

    tabelle = write_f5_metadata(dataset, dataset.root / "f5")
    typer.echo(f"{len(dataset.utterances)} Segmente, {dataset.total_seconds / 60:.1f} Minuten")
    typer.echo(f"Eingabe:  {tabelle}")
    typer.echo(f"Ausgabe:  {ziel}")
    typer.echo("")

    befehl = prepare_command(tabelle, ziel)
    typer.echo(" ".join(befehl))
    ergebnis = subprocess.run(befehl, check=False)
    if ergebnis.returncode != 0:
        typer.secho("\nVorbereiten fehlgeschlagen.", fg=typer.colors.RED)
        raise typer.Exit(1)

    try:
        install_vocab(vocab, ziel)
        check_prepared(ziel)
    except FinetuneError as exc:
        typer.secho(f"\n{exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from None

    typer.secho(
        f"\nBereit. Vokabular des Pretrains übernommen ({vocab.name}).", fg=typer.colors.GREEN
    )
    typer.echo(f"Weiter mit: cloney finetune train {name}")


@finetune_app.command("train")
def finetune_train(
    name: str = typer.Argument(..., help="Name des Datensatzes."),
    batch_frames: int = typer.Option(
        0, help="batch_size_per_gpu in Frames. 0 = Vorschlag für 16 GB."
    ),
    epochs: int = typer.Option(100, help="Durchläufe über den Datensatz."),
    learning_rate: float = typer.Option(1e-5, help="Lernrate."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Nur den Befehl zeigen."),
    neu: bool = typer.Option(
        False,
        "--neu",
        help="Vom Pretrain aus neu beginnen. Vorhandene Stände werden beiseitegelegt.",
    ),
    f5_dir: Path = typer.Option(
        None, "--f5-dir", help="Wurzel von F5-TTS. Standard: aus dem installierten Paket."
    ),
) -> None:
    """Das Finetune starten. Braucht eine GPU und läuft Stunden.

    Ein zweiter Lauf mit demselben Datensatznamen setzt fort, wo der erste
    aufgehört hat -- auch mit erweitertem Material. Wer stattdessen vom Pretrain
    aus beginnen will, nimmt --neu.
    """
    import subprocess

    from cloney.core.finetune import (
        BATCH_FRAMES_16GB,
        FinetuneError,
        check_prepared,
        plan_training,
        staende_beiseite,
        vorhandene_staende,
        write_trainer_pretrain,
    )

    dataset = _lade_datensatz(name)
    ckpt, vocab = _pretrain_dateien()
    try:
        # F5s Trainer lädt den Pretrain zuerst in den EMA-Wrapper. Ein reiner
        # Inferenz-Export -- wie ihn der deutsche Finetune mitbringt -- scheitert
        # dort. Umgeschrieben wird nur, wenn nötig.
        ckpt = write_trainer_pretrain(ckpt, dataset.root / "f5" / f"pretrain_{ckpt.stem}.pt")
    except FinetuneError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from None

    try:
        plan = plan_training(
            dataset,
            ckpt,
            vocab,
            batch_frames=batch_frames or BATCH_FRAMES_16GB,
            epochs=epochs,
            learning_rate=learning_rate,
            f5_dir=f5_dir,
        )
        if not dry_run:
            check_prepared(plan.data_dir)
    except FinetuneError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from None

    typer.echo(f"Datensatz:   {plan.dataset_name}, {plan.total_seconds / 60:.1f} Minuten")
    typer.echo(f"Daten:       {plan.data_dir}")
    typer.echo(f"Checkpoints: {plan.checkpoint_dir}")
    typer.echo(f"Pretrain:    {plan.pretrain_ckpt.name}")

    # F5 entscheidet allein nach den Dateien im Checkpoint-Ordner, woher es
    # lädt -- der Pretrain kommt gar nicht zum Zug, wenn dort schon etwas liegt.
    # Das gehört vor den Lauf, nicht in die Nachbetrachtung.
    staende = vorhandene_staende(plan.checkpoint_dir)
    if staende and neu:
        if not dry_run:
            beiseite = staende_beiseite(plan.checkpoint_dir)
            typer.secho(
                f"{len(staende)} vorhandene Stände liegen jetzt in {beiseite.name}.",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(
                f"--neu würde {len(staende)} vorhandene Stände beiseitelegen.",
                fg=typer.colors.YELLOW,
            )
    elif staende:
        typer.secho(
            f"Fortsetzung: im Ordner liegen {len(staende)} Stände ({staende[-1].name}). "
            "F5 lädt den letzten davon,\n"
            "der Pretrain bleibt außen vor -- samt Optimierer, Schrittzähler und "
            "Stelle im Lernratenverlauf.\n"
            "Eine geänderte --learning-rate wirkt dabei nicht. "
            "Vom Pretrain aus beginnen: --neu.",
            fg=typer.colors.YELLOW,
        )
    typer.echo("")
    typer.echo(
        f"{plan.batch_frames} Frames je Schritt sind {plan.seconds_per_step:.1f}s Ton; "
        f"eine Epoche braucht rund {plan.steps_per_epoch} Schritte,"
    )
    typer.echo(f"{plan.epochs} Epochen also etwa {plan.total_steps} Schritte.")
    typer.echo(
        f"Davon {plan.warmup} zum Aufwärmen; gesichert wird alle {plan.save_interval} Schritte."
    )
    typer.secho(
        "Der Vorschlag für den Speicher ist ein Ausgangspunkt, kein Befund -- "
        "bei einem Speicherfehler --batch-frames halbieren.",
        fg=typer.colors.YELLOW,
    )
    if plan.knappes_material:
        typer.secho(
            f"Nur {plan.total_seconds / 60:.1f} Minuten Material. F5s eigene Angabe für "
            "diesen Fall lautet 10 bis 100 Stunden,\n"
            "die dokumentierten Erfolge einzelner Sprecher liegen bei zwölf Stunden und "
            "darüber. Für weniger gibt es\n"
            "keinen belegten Fall. Der Lauf kostet wenig -- die Erwartung sollte "
            "entsprechend sein.",
            fg=typer.colors.YELLOW,
        )
    typer.echo("")
    typer.echo(" ".join(plan.command()))
    if dry_run:
        return
    typer.echo("")
    ergebnis = subprocess.run(plan.command(), check=False)
    if ergebnis.returncode != 0:
        typer.secho("\nTraining abgebrochen.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Ein Checkpoint, der nirgends eingetragen ist, lässt sich nicht anhören.
    # Deshalb wird der jüngste Stand gleich verwendbar gemacht.
    from cloney.core.models import ModelError, ModelStore, find_checkpoints

    typer.secho(f"\nFertig. Checkpoints in {plan.checkpoint_dir}", fg=typer.colors.GREEN)
    staende = find_checkpoints(plan.checkpoint_dir)
    if not staende:
        typer.secho("Kein Checkpoint gefunden -- nichts einzutragen.", fg=typer.colors.YELLOW)
        return

    settings = get_settings()
    modellname = f"{plan.dataset_name}-ft"
    try:
        ModelStore(settings.models_dir).add(
            modellname,
            staende[0],
            plan.vocab_path,
            note=f"{plan.dataset_name}, {plan.total_seconds / 60:.0f} min, {staende[0].name}",
        )
    except ModelError as exc:
        typer.secho(str(exc), fg=typer.colors.YELLOW)
        return

    typer.echo(f"Als '{modellname}' eingetragen ({staende[0].name}).")
    if len(staende) > 1:
        typer.echo(
            f"Weitere Zwischenstände: {', '.join(p.name for p in staende[1:5])} "
            "-- mit 'cloney models add' eintragen, um sie gegeneinander zu hören."
        )
    typer.echo("")
    typer.echo("Gegen den Pretrain messen:")
    typer.echo(
        f'  cloney compare --text probe.txt --voice <stimme> -m "" -m {modellname} -g speed=1.0'
    )


@app.command()
def compare(
    text: Path = typer.Option(..., exists=True, help="Kurze Textprobe."),
    voice: str = typer.Option(..., help="Name der Referenzstimme."),
    name: str = typer.Option("", help="Name des Vergleichs. Standard: Dateiname."),
    engine: str = typer.Option("", help="Engine. Standard: aus der Konfiguration."),
    grid: list[str] = typer.Option(
        None,
        "--grid",
        "-g",
        help="Achse des Rasters, etwa -g speed=0.8,1.0,1.2. Mehrfach möglich.",
    ),
    model: list[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Trainierter Stand. Mehrfach möglich; leer heißt Pretrain. "
        "Mit mehreren wird auch über die Stände verglichen.",
    ),
    qc: bool = typer.Option(True, help="Qualitätskontrolle per Spracherkennung."),
) -> None:
    """Dieselbe Textprobe je Reglerstellung einmal rendern und gegenüberstellen.

    Macht aus dem Raten an den Reglern eine Messung: gleiche Probe, gleiche
    Seeds, nur die Einstellung ändert sich.
    """
    settings = get_settings()
    settings.ensure_dirs()
    engine_name = engine or settings.engine
    info = engine_info(engine_name)

    store = VoiceStore(settings.voices_dir)
    if not store.exists(voice):
        typer.secho(f"Stimme '{voice}' gibt es nicht.", fg=typer.colors.RED)
        raise typer.Exit(1)
    if info.requires_ref_text and not store.get(voice).transcript.strip():
        typer.secho(
            f"Die Engine '{engine_name}' braucht den Wortlaut der Referenzaufnahme, "
            f"'{voice}' hat aber keinen.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    try:
        comparison = Comparison.create(
            name=name or text.stem,
            text=text.read_text(encoding="utf-8"),
            voice=voice,
            engine=info,
            grid=_parse_grid(grid, info),
            comparisons_dir=settings.comparisons_dir,
            models=_parse_models(model, settings),
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from None

    if not info.reproducible_seed:
        typer.secho(
            f"Hinweis: '{engine_name}' nimmt keinen Seed entgegen. Die Varianten "
            "unterscheiden sich zusätzlich im Zufall, nicht allein im Regler.",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"Vergleich {comparison.id} mit {len(comparison.variants)} Varianten:")
    for variant in comparison.variants:
        typer.echo(f"  {variant.label}")
    typer.echo("")

    run_comparison(
        comparison,
        settings,
        store,
        lambda options, modell: create_engine(
            engine_name, _modell_einstellungen(settings, modell), options
        ),
        _asr_factory(settings, qc),
        _echo,
        _embedder_factory(settings, qc),
    )
    _print_comparison(comparison)


def _parse_models(namen: list[str] | None, settings: Settings) -> list[str]:
    """Modellnamen prüfen, bevor der Lauf beginnt.

    Ein leerer Eintrag steht für den Pretrain -- so lässt sich 'Pretrain gegen
    Finetune' als '-m "" -m anna-ft' schreiben. Wird gar keiner genannt, bleibt
    es beim Pretrain, und das Modell ist keine Achse des Vergleichs.
    """
    from cloney.core.models import ModelStore

    if not namen:
        return []
    store = ModelStore(settings.models_dir)
    geprueft = []
    for name in namen:
        if name and not store.exists(name):
            vorhanden = ", ".join(m.name for m in store.list_all()) or "keine"
            typer.secho(
                f"Modell '{name}' gibt es nicht. Eingetragen: {vorhanden}", fg=typer.colors.RED
            )
            raise typer.Exit(1)
        geprueft.append(name)
    return geprueft


def _parse_grid(rohwerte: list[str] | None, info) -> dict[str, list[float]]:  # noqa: ANN001
    """'-g speed=0.8,1.0' zu Achsen. Unbekannte Regler brechen ab, statt zu wirken,
    als hätten sie etwas bewirkt."""
    achsen: dict[str, list[float]] = {}
    for eintrag in rohwerte or []:
        key, _, rest = eintrag.partition("=")
        key = key.strip()
        if info.option(key) is None:
            erlaubt = ", ".join(o.key for o in info.options) or "keine"
            typer.secho(
                f"'{key}' ist kein Regler von '{info.name}'. Verfügbar: {erlaubt}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        werte = []
        for stueck in rest.replace(",", " ").split():
            try:
                werte.append(float(stueck))
            except ValueError:
                typer.secho(
                    f"'{stueck}' ist keine Zahl (bei --grid {eintrag})", fg=typer.colors.RED
                )
                raise typer.Exit(1) from None
        achsen.setdefault(key, []).extend(werte)
    return achsen


def _print_comparison(comparison: Comparison) -> None:
    """Die Tabelle. Markiert wird je Spalte, nicht als Gesamtnote -- die
    Gewichtung von Fehlerrate gegen Ähnlichkeit kann niemand belegen."""
    bestes_cer = comparison.best_cer()
    beste_stimme = comparison.best_similarity()

    typer.echo("")
    typer.echo(f"{'Variante':28} {'CER':>8} {'Stimme':>8} {'Dauer':>8} {'Tempo':>12}")
    typer.echo("-" * 68)
    for variant in comparison.variants:
        if variant.error:
            typer.secho(f"{variant.label[:28]:28} {variant.error[:38]}", fg=typer.colors.RED)
            continue
        tempo = comparison.chars_per_second(variant.slug)
        zeile = (
            f"{variant.label[:28]:28} "
            f"{_zelle(variant.median_cer, '{:.1%}'):>8}"
            f"{'*' if variant.slug in bestes_cer else ' '}"
            f"{_zelle(variant.median_similarity, '{:.2f}'):>7}"
            f"{'*' if variant.slug in beste_stimme else ' '}"
            f"{_zelle(variant.duration_s, '{:.1f}s'):>8} "
            f"{_zelle(tempo, '{:.1f} Zeichen/s'):>12}"
        )
        typer.echo(zeile)
    typer.echo("")
    typer.echo(
        "* bester Wert der Spalte. Die Zahlen engen die Auswahl ein, entschieden wird am Ohr."
    )


def _zelle(wert: float | None, format_: str) -> str:
    return format_.format(wert) if wert is not None else "--"


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
    from cloney.web.app import create_app
    from cloney.web.launch import open_browser_when_ready, serve

    settings = get_settings()
    url = f"http://{host}:{port}"
    typer.secho(f"Cloney läuft auf {url}", fg=typer.colors.GREEN)
    typer.secho("Beenden mit Strg+C", fg=typer.colors.BRIGHT_BLACK)
    if open_browser:
        open_browser_when_ready(url)
    anwendung = create_app(settings, _asr_factory(settings, qc), _embedder_factory(settings, qc))
    try:
        serve(anwendung, host, port)
    except KeyboardInterrupt:
        typer.echo("")


if __name__ == "__main__":
    app()
