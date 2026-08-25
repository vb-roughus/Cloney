"""FastAPI-Oberfläche.

Die tragende Ansicht ist der Chunk-Editor: eine Zeile pro Satz mit Status,
Fehlerrate, Rückschrift der Spracherkennung und einem eigenen Abspieler. Von
dort lässt sich ein einzelner Satz neu würfeln oder umformulieren, ohne das
ganze Kapitel neu zu rendern -- genau der Arbeitsschritt, an dem eine
Langform-Produktion sonst scheitert.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cloney.config import Settings, get_settings
from cloney.core.project import ChunkStatus, Project
from cloney.core.voices import TYPICAL_CHARS_PER_SECOND, VoiceStore, suggested_speed
from cloney.engines.base import EngineError
from cloney.engines.registry import available_engines, create_engine, engine_info
from cloney.pipeline import synthesize_chunks
from cloney.web.jobs import JobRunner

_HERE = Path(__file__).parent

STATUS_LABEL = {
    ChunkStatus.PENDING: ("offen", "pending"),
    ChunkStatus.SYNTHESIZED: ("erzeugt", "pending"),
    ChunkStatus.OK: ("in Ordnung", "ok"),
    ChunkStatus.NEEDS_REVIEW: ("prüfen", "warn"),
    ChunkStatus.FAILED: ("fehlgeschlagen", "fail"),
}


def create_app(settings: Settings | None = None, asr_factory=None) -> FastAPI:  # noqa: ANN001
    settings = settings or get_settings()
    settings.ensure_dirs()

    app = FastAPI(title="Cloney")
    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
    templates = Jinja2Templates(directory=_HERE / "templates")
    templates.env.globals["typical_rate"] = TYPICAL_CHARS_PER_SECOND
    templates.env.globals["status_label"] = lambda s: STATUS_LABEL[s][0]
    templates.env.globals["status_class"] = lambda s: STATUS_LABEL[s][1]

    runner = JobRunner(settings)
    voices = VoiceStore(settings.voices_dir)

    def load(project_id: str) -> Project:
        try:
            root = Project.resolve(settings.projects_dir, project_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not (root / "project.json").exists():
            raise HTTPException(404, f"Projekt '{project_id}' gibt es nicht")
        return Project.load(root)

    def guard_idle(project_id: str) -> None:
        if runner.is_running(project_id):
            raise HTTPException(409, "Es läuft gerade ein Renderlauf")

    def _reference_context(project: Project) -> dict[str, object]:
        """Referenzstimme samt Sprechtempo.

        F5-TTS übernimmt die Geschwindigkeit der Referenz. Wer ruhiger vorgelesen
        haben will als sein Vorbild spricht, stellt das über den Regler ein --
        deshalb steht hier neben dem gemessenen Tempo gleich der Wert, der es auf
        angenehmes Zuhören bringt.
        """
        voice = voices.get(project.voice) if voices.exists(project.voice) else None
        rate = voices.speaking_rate(project.voice) if voice else None
        vorschlag = suggested_speed(rate)
        # Nur anbieten, solange der Regler nicht schon von Hand gesetzt wurde.
        if "speed" in project.engine_options or engine_info(project.engine).option("speed") is None:
            vorschlag = None
        return {"voice": voice, "reference_rate": rate, "speed_suggestion": vorschlag}

    def render_row(request: Request, project: Project, index: int) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_chunk_row.html", {"project": project, "chunk": project.chunks[index]}
        )

    # -- Projekte ---------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "projects": Project.list_all(settings.projects_dir),
                "voices": voices.list_all(),
                "engines": available_engines(),
                "default_engine": settings.engine,
            },
        )

    @app.post("/projects")
    def create_project(
        name: str = Form(...),
        text: str = Form(...),
        voice: str = Form(...),
        engine: str = Form(...),
    ) -> RedirectResponse:
        if not text.strip():
            raise HTTPException(400, "Der Text ist leer")
        if not voices.exists(voice):
            raise HTTPException(400, f"Stimme '{voice}' gibt es nicht")

        info = engine_info(engine)
        reference = voices.get(voice)
        if info.requires_ref_text and not reference.transcript.strip():
            raise HTTPException(
                400,
                f"Die Engine '{engine}' braucht den Wortlaut der Referenzaufnahme, "
                f"'{voice}' hat aber keinen hinterlegt.",
            )

        project = Project.create(
            name=name.strip() or "Ohne Titel",
            text=text,
            voice=voice,
            engine=info,
            projects_dir=settings.projects_dir,
            reference_seconds=reference.duration_s,
            chars_per_second=settings.chars_per_second,
            target_seconds=settings.target_chunk_seconds,
            max_seconds=settings.max_chunk_seconds,
        )
        return RedirectResponse(f"/projects/{project.id}", status_code=303)

    @app.get("/projects/{project_id}", response_class=HTMLResponse)
    def project_view(request: Request, project_id: str) -> HTMLResponse:
        project = load(project_id)
        return templates.TemplateResponse(
            request,
            "project.html",
            {
                "project": project,
                "engine": engine_info(project.engine),
                "running": runner.is_running(project_id),
                "threshold": settings.cer_threshold,
                **_reference_context(project),
            },
        )

    @app.post("/projects/{project_id}/run", response_class=HTMLResponse)
    def start_run(request: Request, project_id: str) -> HTMLResponse:
        project = load(project_id)
        runner.start(project.root, asr_factory)
        return templates.TemplateResponse(
            request, "_status.html", {"project": project, "running": True}
        )

    @app.get("/projects/{project_id}/status", response_class=HTMLResponse)
    def status(request: Request, project_id: str) -> HTMLResponse:
        project = load(project_id)
        job = runner.get(project_id)
        return templates.TemplateResponse(
            request,
            "_status.html",
            {
                "project": project,
                "running": runner.is_running(project_id),
                "job": job.snapshot() if job else None,
            },
        )

    @app.get("/projects/{project_id}/table", response_class=HTMLResponse)
    def table(request: Request, project_id: str) -> HTMLResponse:
        project = load(project_id)
        return templates.TemplateResponse(
            request, "_chunk_table.html", {"project": project, "threshold": settings.cer_threshold}
        )

    @app.post("/projects/{project_id}/rename", response_class=HTMLResponse)
    def rename_project(request: Request, project_id: str, name: str = Form(...)) -> HTMLResponse:
        project = load(project_id)
        if not name.strip():
            raise HTTPException(400, "Der Name darf nicht leer sein")
        project.rename(name)
        return HTMLResponse(f"<h1>{escape(project.name)}</h1>")

    @app.post("/projects/{project_id}/delete")
    def delete_project(project_id: str) -> RedirectResponse:
        project = load(project_id)
        guard_idle(project_id)
        project.delete()
        return RedirectResponse("/", status_code=303)

    @app.post("/projects/{project_id}/duplicate")
    def duplicate_project(project_id: str) -> RedirectResponse:
        """Gleiche Vorlage, neues Projekt -- der gefahrlose Weg, eine andere
        Reglerstellung zu hören, ohne das Vorhandene zu verlieren."""
        project = load(project_id)
        kopie = project.duplicate(f"{project.name} (Kopie)", settings.projects_dir)
        return RedirectResponse(f"/projects/{kopie.id}", status_code=303)

    @app.post("/projects/{project_id}/discard", response_class=HTMLResponse)
    def discard_all(request: Request, project_id: str) -> HTMLResponse:
        """Erzeugten Ton verwerfen, Seeds behalten."""
        project = load(project_id)
        guard_idle(project_id)
        project.discard_all_audio()
        return templates.TemplateResponse(
            request, "_status.html", {"project": project, "running": False}
        )

    @app.post("/projects/{project_id}/options", response_class=HTMLResponse)
    async def set_options(request: Request, project_id: str) -> HTMLResponse:
        """Regler der Engine verstellen.

        Vorhandene Sätze bleiben stehen. So lässt sich eine Einstellung an einem
        einzelnen Satz abhören, bevor ein ganzes Kapitel dafür neu läuft.
        """
        project = load(project_id)
        guard_idle(project_id)

        info = engine_info(project.engine)
        formular = await request.form()
        roh = {o.key: formular.get(o.key) for o in info.options if formular.get(o.key) is not None}
        # Zusammenführen statt ersetzen: eine Teilangabe soll die übrigen Regler
        # stehen lassen, nicht stillschweigend auf den Standard zurückwerfen.
        project.engine_options = {**project.engine_options, **info.clean_options(roh)}
        project.save()
        return templates.TemplateResponse(
            request,
            "_settings.html",
            {"project": project, "engine": info, **_reference_context(project)},
        )

    @app.post("/projects/{project_id}/rerender", response_class=HTMLResponse)
    def rerender_all(request: Request, project_id: str) -> HTMLResponse:
        """Alle Sätze zum Neurendern vormerken -- ohne sie schon zu erzeugen."""
        project = load(project_id)
        guard_idle(project_id)
        for chunk in project.chunks:
            project.reroll(chunk.index)
        project.save()
        return templates.TemplateResponse(
            request, "_status.html", {"project": project, "running": False}
        )

    # -- Einzelne Chunks --------------------------------------------------

    def _resynthesize(project: Project, index: int) -> None:
        """Einen einzelnen Chunk neu erzeugen.

        Schlägt das fehl -- fehlende Stimme, Modell nicht ladbar, Server weg --,
        dann als HTTP-Fehler mit dem Grund im Text. Sonst bliebe im Browser ein
        nacktes 'Internal Server Error' übrig, und der eigentliche Hinweis stünde
        nur im Serverlog.
        """
        try:
            synthesize_chunks(
                project,
                [project.chunks[index]],
                voices,
                lambda: create_engine(project.engine, settings, project.engine_options),
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                400,
                f"Die Stimme '{project.voice}' ist nicht mehr vorhanden. "
                "Unter 'Stimmen' neu anlegen oder ein neues Projekt beginnen.",
            ) from exc
        except (EngineError, ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

        chunk = project.chunks[index]
        if chunk.status == ChunkStatus.FAILED:
            # Die Pipeline schreibt Fehler ins Manifest, statt sie zu werfen --
            # hier soll der Klick aber sichtbar quittiert werden.
            raise HTTPException(400, chunk.error or "Der Chunk konnte nicht erzeugt werden.")

    @app.post("/projects/{project_id}/chunks/{index}/reroll", response_class=HTMLResponse)
    def reroll(request: Request, project_id: str, index: int) -> HTMLResponse:
        project = load(project_id)
        if runner.is_running(project_id):
            raise HTTPException(409, "Es läuft gerade ein Renderlauf")
        project.reroll(index)
        _resynthesize(project, index)
        return render_row(request, project, index)

    @app.post("/projects/{project_id}/chunks/{index}/text", response_class=HTMLResponse)
    def retext(
        request: Request, project_id: str, index: int, raw_text: str = Form(...)
    ) -> HTMLResponse:
        project = load(project_id)
        if runner.is_running(project_id):
            raise HTTPException(409, "Es läuft gerade ein Renderlauf")
        project.retext(index, raw_text)
        _resynthesize(project, index)
        return render_row(request, project, index)

    @app.post("/projects/{project_id}/chunks/{index}/discard", response_class=HTMLResponse)
    def discard_chunk(request: Request, project_id: str, index: int) -> HTMLResponse:
        """Ton eines Satzes verwerfen, Seed behalten -- so lässt sich die Wirkung
        einer Reglerstellung beurteilen, ohne dass zugleich der Zufall wechselt."""
        project = load(project_id)
        guard_idle(project_id)
        project.discard_audio(index)
        project.save()
        return render_row(request, project, index)

    @app.post("/projects/{project_id}/chunks/{index}/accept", response_class=HTMLResponse)
    def accept(request: Request, project_id: str, index: int) -> HTMLResponse:
        """Markierten Chunk trotz erhöhter Fehlerrate durchwinken."""
        project = load(project_id)
        project.chunks[index].status = ChunkStatus.OK
        project.save()
        return render_row(request, project, index)

    @app.get("/projects/{project_id}/chunks/{index}/audio")
    def chunk_audio(project_id: str, index: int) -> FileResponse:
        project = load(project_id)
        path = project.chunk_path(index)
        if not path.exists():
            raise HTTPException(404, "Für diesen Chunk gibt es noch kein Audio")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/projects/{project_id}/output")
    def output(project_id: str) -> FileResponse:
        project = load(project_id)
        if not project.output_path.exists():
            raise HTTPException(404, "Es gibt noch keine fertige Spur")
        return FileResponse(
            project.output_path, media_type="audio/wav", filename=f"{project.id}.wav"
        )

    # -- Stimmen -----------------------------------------------------------

    @app.post("/voices/{name}/transcript", response_class=HTMLResponse)
    def set_transcript(request: Request, name: str, transcript: str = Form("")) -> HTMLResponse:
        """Wortlaut ändern und die Aufnahme neu beurteilen.

        Das Sprechtempo ergibt sich aus Transkript und Dauer -- ein geänderter
        Wortlaut ändert also den Befund, ohne dass die Aufnahme angefasst wurde.
        """
        if not voices.exists(name):
            raise HTTPException(404, f"Stimme '{name}' gibt es nicht")
        voices.set_transcript(name, transcript)
        check = voices.recheck(name)
        return templates.TemplateResponse(
            request, "voices.html", {"voices": voices.list_all(), "check": check, "geprueft": name}
        )

    @app.post("/voices/{name}/delete")
    def delete_voice(name: str) -> RedirectResponse:
        if not voices.exists(name):
            raise HTTPException(404, f"Stimme '{name}' gibt es nicht")
        benutzt = [p.name for p in Project.list_all(settings.projects_dir) if p.voice == name]
        if benutzt:
            raise HTTPException(
                409,
                f"'{name}' wird noch verwendet von: {', '.join(benutzt[:5])}"
                + (" und weiteren" if len(benutzt) > 5 else "")
                + ". Diese Projekte zuerst löschen.",
            )
        voices.delete(name)
        return RedirectResponse("/voices", status_code=303)

    @app.get("/voices/{name}/audio")
    def voice_audio(name: str) -> FileResponse:
        if not voices.exists(name):
            raise HTTPException(404, f"Stimme '{name}' gibt es nicht")
        return FileResponse(voices.get(name).audio_path, media_type="audio/wav")

    @app.get("/voices", response_class=HTMLResponse)
    def voice_list(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "voices.html", {"voices": voices.list_all(), "check": None}
        )

    @app.post("/voices", response_class=HTMLResponse)
    async def add_voice(
        request: Request,
        name: str = Form(...),
        transcript: str = Form(""),
        audio: UploadFile = File(...),
    ) -> HTMLResponse:
        upload_dir = settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp = upload_dir / (audio.filename or "referenz.wav")
        temp.write_bytes(await audio.read())

        try:
            _, check = voices.add(
                name,
                temp,
                transcript=transcript,
                min_seconds=settings.ref_min_seconds,
                max_seconds=settings.ref_max_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Referenzaufnahme nicht lesbar: {exc}") from exc
        finally:
            temp.unlink(missing_ok=True)

        return templates.TemplateResponse(
            request, "voices.html", {"voices": voices.list_all(), "check": check}
        )

    return app


def get_app() -> FastAPI:
    """Einstiegspunkt für "uvicorn cloney.web.app:get_app --factory"."""
    return create_app()
