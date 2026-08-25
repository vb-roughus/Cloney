"""FastAPI-Oberfläche.

Die tragende Ansicht ist der Chunk-Editor: eine Zeile pro Satz mit Status,
Fehlerrate, Rückschrift der Spracherkennung und einem eigenen Abspieler. Von
dort lässt sich ein einzelner Satz neu würfeln oder umformulieren, ohne das
ganze Kapitel neu zu rendern -- genau der Arbeitsschritt, an dem eine
Langform-Produktion sonst scheitert.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cloney.config import Settings, get_settings
from cloney.core.project import ChunkStatus, Project
from cloney.core.voices import VoiceStore
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
    templates.env.globals["status_label"] = lambda s: STATUS_LABEL[s][0]
    templates.env.globals["status_class"] = lambda s: STATUS_LABEL[s][1]

    runner = JobRunner(settings)
    voices = VoiceStore(settings.voices_dir)

    def load(project_id: str) -> Project:
        root = settings.projects_dir / project_id
        if not (root / "project.json").exists():
            raise HTTPException(404, f"Projekt '{project_id}' gibt es nicht")
        return Project.load(root)

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

        project = Project.create(
            name=name.strip() or "Ohne Titel",
            text=text,
            voice=voice,
            engine=engine,
            sample_rate=engine_info(engine).sample_rate,
            projects_dir=settings.projects_dir,
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

    # -- Einzelne Chunks --------------------------------------------------

    def _resynthesize(project: Project, index: int) -> None:
        synthesize_chunks(
            project,
            [project.chunks[index]],
            voices,
            lambda: create_engine(project.engine, settings),
        )

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
