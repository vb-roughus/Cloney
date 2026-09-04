"""FastAPI-Oberfläche.

Die tragende Ansicht ist der Chunk-Editor: eine Zeile pro Satz mit Status,
Fehlerrate, Rückschrift der Spracherkennung und einem eigenen Abspieler. Von
dort lässt sich ein einzelner Satz neu würfeln oder umformulieren, ohne das
ganze Kapitel neu zu rendern -- genau der Arbeitsschritt, an dem eine
Langform-Produktion sonst scheitert.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cloney.config import Settings, get_settings
from cloney.core.audio import describe_audio, media_type
from cloney.core.compare import MAX_VARIANTS, Comparison, build_variants, pruefe_raster
from cloney.core.lexicon import Lexicon
from cloney.core.models import ModelError, ModelStore, settings_for
from cloney.core.project import ChunkStatus, Project
from cloney.core.pronounce import acronyms, spell_out
from cloney.core.voices import TYPICAL_CHARS_PER_SECOND, VoiceStore, suggested_speed
from cloney.engines.base import EngineError
from cloney.engines.registry import available_engines, create_engine, engine_info
from cloney.pipeline import AssembleError, build_prototype, quality_check, synthesize_chunks
from cloney.web.filters import LABELS, select
from cloney.web.jobs import ComparisonRunner, JobRunner
from cloney.web.overview import summarize

_HERE = Path(__file__).parent

STATUS_LABEL = {
    ChunkStatus.PENDING: ("offen", "pending"),
    ChunkStatus.SYNTHESIZED: ("erzeugt", "pending"),
    ChunkStatus.OK: ("in Ordnung", "ok"),
    ChunkStatus.NEEDS_REVIEW: ("prüfen", "warn"),
    ChunkStatus.FAILED: ("fehlgeschlagen", "fail"),
}


def create_app(
    settings: Settings | None = None, asr_factory=None, embedder_factory=None
) -> FastAPI:  # noqa: ANN001
    settings = settings or get_settings()
    settings.ensure_dirs()

    app = FastAPI(title="Cloney")
    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
    templates = Jinja2Templates(directory=_HERE / "templates")
    templates.env.globals["typical_rate"] = TYPICAL_CHARS_PER_SECOND
    templates.env.globals["anzahl"] = anzahl
    templates.env.filters["zeitpunkt"] = zeitpunkt
    templates.env.globals["status_label"] = lambda s: STATUS_LABEL[s][0]
    templates.env.globals["status_class"] = lambda s: STATUS_LABEL[s][1]
    templates.env.globals["tonstand"] = tonstand

    runner = JobRunner(settings)
    vergleiche = ComparisonRunner(settings)
    voices = VoiceStore(settings.voices_dir)
    modelle = ModelStore(settings.models_dir)

    def datei(path: Path, media_type: str) -> FileResponse:
        """Eine Tondatei ausliefern, ohne sie im Browser altern zu lassen.

        Die Kopfzeile allein reicht nicht -- siehe ``tonstand``. Sie bleibt
        trotzdem: sie deckt den Fall ab, dass jemand die Adresse ohne Stand
        aufruft, etwa aus einem Lesezeichen.
        """
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache"})

    def lexikon() -> Lexicon:
        """Das Aussprache-Wörterbuch, frisch von Platte.

        Nicht zwischengespeichert: es wird selten gelesen und darf nie älter
        sein als das, was auf der Verwaltungsseite gerade eingetragen wurde.
        """
        return Lexicon.load(settings.data_dir)

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

    def modell_settings(name: str) -> Settings:
        """Einstellungen, die auf einen trainierten Stand zeigen.

        Ohne Namen bleibt es beim Pretrain aus der Konfiguration. Ein Eintrag,
        dessen Checkpoint nicht mehr liegt, muss hier auffallen -- sonst
        renderte der Klick stillschweigend gegen ein anderes Modell.
        """
        if not name:
            return settings
        try:
            return settings_for(modelle.get(name), settings)
        except ModelError as exc:
            raise HTTPException(400, str(exc)) from exc

    def pruefe_modell(name: str | None) -> str | None:
        """Leerer Name heißt Pretrain, gar keiner heißt: lass ihn, wie er ist."""
        if name and not modelle.exists(name):
            eingetragen = ", ".join(m.name for m in modelle.list_all()) or "keins"
            raise HTTPException(400, f"Modell '{name}' gibt es nicht. Eingetragen: {eingetragen}")
        return name

    def _reference_context(project: Project) -> dict[str, object]:
        """Referenzstimme samt Sprechtempo.

        F5-TTS übernimmt die Geschwindigkeit der Referenz. Wer ruhiger vorgelesen
        haben will als sein Vorbild spricht, stellt das über den Regler ein --
        deshalb steht hier neben dem gemessenen Tempo gleich der Wert, der es auf
        angenehmes Zuhören bringt.
        """
        voice = voices.get(project.voice) if voices.exists(project.voice) else None
        rate = voices.speaking_rate(project.voice) if voice else None
        info = engine_info(project.engine)
        vorschlag = suggested_speed(rate)
        # Nur anbieten, wenn die Engine ihr Tempo überhaupt aus der Referenz
        # ableitet -- und solange der Regler nicht schon von Hand gesetzt wurde.
        if (
            not info.derives_tempo_from_reference
            or "speed" in project.engine_options
            or info.option("speed") is None
        ):
            vorschlag = None
        return {"voice": voice, "reference_rate": rate, "speed_suggestion": vorschlag}

    def _lagen_der_stimme(voice: str) -> list[str]:
        """Welche Lagen diese Stimme kennt. Leer, wenn es sie nicht mehr gibt.

        Einmal je Anfrage nachgeschlagen und nicht je Satz: ein Kapitel hat
        hundert Sätze und dieselbe Stimme.
        """
        return voices.lagen(voice) if voices.exists(voice) else []

    def _tabelle(
        project: Project,
        status: str = "alle",
        q: str = "",
        kompakt: bool = False,
        offen: tuple[int, ...] = (),
    ) -> dict[str, object]:
        """Kontext der Satztabelle. Eine Stelle, damit die Seite und ihr
        Nachladen nicht mit verschiedenen Filtern enden.

        Filter, Klappzustand und die einzeln ausgeklappten Sätze hängen alle an
        der Adresse. Die Tabelle tauscht sich während eines Laufs alle zwei
        Sekunden aus -- was nur im DOM stünde, wäre nach dem ersten Tausch weg.
        """

        def adresse(**abweichung: object) -> str:
            """Die Adresse dieser Tabelle, wahlweise mit geändertem Zustand."""
            werte: dict[str, object] = {
                "status": status,
                "q": q,
                "kompakt": kompakt,
                "offen": list(offen),
            }
            werte.update(abweichung)
            return tabellenadresse(
                str(werte["status"]),
                str(werte["q"]),
                bool(werte["kompakt"]),
                werte["offen"],  # type: ignore[arg-type]
            )

        def klappziel(index: int) -> str:
            """Die Adresse, die genau diesen Satz auf- oder zuklappt."""
            gewechselt = set(offen) ^ {index}
            return adresse(offen=sorted(gewechselt))

        return {
            "project": project,
            "threshold": settings.cer_threshold,
            "running": runner.is_running(project.id),
            "auswahl": select(project.chunks, status, q),
            "gruppen": LABELS,
            "lagen": _lagen_der_stimme(project.voice),
            "grenze_zeichen": _grenze_zeichen(project),
            "kompakt": kompakt,
            "offen": set(offen),
            "adresse": adresse,
            "klappziel": klappziel,
        }

    def _grenze_zeichen(project: Project) -> int:
        """Ab wie vielen Zeichen ein Satz der Engine zu lang wird. 0 = kein Limit.

        Nur Engines mit einer Grenze je Generierung haben eine: alle anderen
        sprechen auch einen langen Satz am Stück, und eine Warnung wäre dort
        keine. Von Hand verschmolzene Sätze können darüber geraten -- deshalb
        steht die Zahl in der Tabelle und nicht als Verbot im Handgriff.
        """
        if engine_info(project.engine).max_generation_seconds is None:
            return 0
        return int(project.target_chunk_seconds * settings.chars_per_second)

    def _offene(roh: str) -> tuple[int, ...]:
        """'0,3,7' zu Zahlen. Unlesbares wird übergangen -- der Wert kommt aus
        einer Adresse, und ein Tippfehler darin soll die Tabelle nicht kippen."""
        gelesen = []
        for stueck in roh.replace(",", " ").split():
            try:
                gelesen.append(int(stueck))
            except ValueError:
                continue
        return tuple(dict.fromkeys(gelesen))

    def render_row(request: Request, project: Project, index: int) -> HTMLResponse:
        """Eine einzelne Satzzeile -- und die Nachricht, dass der Zähler nicht
        mehr stimmt.

        Ein Handgriff an einer Zeile ändert den Fortschritt des Projekts: ein
        verworfener Ton, eine gewechselte Lage, ein nachgerenderter Satz. Die
        Statusleiste steht aber außerhalb der Zeile und wird nicht mit
        ausgetauscht. Ohne dieses Ereignis bliebe dort eine Zahl stehen, die
        nicht mehr gilt.
        """
        return templates.TemplateResponse(
            request,
            "_chunk_row.html",
            {
                "project": project,
                "chunk": project.chunks[index],
                "lagen": _lagen_der_stimme(project.voice),
                "grenze_zeichen": _grenze_zeichen(project),
            },
            headers={"HX-Trigger": "satz-geaendert"},
        )

    # -- Übersicht --------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        """Was steht an? Die Startseite beantwortet das, statt ein Formular zu
        zeigen, das man höchstens einmal je Kapitel braucht."""
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                **_uebersicht_kontext(),
                "comparisons": Comparison.list_all(settings.comparisons_dir)[:3],
                "models": modelle.list_all(),
                "voices": voices.list_all(),
            },
        )

    def _uebersicht_kontext() -> dict[str, object]:
        """Zahlen und laufende Läufe. Eine Stelle, damit die Seite und ihre
        Nachlade-Route nicht mit verschiedenen Ständen enden."""
        projekte = Project.list_all(settings.projects_dir)
        laufend = [p for p in projekte if runner.is_running(p.id)]
        return {
            "uebersicht": summarize(
                projekte,
                running=len(laufend),
                voices=len(voices.list_all()),
                comparisons=len(Comparison.list_all(settings.comparisons_dir)),
                models=len(modelle.list_all()),
            ),
            "projects": projekte[:5],
            "laufend": laufend,
        }

    @app.get("/uebersicht", response_class=HTMLResponse)
    def uebersicht_teil(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "_uebersicht.html", _uebersicht_kontext())

    # -- Projekte ---------------------------------------------------------

    @app.get("/projects", response_class=HTMLResponse)
    def project_index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "projects.html",
            {
                "projects": Project.list_all(settings.projects_dir),
                "voices": voices.list_all(),
            },
        )

    @app.get("/projects/new", response_class=HTMLResponse)
    def new_project(request: Request) -> HTMLResponse:
        """Das Anlegen hat eine eigene Seite.

        Auf der Liste stünde ein Formular, das man je Kapitel einmal braucht,
        dauerhaft im Weg -- und aufgeklappt wäre es eine Klappbox mehr. Die
        Route steht vor '/projects/{project_id}', sonst finge die Kennung sie ab.
        """
        return templates.TemplateResponse(
            request,
            "project_new.html",
            {
                "voices": voices.list_all(),
                "engines": available_engines(),
                "models": modelle.list_all(),
                "default_engine": settings.engine,
            },
        )

    @app.post("/projects")
    def create_project(
        name: str = Form(...),
        text: str = Form(...),
        voice: str = Form(...),
        engine: str = Form(...),
        model: str = Form(""),
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
            model=pruefe_modell(model) or "",
            lexicon=lexikon(),
            projects_dir=settings.projects_dir,
            reference_seconds=voices.longest_reference_seconds(voice),
            chars_per_second=settings.chars_per_second,
            target_seconds=settings.target_chunk_seconds,
            max_seconds=settings.max_chunk_seconds,
        )
        return RedirectResponse(f"/projects/{project.id}", status_code=303)

    def project_page(request: Request, project: Project, **extra: object) -> HTMLResponse:
        """Die ganze Projektseite. Eine Stelle, damit Ansicht und Umbau nicht
        mit verschiedenen Zusammenstellungen enden."""
        return templates.TemplateResponse(
            request,
            "project.html",
            {
                "project": project,
                "engine": engine_info(project.engine),
                "running": runner.is_running(project.id),
                "threshold": settings.cer_threshold,
                "voices": voices.list_all(),
                "engines": available_engines(),
                "models": modelle.list_all(),
                **_tabelle(project),
                **_reference_context(project),
                **extra,
            },
        )

    @app.get("/projects/{project_id}", response_class=HTMLResponse)
    def project_view(request: Request, project_id: str) -> HTMLResponse:
        return project_page(request, load(project_id))

    @app.post("/projects/{project_id}/configure", response_class=HTMLResponse)
    def configure(
        request: Request,
        project_id: str,
        text: str = Form(...),
        voice: str = Form(...),
        engine: str = Form(...),
        model: str | None = Form(None),
    ) -> HTMLResponse:
        """Text, Stimme, Engine oder trainierten Stand eines Projekts ändern.

        Dieselben Angaben wie beim Anlegen -- nur dass hier nicht alles neu
        entsteht: Sätze, deren Sprechfassung gleich bleibt, behalten ihren Ton.
        """
        project = load(project_id)
        guard_idle(project_id)
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

        # Heißt 'umbau' und nicht 'bericht': unter dem Namen steht überall sonst
        # ein fertiger Satz für den Menschen. Beide zugleich hießen, dass die
        # Statusleiste hier den rohen Zähler ausgäbe und die Meldung einer Lage
        # umgekehrt als Zähler ohne Zahlen erschiene -- beides war zu sehen.
        umbau = project.reconfigure(
            text=text,
            voice=voice,
            engine=info,
            model=pruefe_modell(model),
            lexicon=lexikon(),
            reference_seconds=voices.longest_reference_seconds(voice),
            chars_per_second=settings.chars_per_second,
            target_seconds=settings.target_chunk_seconds,
            max_seconds=settings.max_chunk_seconds,
        )
        return project_page(request, project, umbau=umbau, aktiver_reiter="einstellungen")

    @app.post("/projects/{project_id}/run", response_class=HTMLResponse)
    def start_run(request: Request, project_id: str) -> HTMLResponse:
        project = load(project_id)
        runner.start(project.root, asr_factory, embedder_factory)
        # Der Knopf tauscht nur die Statusleiste aus. Die Satztabelle wurde
        # gerendert, als noch nichts lief, und käme von selbst nie in Gang --
        # dieses Ereignis weckt sie.
        return templates.TemplateResponse(
            request,
            "_status.html",
            {"project": project, "running": True},
            headers={"HX-Trigger": "lauf-gestartet"},
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
    def table(
        request: Request,
        project_id: str,
        status: str = "alle",
        q: str = "",
        kompakt: int = 0,
        offen: str = "",
    ) -> HTMLResponse:
        project = load(project_id)
        return templates.TemplateResponse(
            request,
            "_chunk_table.html",
            _tabelle(project, status, q, bool(kompakt), _offene(offen)),
        )

    @app.post("/projects/{project_id}/lage", response_class=HTMLResponse)
    def projekt_lage(request: Request, project_id: str, lage: str = Form("")) -> HTMLResponse:
        """Die Emotionslage des ganzen Projekts setzen.

        Getroffen wird nur, wer keine eigene trägt -- von Hand gesetzte Lagen
        überleben den Wechsel.
        """
        project = load(project_id)
        guard_idle(project_id)
        betroffen = project.set_default_lage(lage)
        bericht = f"Vorgabe auf „{lage or 'neutral'}“ gesetzt. " + (
            f"{anzahl(betroffen, 'Satz verliert', 'Sätze verlieren')} ihren Ton."
            if betroffen
            else "Kein Satz hatte Ton, der davon betroffen wäre."
        )
        return project_page(request, project, bericht=bericht, aktiver_reiter="einstellungen")

    @app.post("/projects/{project_id}/chunks/lage", response_class=HTMLResponse)
    async def saetze_lage(request: Request, project_id: str) -> HTMLResponse:
        """Eine Lage auf alle markierten Sätze anwenden.

        Der Filter und der Klappzustand reisen im Formular mit: die Antwort
        ersetzt die ganze Tabelle, und ohne sie stünde danach die ungefilterte,
        ausgeklappte Liste da.
        """
        project = load(project_id)
        guard_idle(project_id)
        formular = await request.form()
        indices = [int(i) for i in formular.getlist("auswahl") if str(i).isdigit()]
        gueltig = [i for i in indices if 0 <= i < len(project.chunks)]
        project.set_lage_many(gueltig, str(formular.get("lage") or ""))
        return templates.TemplateResponse(
            request,
            "_chunk_table.html",
            _tabelle(
                project,
                str(formular.get("status") or "alle"),
                str(formular.get("q") or ""),
                str(formular.get("kompakt") or "0") == "1",
            ),
            headers={"HX-Trigger": "satz-geaendert"},
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
        return RedirectResponse("/projects", status_code=303)

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
            {
                "project": project,
                "engine": info,
                "voices": voices.list_all(),
                "engines": available_engines(),
                "models": modelle.list_all(),
                "lagen": _lagen_der_stimme(project.voice),
                **_reference_context(project),
            },
        )

    @app.post("/projects/{project_id}/refresh-spoken", response_class=HTMLResponse)
    def refresh_spoken(request: Request, project_id: str) -> HTMLResponse:
        """Sprechfassungen aller Sätze neu erzeugen.

        Der Weg nach einer Änderung am Aussprache-Wörterbuch: betroffen sind nur
        die Sätze, deren Sprechfassung sich dadurch ändert -- die übrigen
        behalten ihren Ton.
        """
        project = load(project_id)
        guard_idle(project_id)
        geaendert = project.refresh_all_spoken(lexikon())
        return templates.TemplateResponse(
            request,
            "_status.html",
            {
                "project": project,
                "running": False,
                "bericht": (
                    f"{len(geaendert)} Satz/Sätze neu gefasst und zum Rendern vorgemerkt"
                    if geaendert
                    else "Keine Sprechfassung hat sich geändert"
                ),
            },
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
        """Einen einzelnen Chunk neu erzeugen und gleich messen.

        Die Messung gehört dazu: Ohne sie bliebe der Satz ohne Fehlerrate stehen,
        und der Referenz-Vorspann würde nicht abgeschnitten -- gerade beim
        Abhören einer Reglerstellung, wo einzelne Sätze neu erzeugt werden, wäre
        das Ergebnis also ein anderes als im vollständigen Lauf.

        Schlägt etwas fehl -- fehlende Stimme, Modell nicht ladbar, Server weg --,
        dann als HTTP-Fehler mit dem Grund im Text. Sonst bliebe im Browser ein
        nacktes 'Internal Server Error' übrig, und der eigentliche Hinweis stünde
        nur im Serverlog.
        """
        try:
            # Gegen denselben Stand wie der volle Lauf. Ohne das würfelte ein
            # einzelner Satz gegen den Pretrain und klänge neben seinen
            # Nachbarn nach einem anderen Sprecher.
            eigene = modell_settings(project.model)
            synthesize_chunks(
                project,
                [project.chunks[index]],
                voices,
                lambda: create_engine(project.engine, eigene, project.engine_options),
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

        try:
            quality_check(project, settings, asr_factory)
        except (EngineError, ValueError, RuntimeError) as exc:
            # Der Ton steht bereits; nur die Messung fehlt. Das ist kein Grund,
            # den Klick als gescheitert zu melden.
            chunk.error = f"Erzeugt, aber nicht gemessen: {exc}"
            project.save()

    @app.post("/projects/{project_id}/chunks/{index}/reroll", response_class=HTMLResponse)
    def reroll(request: Request, project_id: str, index: int) -> HTMLResponse:
        project = load(project_id)
        if runner.is_running(project_id):
            raise HTTPException(409, "Es läuft gerade ein Renderlauf")
        # Vor dem Würfeln die Sprechfassung auffrischen: ein Eintrag im
        # Aussprache-Wörterbuch, der nach dem Anlegen dazukam, soll hier wirken.
        # Sonst hörte man weiter die alte Fassung und suchte den Fehler im Modell.
        project.refresh_spoken(index, lexikon())
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
        project.retext(index, raw_text, lexikon())
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

    @app.get("/projects/{project_id}/chunks/{index}/lage", response_class=HTMLResponse)
    def lage_waehlen(request: Request, project_id: str, index: int) -> HTMLResponse:
        """Die Auswahl der Lagen -- erst auf Klick, nicht in jeder Zeile."""
        project = load(project_id)
        return templates.TemplateResponse(
            request,
            "_lage_auswahl.html",
            {
                "project": project,
                "chunk": project.chunks[index],
                "lagen": _lagen_der_stimme(project.voice),
            },
        )

    @app.post("/projects/{project_id}/chunks/{index}/lage", response_class=HTMLResponse)
    def lage_setzen(
        request: Request, project_id: str, index: int, lage: str = Form("")
    ) -> HTMLResponse:
        """Lage eines Satzes wechseln. Der Ton fällt weg, der Seed bleibt.

        Gerendert wird hier nicht: wer ein Kapitel durchgeht, vergibt erst die
        Lagen und lässt danach in einem Zug laufen. Für den einzelnen Satz steht
        'Jetzt rendern' daneben.
        """
        project = load(project_id)
        guard_idle(project_id)
        project.set_lage(index, lage)
        project.save()
        return render_row(request, project, index)

    @app.post("/projects/{project_id}/chunks/{index}/render", response_class=HTMLResponse)
    def render_chunk(request: Request, project_id: str, index: int) -> HTMLResponse:
        """Einen einzelnen Satz erzeugen, ohne Seed und Text anzurühren."""
        project = load(project_id)
        guard_idle(project_id)
        _resynthesize(project, index)
        return render_row(request, project, index)

    # -- Satzbau von Hand -------------------------------------------------

    def _satztabelle(
        request: Request,
        project: Project,
        status: str,
        q: str,
        kompakt: int,
        offen: tuple[int, ...],
    ) -> HTMLResponse:
        """Die ganze Tabelle als Antwort auf einen Handgriff am Satzbau.

        Eine einzelne Zeile genügte hier nicht: Einfügen und Verschmelzen
        verschieben alle Nummern dahinter, und die Zeilen tragen ihre Nummer in
        jeder Adresse mit sich.
        """
        return templates.TemplateResponse(
            request,
            "_chunk_table.html",
            _tabelle(project, status, q, bool(kompakt), offen),
            headers={"HX-Trigger": "satz-geaendert"},
        )

    @app.get("/projects/{project_id}/chunks/{index}/einfuegen", response_class=HTMLResponse)
    def satz_einfuegen_formular(
        request: Request,
        project_id: str,
        index: int,
        danach: int = 0,
        status: str = "alle",
        q: str = "",
        kompakt: int = 0,
        offen: str = "",
    ) -> HTMLResponse:
        """Das Feld für den neuen Satz -- erst auf Klick, wie die Lagenauswahl.

        In jeder Zeile dauerhaft ein zweites Textfeld zu zeigen machte aus der
        Tabelle eine Wand aus Eingabefeldern.
        """
        project = load(project_id)
        return templates.TemplateResponse(
            request,
            "_satz_einfuegen.html",
            {
                "project": project,
                "chunk": project.chunks[index],
                "danach": bool(danach),
                "tabellenadresse": tabellenadresse(status, q, bool(kompakt), _offene(offen)),
            },
        )

    @app.post("/projects/{project_id}/chunks/{index}/einfuegen", response_class=HTMLResponse)
    def satz_einfuegen(
        request: Request,
        project_id: str,
        index: int,
        raw_text: str = Form(...),
        danach: int = Form(0),
        status: str = "alle",
        q: str = "",
        kompakt: int = 0,
        offen: str = "",
    ) -> HTMLResponse:
        """Einen neuen Satz vor oder nach diesem einsetzen.

        Gerendert wird er nicht: wer ein Kapitel durchgeht, schreibt erst und
        lässt danach in einem Zug laufen. Für den einzelnen Satz steht 'Jetzt
        rendern' in seiner Zeile.
        """
        project = load(project_id)
        guard_idle(project_id)
        try:
            stelle = project.insert_chunk(index, raw_text, danach=bool(danach), lexicon=lexikon())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        # Der neue Satz steht offen da: eingeklappt wäre von ihm nur der eben
        # getippte Wortlaut zu sehen, und gerade er will noch gerendert werden.
        offene = tuple(i + 1 if i >= stelle else i for i in _offene(offen)) + (stelle,)
        return _satztabelle(request, project, status, q, kompakt, offene)

    @app.post("/projects/{project_id}/chunks/{index}/verschmelzen", response_class=HTMLResponse)
    def satz_verschmelzen(
        request: Request,
        project_id: str,
        index: int,
        status: str = "alle",
        q: str = "",
        kompakt: int = 0,
        offen: str = "",
    ) -> HTMLResponse:
        """Diesen Satz mit dem folgenden zu einem zusammenziehen."""
        project = load(project_id)
        guard_idle(project_id)
        try:
            stelle = project.merge_chunks(index, lexicon=lexikon())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        offene = tuple(i - 1 if i > stelle else i for i in _offene(offen)) + (stelle,)
        return _satztabelle(request, project, status, q, kompakt, offene)

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
        return datei(path, "audio/wav")

    @app.post("/projects/{project_id}/prototyp", response_class=HTMLResponse)
    def prototyp_bauen(request: Request, project_id: str) -> HTMLResponse:
        """Die bis jetzt erzeugten Sätze der Reihe nach zu einer Spur.

        Auch während eines Laufs: gelesen werden fertig geschriebene Dateien,
        und ins Manifest schreibt der Prototyp nichts -- dort führt der Lauf
        seinen eigenen Stand.
        """
        project = load(project_id)
        try:
            saetze, sekunden = build_prototype(project, settings)
        except AssembleError as exc:
            raise HTTPException(400, str(exc)) from None

        fehlend = len(project.chunks) - saetze
        bericht = (
            f"Prototyp aus {saetze} von "
            f"{anzahl(len(project.chunks), 'Satz', 'Sätzen')}, {dauer(sekunden)}."
        )
        if fehlend:
            bericht += (
                " Ein Satz fehlt und wurde übersprungen."
                if fehlend == 1
                else f" {fehlend} Sätze fehlen und wurden übersprungen."
            )
        return templates.TemplateResponse(
            request, "_prototyp.html", {"project": project, "bericht": bericht}
        )

    @app.get("/projects/{project_id}/prototyp")
    def prototyp_datei(project_id: str) -> FileResponse:
        project = load(project_id)
        if not project.prototype_path.exists():
            raise HTTPException(404, "Für dieses Projekt gibt es keinen Prototyp")
        return datei(project.prototype_path, "audio/wav")

    @app.get("/projects/{project_id}/output")
    def output(project_id: str) -> FileResponse:
        project = load(project_id)
        if not project.output_path.exists():
            raise HTTPException(404, "Es gibt noch keine fertige Spur")
        return FileResponse(
            project.output_path, media_type="audio/wav", filename=f"{project.id}.wav"
        )

    # -- Aussprache --------------------------------------------------------

    def _lexikon_seite(request: Request, **extra: object) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "lexicon.html",
            {
                "lexikon": lexikon(),
                "kandidaten": _offene_kandidaten(),
                **extra,
            },
        )

    def _offene_kandidaten() -> list[tuple[str, str, list[str]]]:
        """Ketten aus Großbuchstaben aus allen Projekttexten, ohne Eintrag.

        Woher ein Kandidat stammt, gehört dazu: dieselbe Abkürzung kann in zwei
        Büchern verschieden gemeint sein, und wer entscheidet, will wissen, in
        welchem Text sie steht.
        """
        eingetragen = {w.lower() for w in lexikon().entries}
        gefunden: dict[str, list[str]] = {}
        for project in Project.list_all(settings.projects_dir):
            for wort in acronyms(project.source_text):
                if wort.lower() in eingetragen:
                    continue
                gefunden.setdefault(wort, [])
                if project.name not in gefunden[wort]:
                    gefunden[wort].append(project.name)
        return [(wort, spell_out(wort), woher) for wort, woher in gefunden.items()]

    @app.get("/lexicon", response_class=HTMLResponse)
    def lexicon_page(request: Request) -> HTMLResponse:
        return _lexikon_seite(request)

    @app.post("/lexicon", response_class=HTMLResponse)
    def lexicon_set(
        request: Request, word: str = Form(...), spoken: str = Form(...)
    ) -> HTMLResponse:
        buch = lexikon()
        try:
            buch.set(word, spoken)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        buch.save(settings.data_dir)
        return _lexikon_seite(
            request, bericht=f"'{word.strip()}' wird gesprochen: {spoken.strip()}"
        )

    @app.post("/lexicon/{word}/edit", response_class=HTMLResponse)
    def lexicon_edit(
        request: Request, word: str, new_word: str = Form(...), spoken: str = Form(...)
    ) -> HTMLResponse:
        """Einen Eintrag ändern, Wort inbegriffen."""
        buch = lexikon()
        try:
            buch.rename(word, new_word, spoken)
        except KeyError as exc:
            raise HTTPException(404, f"'{word}' ist nicht eingetragen") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        buch.save(settings.data_dir)
        return _lexikon_seite(
            request, bericht=f"'{new_word.strip()}' wird gesprochen: {spoken.strip()}"
        )

    @app.post("/lexicon/{word}/delete", response_class=HTMLResponse)
    def lexicon_remove(request: Request, word: str) -> HTMLResponse:
        buch = lexikon()
        if not buch.remove(word):
            raise HTTPException(404, f"'{word}' ist nicht eingetragen")
        buch.save(settings.data_dir)
        return _lexikon_seite(request, bericht=f"'{word}' entfernt")

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
            request, "voices.html", _voice_context(check=check, geprueft=name)
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
        """Die Aufnahme so ausliefern, wie sie hereinkam.

        Der Medientyp ergibt sich aus der Endung: seit die Datei unverändert
        abgelegt wird, ist sie nicht mehr zwingend eine WAV, und ein falsch
        angegebener Typ hindert manche Browser am Abspielen.
        """
        if not voices.exists(name):
            raise HTTPException(404, f"Stimme '{name}' gibt es nicht")
        pfad = voices.get(name).audio_path
        return datei(pfad, media_type(pfad))

    # -- Vergleichsläufe --------------------------------------------------

    def load_comparison(comparison_id: str) -> Comparison:
        try:
            root = Comparison.resolve(settings.comparisons_dir, comparison_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not (root / "comparison.json").exists():
            raise HTTPException(404, f"Vergleich '{comparison_id}' gibt es nicht")
        return Comparison.load(root)

    def _grid(formular, info) -> dict[str, list[float]]:  # noqa: ANN001
        """Die Reglerwerte aus dem Formular -- je Regler eine Achse.

        Die Oberfläche schickt je Wert ein eigenes Auswahlfeld, also mehrere
        Felder desselben Namens. Kommagetrennt in einem Feld geht weiterhin: so
        kommen die Werte aus der Kommandozeile und aus älteren Lesezeichen.

        Unlesbares wird übergangen statt abgelehnt: ein Tippfehler in einer von
        drei Achsen soll nicht das ganze Formular zurückweisen.
        """
        achsen: dict[str, list[float]] = {}
        for option in info.options:
            werte: list[float] = []
            for feld in formular.getlist(f"werte_{option.key}"):
                for stueck in str(feld).replace(",", " ").split():
                    try:
                        werte.append(float(stueck))
                    except ValueError:
                        continue
            if werte:
                achsen[option.key] = werte
        return achsen

    def _zuschnitt(formular) -> dict[str, object]:  # noqa: ANN001
        """Alles, was ein Vergleich aus dem Formular braucht -- einmal geprüft.

        Anlegen und Ändern stellen dieselben Fragen; sie hier zweimal zu
        beantworten hieße, sie beim nächsten Mal an einer Stelle zu vergessen.
        """
        text = str(formular.get("text") or "")
        voice = str(formular.get("voice") or "")
        engine = str(formular.get("engine") or "")
        if not text.strip():
            raise HTTPException(400, "Die Textprobe ist leer. Sie steht hinter „Text bearbeiten“.")
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
        # Mehrfachauswahl: der leere Wert steht für den Pretrain, damit sich ein
        # Finetune gegen den Stand messen lässt, von dem er ausging.
        bekannte = set(voices.lagen(voice))
        return {
            "name": str(formular.get("name") or "").strip() or "Vergleich",
            "text": text,
            "voice": voice,
            "engine": info,
            "grid": _grid(formular, info),
            "models": [pruefe_modell(str(m)) for m in formular.getlist("models")],
            # Eine Lage, die es bei dieser Stimme nicht gibt, fiele beim Rendern
            # ohnehin auf die Hauptaufnahme zurück -- dann stünden zwei Zeilen
            # da, die dasselbe messen.
            "lagen": [str(x) for x in formular.getlist("lagen") if str(x) in bekannte],
        }

    def _achsen_kontext(
        engine: str,
        voice: str,
        formular=None,  # noqa: ANN001
    ) -> dict[str, object]:
        """Kontext der Achsen. Eine Stelle für Seite, Nachladen und Vorschau."""
        info = engine_info(engine)
        gewaehlt = _grid(formular, info) if formular is not None else {}
        return {
            "engine": info,
            "models": modelle.list_all(),
            "lagen_der_stimme": _lagen_der_stimme(voice),
            "gewaehlte_werte": gewaehlt,
            "gewaehlte_modelle": (
                [str(m) for m in formular.getlist("models")] if formular is not None else []
            ),
            "gewaehlte_lagen": (
                [str(x) for x in formular.getlist("lagen")] if formular is not None else []
            ),
            "max_variants": MAX_VARIANTS,
        }

    def _formular_kontext(comparison: Comparison | None, **extra: object) -> dict[str, object]:
        """Kontext der Zuschnittsmaske, aus einem bestehenden Vergleich oder leer."""
        engine = comparison.engine if comparison else settings.engine
        stimmen = voices.list_all()
        voice = comparison.voice if comparison else (stimmen[0].name if stimmen else "")
        info = engine_info(engine)
        # Ein neuer Vergleich startet auf den Vorgaben der Regler -- und zwar
        # hier und in der Maske mit demselben Wert. Getrennt gerechnet zeigte
        # die Maske '1' und die Vorschau daneben 'keine Variante'.
        vorbelegt = (
            {
                key: [v.options[key] for v in comparison.variants if key in v.options]
                for key in {k for v in comparison.variants for k in v.options}
            }
            if comparison
            else {o.key: [o.default] for o in info.options}
        )
        return {
            "comparison": comparison,
            "voices": stimmen,
            "engines": available_engines(),
            "engine": info,
            "models": modelle.list_all(),
            "lagen_der_stimme": _lagen_der_stimme(voice),
            "gewaehlte_werte": {k: sorted(set(w)) for k, w in vorbelegt.items()},
            "gewaehlte_modelle": list(comparison.models) if comparison else [],
            "gewaehlte_lagen": list(comparison.lagen) if comparison else [],
            "max_variants": MAX_VARIANTS,
            **_vorschau(
                info,
                {k: sorted(set(w)) for k, w in vorbelegt.items()},
                list(comparison.models) if comparison else [],
                list(comparison.lagen) if comparison else [],
            ),
            **extra,
        }

    def _vorschau(
        info, grid: dict[str, list[float]], models: list[str], lagen: list[str]
    ) -> dict[str, object]:  # noqa: ANN001
        """Die Zeilen, die dieser Zuschnitt ergäbe -- oder warum es keine gibt."""
        varianten = build_variants(info, grid, models=models, lagen=lagen)
        try:
            pruefe_raster(varianten)
        except ValueError as exc:
            return {"vorschau": varianten, "vorschau_fehler": str(exc)}
        return {"vorschau": varianten, "vorschau_fehler": None}

    @app.get("/comparisons", response_class=HTMLResponse)
    def comparison_index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "comparisons.html",
            _formular_kontext(None, comparisons=Comparison.list_all(settings.comparisons_dir)),
        )

    @app.get("/comparisons/achsen", response_class=HTMLResponse)
    def comparison_axes(request: Request) -> HTMLResponse:
        """Die Achsen hängen an Engine und Stimme -- beim Wechsel neu laden.

        Mitgeschickt wird das ganze Formular, damit die schon gewählten Werte
        stehen bleiben: wer nur die Stimme wechselt, soll nicht sein Raster
        verlieren. Gelesen wird aus der Adresse und nicht aus dem Rumpf: htmx
        hängt die eingeschlossenen Felder eines hx-get an die Adresse an, und
        eine GET-Anfrage hat ohnehin kein Formular im Rumpf.
        """
        formular = request.query_params
        return templates.TemplateResponse(
            request,
            "_comparison_axes.html",
            _achsen_kontext(
                str(formular.get("engine") or settings.engine),
                str(formular.get("voice") or ""),
                formular,
            ),
        )

    @app.get("/comparisons/wertfeld", response_class=HTMLResponse)
    def comparison_value_field(request: Request, engine: str, key: str) -> HTMLResponse:
        """Ein weiteres Auswahlfeld für eine Achse."""
        option = engine_info(engine).option(key)
        if option is None:
            raise HTTPException(404, f"Die Engine '{engine}' kennt keinen Regler '{key}'")
        return templates.TemplateResponse(
            request, "_wertfeld.html", {"option": option, "wert": option.default}
        )

    @app.get("/comparisons/vorschau", response_class=HTMLResponse)
    def comparison_preview(request: Request) -> HTMLResponse:
        """Welche Zeilen dieser Zuschnitt ergäbe -- vor dem Rendern.

        Wie bei den Achsen aus der Adresse gelesen, nicht aus dem Rumpf.
        """
        formular = request.query_params
        info = engine_info(str(formular.get("engine") or settings.engine))
        voice = str(formular.get("voice") or "")
        bekannte = set(voices.lagen(voice)) if voices.exists(voice) else set()
        return templates.TemplateResponse(
            request,
            "_comparison_preview.html",
            {
                "max_variants": MAX_VARIANTS,
                **_vorschau(
                    info,
                    _grid(formular, info),
                    [str(m) for m in formular.getlist("models")],
                    [str(x) for x in formular.getlist("lagen") if str(x) in bekannte],
                ),
            },
        )

    @app.post("/comparisons")
    async def create_comparison(request: Request) -> RedirectResponse:
        zuschnitt = _zuschnitt(await request.form())
        try:
            comparison = Comparison.create(**zuschnitt, comparisons_dir=settings.comparisons_dir)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return RedirectResponse(f"/comparisons/{comparison.id}", status_code=303)

    @app.get("/comparisons/{comparison_id}/edit", response_class=HTMLResponse)
    def edit_comparison(request: Request, comparison_id: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "comparison_edit.html", _formular_kontext(load_comparison(comparison_id))
        )

    @app.post("/comparisons/{comparison_id}/edit")
    async def apply_comparison(request: Request, comparison_id: str) -> RedirectResponse:
        comparison = load_comparison(comparison_id)
        if vergleiche.is_running(comparison_id):
            raise HTTPException(409, "Es läuft gerade ein Vergleich")
        try:
            comparison.reconfigure(**_zuschnitt(await request.form()))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return RedirectResponse(f"/comparisons/{comparison.id}", status_code=303)

    @app.get("/comparisons/{comparison_id}", response_class=HTMLResponse)
    def comparison_view(request: Request, comparison_id: str) -> HTMLResponse:
        comparison = load_comparison(comparison_id)
        return templates.TemplateResponse(
            request,
            "comparison.html",
            {
                "comparison": comparison,
                "engine": engine_info(comparison.engine),
                "running": vergleiche.is_running(comparison_id),
            },
        )

    @app.post("/comparisons/{comparison_id}/run", response_class=HTMLResponse)
    def start_comparison(request: Request, comparison_id: str) -> HTMLResponse:
        comparison = load_comparison(comparison_id)
        vergleiche.start(comparison.root, asr_factory, embedder_factory)
        return templates.TemplateResponse(
            request,
            "_comparison_table.html",
            {"comparison": comparison, "engine": engine_info(comparison.engine), "running": True},
        )

    @app.get("/comparisons/{comparison_id}/table", response_class=HTMLResponse)
    def comparison_table(request: Request, comparison_id: str) -> HTMLResponse:
        comparison = load_comparison(comparison_id)
        job = vergleiche.get(comparison_id)
        return templates.TemplateResponse(
            request,
            "_comparison_table.html",
            {
                "comparison": comparison,
                "engine": engine_info(comparison.engine),
                "running": vergleiche.is_running(comparison_id),
                "job": job.snapshot() if job else None,
            },
        )

    @app.get("/comparisons/{comparison_id}/variants/{slug}/audio")
    def variant_audio(comparison_id: str, slug: str) -> FileResponse:
        comparison = load_comparison(comparison_id)
        try:
            project = comparison.variant_project(slug)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if not project.output_path.exists():
            raise HTTPException(404, "Für diese Variante gibt es noch keinen Ton")
        return datei(project.output_path, "audio/wav")

    @app.post("/comparisons/{comparison_id}/delete")
    def delete_comparison(comparison_id: str) -> RedirectResponse:
        comparison = load_comparison(comparison_id)
        if vergleiche.is_running(comparison_id):
            raise HTTPException(409, "Es läuft gerade ein Vergleich")
        comparison.delete()
        return RedirectResponse("/comparisons", status_code=303)

    # -- Stimmen ----------------------------------------------------------

    def _voice_context(**extra: object) -> dict[str, object]:
        vorhandene = voices.list_all()
        formate = {}
        lagen: dict[str, list] = {}
        for stimme in vorhandene:
            # Eine unlesbare Datei darf die Liste nicht kippen.
            with contextlib.suppress(Exception):
                formate[stimme.name] = describe_audio(stimme.audio_path)
            # Die neutrale Lage steht schon als Hauptaufnahme da; hier sind nur
            # die weiteren gemeint.
            lagen[stimme.name] = voices.list_lagen(stimme.name)[1:]
        return {
            "voices": vorhandene,
            "formate": formate,
            "lagen": lagen,
            "check": None,
            **extra,
        }

    def _upload(datei: bytes, dateiname: str | None) -> Path:
        upload_dir = settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp = upload_dir / (dateiname or "referenz.wav")
        temp.write_bytes(datei)
        return temp

    @app.post("/voices/{name}/lagen", response_class=HTMLResponse)
    async def add_lage(
        request: Request,
        name: str,
        lage: str = Form(...),
        transcript: str = Form(""),
        audio: UploadFile = File(...),
    ) -> HTMLResponse:
        """Eine weitere Aufnahme derselben Stimme, für eine andere Lage."""
        if not voices.exists(name):
            raise HTTPException(404, f"Stimme '{name}' gibt es nicht")
        temp = _upload(await audio.read(), audio.filename)
        try:
            _, check = voices.add_lage(
                name,
                lage,
                temp,
                transcript=transcript,
                min_seconds=settings.ref_min_seconds,
                max_seconds=settings.ref_max_seconds,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Aufnahme nicht lesbar: {exc}") from exc
        finally:
            temp.unlink(missing_ok=True)
        return templates.TemplateResponse(
            request, "voices.html", _voice_context(check=check, geprueft=f"{name} / {lage}")
        )

    @app.post("/voices/{name}/lagen/{lage}/transcript", response_class=HTMLResponse)
    def lage_transcript(
        request: Request, name: str, lage: str, transcript: str = Form("")
    ) -> HTMLResponse:
        if not voices.exists(name):
            raise HTTPException(404, f"Stimme '{name}' gibt es nicht")
        try:
            check = voices.set_lage_transcript(name, lage, transcript)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return templates.TemplateResponse(
            request, "voices.html", _voice_context(check=check, geprueft=f"{name} / {lage}")
        )

    @app.post("/voices/{name}/lagen/{lage}/delete")
    def delete_lage(name: str, lage: str) -> RedirectResponse:
        if not voices.exists(name):
            raise HTTPException(404, f"Stimme '{name}' gibt es nicht")
        try:
            voices.delete_lage(name, lage)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return RedirectResponse("/voices", status_code=303)

    @app.get("/voices/{name}/lagen/{lage}/audio")
    def lage_audio(name: str, lage: str) -> FileResponse:
        if not voices.exists(name):
            raise HTTPException(404, f"Stimme '{name}' gibt es nicht")
        stimme = voices.get(name, lage)
        if not stimme.audio_path.exists():
            raise HTTPException(404, "Für diese Lage gibt es keine Aufnahme")
        return datei(stimme.audio_path, "audio/wav")

    @app.get("/voices", response_class=HTMLResponse)
    def voice_list(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "voices.html", _voice_context())

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

        return templates.TemplateResponse(request, "voices.html", _voice_context(check=check))

    return app


def tonstand(path: Path) -> str:
    """Wie alt der Ton unter diesem Pfad ist -- als Anhängsel für seine Adresse.

    Ein neu gerenderter Satz liegt unter derselben Adresse wie der alte, und ein
    <audio>-Element fragt für eine unveränderte Adresse **gar nicht erneut**.
    Im Browser gemessen: nach einem 'Text übernehmen und neu rendern' lagen auf
    Platte 8,11 Sekunden und im Abspieler weiterhin 3,90 -- bei genau einer
    einzigen Anfrage an die Tonadresse, ganz am Anfang. 'Cache-Control:
    no-cache' verlangt eine Rückfrage; wo keine Anfrage gestellt wird, gibt es
    auch nichts zurückzufragen.

    Deshalb wandert der Änderungszeitpunkt der Datei in die Adresse. Ändert sich
    der Ton, ändert sich die Adresse, und der Browser holt sie neu. Bleibt er
    gleich, bleibt sie gleich, und der Zwischenspeicher darf weiter greifen.

    Fehlt die Datei, steht dort eine Null -- die Adresse führt dann ohnehin auf
    einen 404, und ein Ausnahmefall in der Vorlage wäre nur Lärm.
    """
    try:
        return str(path.stat().st_mtime_ns)
    except OSError:
        return "0"


def tabellenadresse(
    status: str = "alle",
    q: str = "",
    kompakt: bool = False,
    offen: Iterable[int] = (),
) -> str:
    """Die Adresse einer Satztabelle: Filter, Klappzustand, offene Sätze.

    Steht hier und nicht in der Tabelle, weil auch die Handgriffe am Satzbau sie
    brauchen: sie antworten mit der ganzen Tabelle -- die Nummern verschieben
    sich -- und ohne diesen Zustand stünde danach die ungefilterte, ausgeklappte
    Liste da.
    """
    teile = [f"status={status}", f"q={quote(q)}"]
    if kompakt:
        teile.append("kompakt=1")
    nummern = sorted(dict.fromkeys(offen))
    if nummern:
        teile.append("offen=" + ",".join(str(i) for i in nummern))
    return "?" + "&".join(teile)


def anzahl(wert: int, eins: str, viele: str) -> str:
    """'1 Vergleich' statt '1 Vergleiche'."""
    return f"{wert} {eins if wert == 1 else viele}"


def dauer(sekunden: float) -> str:
    """Sekunden lesbar: '48 s', '6:12 min', '1:04:30 h'.

    Ein Kapitel dauert Minuten bis Stunden -- '3762.4s' beantwortet die Frage
    'wie lang ist das geworden?' nicht.
    """
    ganz = int(round(sekunden))
    if ganz < 60:
        return f"{ganz} s"
    stunden, rest = divmod(ganz, 3600)
    minuten, s = divmod(rest, 60)
    if stunden:
        return f"{stunden}:{minuten:02d}:{s:02d} h"
    return f"{minuten}:{s:02d} min"


def zeitpunkt(iso: str) -> str:
    """ISO-Zeitstempel lesbar machen, in der Zeitzone des Rechners.

    Gespeichert wird in UTC -- angezeigt gehört, was auf der Uhr im Raum stand.
    """
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d.%m.%Y, %H:%M")
    except ValueError:
        return iso


def get_app() -> FastAPI:
    """Einstiegspunkt für "uvicorn cloney.web.app:get_app --factory"."""
    return create_app()
