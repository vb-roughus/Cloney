"""Orchestrierung eines Renderlaufs in Phasen.

Die Phasentrennung ist keine Kosmetik, sondern die Antwort auf die
VRAM-Beschränkung: pro Phase liegt genau ein Modell im Speicher, und zwar für
das gesamte Skript, nicht pro Chunk. Zwischen den Phasen wird freigegeben.

    SYNTH     TTS lädt -> alle offenen Chunks -> TTS entlädt
    QC        ASR lädt -> Rückschrift und CER für alle -> ASR entlädt
    RETRY     auffällige Chunks mit neuem Seed, solange Versuche übrig sind
    ASSEMBLE  reine CPU-Arbeit

Alle Modelle werden über Fabriken hereingereicht. Dadurch läuft dieselbe
Pipeline in Tests gegen DummyEngine und DummyASR -- ohne GPU, ohne Netz.

``run_comparison`` setzt darauf auf: es rendert dieselbe Textprobe je
Reglerstellung einmal durch genau diesen Weg, damit der Vergleich misst, was im
echten Lauf auch geschieht.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cloney.asr.base import ASREngine
from cloney.config import Settings
from cloney.core.audio import Segment, assemble, read_wav, write_wav
from cloney.core.bleed import cut_point, find_content_start
from cloney.core.compare import Comparison, Variant, VariantStatus
from cloney.core.metrics import cer, cosine_similarity
from cloney.core.project import Chunk, ChunkStatus, Project
from cloney.core.voices import VoiceStore
from cloney.engines.base import EngineInfo, TTSEngine, VoiceRef, strip_unsupported_tags
from cloney.speaker.base import SpeakerEmbedder
from cloney.vram import model_slot

EngineFactory = Callable[[], TTSEngine]
ASRFactory = Callable[[], ASREngine]
EmbedderFactory = Callable[[], SpeakerEmbedder]


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    message: str
    done: int = 0
    total: int = 0


ProgressCallback = Callable[[ProgressEvent], None]


def _noop(event: ProgressEvent) -> None:
    return None


def _referenzen(
    voice_store: VoiceStore, project: Project, chunks: list[Chunk]
) -> dict[str, VoiceRef]:
    """Die gebrauchten Referenzaufnahmen, eine je vorkommender Lage.

    Über das Projekt und nicht über den Satz: ein Satz trägt nur, was von der
    Vorgabe des Projekts abweicht, und kann die Frage allein nicht beantworten.
    """
    lagen = {project.lage_of(c) for c in chunks}
    return {lage: voice_store.get(project.voice, lage) for lage in lagen}


def synthesize_chunks(
    project: Project,
    chunks: list[Chunk],
    voice_store: VoiceStore,
    engine_factory: EngineFactory,
    on_event: ProgressCallback = _noop,
) -> None:
    """Phase SYNTH. Jeder Chunk wird gegen eine unveränderte Referenz konditioniert.

    Genau darin liegt die Drift-Vermeidung: konditioniert wird nie gegen das
    Ergebnis des Vorgängers, sondern immer gegen eine feststehende Aufnahme.
    Welche das ist, entscheidet die Lage des Satzes -- sie steht wie der Seed im
    Manifest, und beide zusammen machen jeden Chunk einzeln und identisch neu
    erzeugbar.

    Die Aufnahmen werden dabei einmal je Lage nachgeschlagen und nicht je Satz:
    ein Kapitel hat hundert Sätze und drei Lagen.
    """
    if not chunks:
        return

    referenzen = _referenzen(voice_store, project, chunks)
    with model_slot(engine_factory) as engine:
        on_event(ProgressEvent("synth", f"Engine '{engine.info.name}' geladen", 0, len(chunks)))
        for done, chunk in enumerate(chunks, start=1):
            text = strip_unsupported_tags(chunk.normalized_text, engine.info.supported_tags)
            voice = referenzen[project.lage_of(chunk)]
            try:
                audio = engine.synthesize(text, voice, chunk.seed)
            except Exception as exc:  # noqa: BLE001 - Fehler gehört ins Manifest, nicht in den Stack
                chunk.status = ChunkStatus.FAILED
                chunk.error = str(exc)[:500]
                project.save()
                on_event(ProgressEvent("synth", f"Chunk {chunk.index}: {exc}", done, len(chunks)))
                continue

            write_wav(project.chunk_path(chunk.index), audio, engine.info.sample_rate)
            chunk.audio_file = project.chunk_path(chunk.index).name
            chunk.engine = engine.info.name
            chunk.status = ChunkStatus.SYNTHESIZED
            chunk.error = None
            project.sample_rate = engine.info.sample_rate
            project.save()
            on_event(ProgressEvent("synth", f"Chunk {chunk.index} erzeugt", done, len(chunks)))


def quality_check(
    project: Project,
    settings: Settings,
    asr_factory: ASRFactory | None,
    on_event: ProgressCallback = _noop,
) -> None:
    """Phase QC. Rückschrift jedes Chunks, Vergleich mit der Sprechfassung.

    Ohne ASR entfällt die Messung -- die Chunks gelten dann als in Ordnung, und
    das Manifest sagt durch ``cer = None`` ehrlich, dass nicht geprüft wurde.
    """
    pending = project.pending_qc()
    if not pending:
        return

    if asr_factory is None:
        for chunk in pending:
            chunk.status = ChunkStatus.OK
        project.save()
        on_event(ProgressEvent("qc", "Ohne ASR -- keine Qualitätsmessung", 0, 0))
        return

    with model_slot(asr_factory) as asr:
        on_event(ProgressEvent("qc", "Spracherkennung geladen", 0, len(pending)))
        for done, chunk in enumerate(pending, start=1):
            audio, sample_rate = read_wav(project.chunk_path(chunk.index))
            transcript = asr.transcribe(audio, sample_rate, settings.asr_language)
            spoken = strip_unsupported_tags(chunk.normalized_text, frozenset())

            hypothesis = transcript.text
            if settings.trim_reference_bleed:
                hypothesis = _trim_reference_bleed(
                    project, chunk, audio, sample_rate, transcript, spoken, settings
                )
            score = cer(spoken, hypothesis)

            chunk.asr_text = hypothesis
            chunk.cer = round(score, 4)
            chunk.status = (
                ChunkStatus.OK if score <= settings.cer_threshold else ChunkStatus.NEEDS_REVIEW
            )
            project.save()
            on_event(
                ProgressEvent("qc", f"Chunk {chunk.index}: CER {score:.3f}", done, len(pending))
            )


def _trim_reference_bleed(
    project: Project,
    chunk: Chunk,
    audio,  # noqa: ANN001 - np.ndarray, ohne Import in der Signatur
    sample_rate: int,
    transcript,  # noqa: ANN001 - Transcript
    spoken: str,
    settings: Settings,
) -> str:
    """Schneidet ein am Anfang stehen gebliebenes Stück der Referenz weg.

    F5-TTS erzeugt Referenz und neuen Text am Stück und trennt sie an einer
    berechneten Stelle. Weicht die Aufnahme von ihrem Wortlaut ab, rutscht ein
    Rest der Referenz hinter diese Stelle. Nach Lautstärke ist er nicht zu
    fassen -- er ist Sprache. Über die Rückschrift schon: sie sagt, ab welchem
    Wort der gewünschte Text beginnt, und wann dieses Wort erklingt.

    Erkannt wird nur, was sich sicher zuordnen lässt; im Zweifel bleibt das
    Audio unangetastet.

    Der gemeldete Wortanfang ist der Kandidat, nicht der Schnitt. Wo genau
    getrennt wird, entscheidet ``cut_point`` an der Wellenform: Whispers Zeiten
    sind geschätzt, und ein paar Hundertstel zu spät kosten dem ersten Wort
    seinen Anlaut.
    """
    start, vorspann_woerter = find_content_start(transcript.words, spoken)
    if start is None or start < settings.min_bleed_seconds:
        return transcript.text

    schnitt = cut_point(audio, sample_rate, start)
    ab = int(schnitt * sample_rate)
    if ab <= 0 or ab >= len(audio):
        return transcript.text

    write_wav(project.chunk_path(chunk.index), audio[ab:], sample_rate)
    chunk.trimmed_bleed_s = round(schnitt, 3)
    # Verglichen wird gegen die Rückschrift ohne den Vorspann -- sonst zählte
    # der eben entfernte Teil als Fehler.
    return " ".join(w.text for w in transcript.words[vorspann_woerter:])


def check_speaker_similarity(
    project: Project,
    settings: Settings,
    voice_store: VoiceStore,
    embedder_factory: EmbedderFactory | None,
    on_event: ProgressCallback = _noop,
) -> None:
    """Phase SIMILARITY. Klingt das Ergebnis noch nach der Referenzstimme?

    Die Fehlerrate prüft die Wörter, nicht die Stimme -- ein Chunk kann
    fehlerfrei sein und nach jemand anderem klingen. Verglichen werden deshalb
    die Stimmeinbettungen von Referenz und Ergebnis.

    Markiert wird nur, wenn eine Schwelle gesetzt ist. Welchen Wert ein guter
    Klon erreicht, hängt an Modell und Aufnahme; ohne eigene Messung wäre jede
    Vorgabe geraten und erzeugte Fehlalarme.

    **Diese Phase darf niemals einen Lauf zum Scheitern bringen.** Sie ist eine
    Messung an fertigem Ton, keine Voraussetzung für ihn. Fehlt das Paket, ist
    das Modell nicht ladbar oder die Referenz verschwunden, dann wird der
    Vergleich übersprungen und der Grund im Manifest vermerkt -- der Zusammenbau
    läuft weiter. Alles andere hieße: eine freiwillige Zusatzzahl vernichtet ein
    fertig gerendertes Kapitel.
    """
    if embedder_factory is None:
        return
    zu_pruefen = [c for c in project.chunks if project.chunk_path(c.index).exists()]
    if not zu_pruefen:
        return

    try:
        _measure_similarity(project, settings, voice_store, embedder_factory, zu_pruefen, on_event)
    except Exception as exc:  # noqa: BLE001 - der Grund gehört ins Manifest, nicht in den Abbruch
        project.similarity_note = str(exc)[:300]
        project.save()
        on_event(ProgressEvent("similarity", f"Stimmvergleich übersprungen: {exc}"))


def _measure_similarity(
    project: Project,
    settings: Settings,
    voice_store: VoiceStore,
    embedder_factory: EmbedderFactory,
    zu_pruefen: list[Chunk],
    on_event: ProgressCallback,
) -> None:
    referenzen = _referenzen(voice_store, project, zu_pruefen)
    with model_slot(embedder_factory) as embedder:
        on_event(ProgressEvent("similarity", "Stimmvergleich geladen", 0, len(zu_pruefen)))
        # Verglichen wird mit der Aufnahme, gegen die auch konditioniert wurde.
        # Gegen die neutrale gemessen, geriete jeder Satz in einer anderen Lage
        # unter Verdacht -- und zwar gerade dann, wenn die Lage gut sitzt.
        vektoren = {
            lage: embedder.embed(*read_wav(ref.audio_path)) for lage, ref in referenzen.items()
        }
        project.similarity_note = None

        for done, chunk in enumerate(zu_pruefen, start=1):
            audio, sample_rate = read_wav(project.chunk_path(chunk.index))
            wert = cosine_similarity(
                vektoren[project.lage_of(chunk)], embedder.embed(audio, sample_rate)
            )
            chunk.speaker_similarity = round(wert, 4)
            if settings.similarity_threshold > 0 and wert < settings.similarity_threshold:
                chunk.status = ChunkStatus.NEEDS_REVIEW
            project.save()
            on_event(
                ProgressEvent(
                    "similarity",
                    f"Chunk {chunk.index}: Ähnlichkeit {wert:.2f}",
                    done,
                    len(zu_pruefen),
                )
            )


def assemble_output(
    project: Project,
    settings: Settings,
    on_event: ProgressCallback = _noop,
) -> None:
    """Phase ASSEMBLE. Alle vorhandenen Chunks zur fertigen Spur."""
    segments = []
    raten: set[int] = set()
    for chunk in project.chunks:
        path = project.chunk_path(chunk.index)
        if not path.exists():
            continue
        audio, rate = read_wav(path)
        raten.add(rate)
        segments.append(Segment(audio, chunk.ends_paragraph, chunk.is_heading))

    if not segments:
        on_event(ProgressEvent("assemble", "Nichts zusammenzubauen -- kein Chunk erzeugt"))
        return

    # Maßgeblich ist, was in den Dateien steht, nicht was beim Anlegen des
    # Projekts angenommen wurde. Eine Engine hinter einem Server kann eine
    # andere Rate liefern als ihre EngineInfo verspricht; würde hier die
    # Annahme gelten, liefe die fertige Spur zu langsam oder zu schnell.
    if len(raten) > 1:
        on_event(
            ProgressEvent(
                "assemble",
                f"Chunks haben verschiedene Abtastraten ({sorted(raten)}) -- "
                "die betroffenen Sätze neu rendern",
            )
        )
        return
    sample_rate = raten.pop()
    if sample_rate != project.sample_rate:
        project.sample_rate = sample_rate
        project.save()

    track = assemble(
        segments,
        sample_rate,
        target_lufs=settings.target_lufs,
        pause_sentence_ms=settings.pause_sentence_ms,
        pause_paragraph_ms=settings.pause_paragraph_ms,
        pause_heading_ms=settings.pause_heading_ms,
        edge_fade_ms=settings.edge_fade_ms,
        trim_threshold_db=settings.trim_threshold_db,
    )
    write_wav(project.output_path, track, sample_rate)
    project.output_file = project.output_path.name
    project.save()
    on_event(
        ProgressEvent(
            "assemble",
            f"{len(segments)} Chunks, {len(track) / sample_rate:.1f}s geschrieben",
        )
    )


def run_project(
    project: Project,
    settings: Settings,
    voice_store: VoiceStore,
    engine_factory: EngineFactory,
    asr_factory: ASRFactory | None = None,
    on_event: ProgressCallback = _noop,
    embedder_factory: EmbedderFactory | None = None,
) -> Project:
    """Führt den Lauf bis zum fertigen WAV. Bereits erledigte Chunks bleiben unberührt."""
    for attempt in range(settings.max_retries + 1):
        todo = project.pending_synthesis()
        if not todo:
            break

        phase = "synth" if attempt == 0 else "retry"
        on_event(ProgressEvent(phase, f"{len(todo)} Chunks zu erzeugen", 0, len(todo)))
        synthesize_chunks(project, todo, voice_store, engine_factory, on_event)
        quality_check(project, settings, asr_factory, on_event)

        flagged = [c for c in project.chunks if c.status == ChunkStatus.NEEDS_REVIEW]
        if not flagged:
            break
        if attempt == settings.max_retries:
            # Versuche aufgebraucht. Die Chunks bleiben markiert, statt still
            # in die fertige Spur zu rutschen -- die Entscheidung gehört dem Menschen.
            on_event(
                ProgressEvent("retry", f"{len(flagged)} Chunks bleiben zur Durchsicht markiert")
            )
            break

        for chunk in flagged:
            project.reroll(chunk.index)
        project.save()
        on_event(ProgressEvent("retry", f"{len(flagged)} Chunks werden neu gewürfelt"))

    if settings.check_speaker_similarity:
        check_speaker_similarity(project, settings, voice_store, embedder_factory, on_event)

    assemble_output(project, settings, on_event)
    return project


def run_comparison(
    comparison: Comparison,
    settings: Settings,
    voice_store: VoiceStore,
    engine_factory: Callable[[dict[str, float], str], TTSEngine],
    asr_factory: ASRFactory | None = None,
    on_event: ProgressCallback = _noop,
    embedder_factory: EmbedderFactory | None = None,
) -> Comparison:
    """Rendert eine Textprobe je Reglerstellung einmal und misst jedes Ergebnis.

    Jede Variante durchläuft dieselbe Pipeline wie ein Hörbuch -- gemessen wird
    also, was später auch passiert. Zwei Abweichungen sind Absicht:

    * **Keine Wiederholungsversuche.** Ein auffälliger Satz würde sonst mit
      neuem Seed nachgerendert, und die Varianten unterschieden sich nicht mehr
      allein in der Reglerstellung.
    * **Eine gescheiterte Variante bricht den Vergleich nicht ab.** Sie bekommt
      ihre Fehlermeldung in die Zeile; die übrigen Zahlen bleiben brauchbar.
    """
    messsettings = settings.model_copy(update={"max_retries": 0})
    info = _engine_info(comparison.engine)
    # Nach der längsten Lage geschnitten: jede Variante kann eine andere wählen,
    # und an der neutralen bemessen wäre der Schnitt für eine längere zu
    # großzügig -- das Modell teilte dann selbst.
    referenzlaenge = voice_store.longest_reference_seconds(comparison.voice)
    gesamt = len(comparison.variants)

    for nummer, variant in enumerate(comparison.variants, start=1):
        if variant.is_done:
            continue
        variant.status = VariantStatus.RUNNING
        comparison.save()
        on_event(ProgressEvent("compare", f"{variant.label}", nummer - 1, gesamt))

        try:
            project = comparison.prepare(variant.slug, info, referenzlaenge)
            run_project(
                project,
                messsettings,
                voice_store,
                lambda opts=dict(variant.options), modell=variant.model: engine_factory(
                    opts, modell
                ),
                asr_factory,
                on_event,
                embedder_factory,
            )
            gemessen = comparison.record(variant.slug, project)
            on_event(
                ProgressEvent(
                    "compare",
                    f"{variant.label}: {_kurzfassung(gemessen)}",
                    nummer,
                    gesamt,
                )
            )
        except Exception as exc:  # noqa: BLE001 - der Fehler gehört in die Zeile, nicht ins Log
            comparison.fail(variant.slug, str(exc))
            on_event(ProgressEvent("compare", f"{variant.label}: {exc}", nummer, gesamt))

    return comparison


def _kurzfassung(variant: Variant) -> str:
    teile = []
    if variant.median_cer is not None:
        teile.append(f"CER {variant.median_cer:.1%}")
    if variant.median_similarity is not None:
        teile.append(f"Stimme {variant.median_similarity:.2f}")
    if variant.duration_s is not None:
        teile.append(f"{variant.duration_s:.1f}s")
    return ", ".join(teile) or "gerendert"


def _engine_info(name: str) -> EngineInfo:
    from cloney.engines.registry import engine_info

    return engine_info(name)
