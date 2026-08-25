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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cloney.asr.base import ASREngine
from cloney.config import Settings
from cloney.core.audio import assemble, read_wav, write_wav
from cloney.core.metrics import cer
from cloney.core.project import Chunk, ChunkStatus, Project
from cloney.core.voices import VoiceStore
from cloney.engines.base import TTSEngine, strip_unsupported_tags
from cloney.vram import model_slot

EngineFactory = Callable[[], TTSEngine]
ASRFactory = Callable[[], ASREngine]


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    message: str
    done: int = 0
    total: int = 0


ProgressCallback = Callable[[ProgressEvent], None]


def _noop(event: ProgressEvent) -> None:
    return None


def synthesize_chunks(
    project: Project,
    chunks: list[Chunk],
    voice_store: VoiceStore,
    engine_factory: EngineFactory,
    on_event: ProgressCallback = _noop,
) -> None:
    """Phase SYNTH. Jeder Chunk wird gegen dieselbe Referenz konditioniert.

    Genau darin liegt die Drift-Vermeidung: die Referenz ändert sich über den
    gesamten Lauf nicht, und der Seed steht im Manifest -- ein Chunk lässt sich
    dadurch jederzeit einzeln und identisch neu erzeugen.
    """
    if not chunks:
        return

    voice = voice_store.get(project.voice)
    with model_slot(engine_factory) as engine:
        on_event(ProgressEvent("synth", f"Engine '{engine.info.name}' geladen", 0, len(chunks)))
        for done, chunk in enumerate(chunks, start=1):
            text = strip_unsupported_tags(chunk.normalized_text, engine.info.supported_tags)
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
            hypothesis = asr.transcribe(audio, sample_rate, settings.asr_language)
            spoken = strip_unsupported_tags(chunk.normalized_text, frozenset())
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


def assemble_output(
    project: Project,
    settings: Settings,
    on_event: ProgressCallback = _noop,
) -> None:
    """Phase ASSEMBLE. Alle vorhandenen Chunks zur fertigen Spur."""
    segments = []
    for chunk in project.chunks:
        path = project.chunk_path(chunk.index)
        if not path.exists():
            continue
        audio, _ = read_wav(path)
        segments.append((audio, chunk.ends_paragraph))

    if not segments:
        on_event(ProgressEvent("assemble", "Nichts zusammenzubauen -- kein Chunk erzeugt"))
        return

    track = assemble(
        segments,
        project.sample_rate,
        target_lufs=settings.target_lufs,
        pause_sentence_ms=settings.pause_sentence_ms,
        pause_paragraph_ms=settings.pause_paragraph_ms,
        edge_fade_ms=settings.edge_fade_ms,
        trim_threshold_db=settings.trim_threshold_db,
    )
    write_wav(project.output_path, track, project.sample_rate)
    project.output_file = project.output_path.name
    project.save()
    on_event(
        ProgressEvent(
            "assemble",
            f"{len(segments)} Chunks, {len(track) / project.sample_rate:.1f}s geschrieben",
        )
    )


def run_project(
    project: Project,
    settings: Settings,
    voice_store: VoiceStore,
    engine_factory: EngineFactory,
    asr_factory: ASRFactory | None = None,
    on_event: ProgressCallback = _noop,
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

    assemble_output(project, settings, on_event)
    return project
