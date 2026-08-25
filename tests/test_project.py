from __future__ import annotations

from cloney.config import Settings
from cloney.core.project import ChunkStatus, Project, derive_seed

TEXT = "Erster Satz hier.\n\nZweiter Absatz mit Inhalt. Und noch ein Satz dazu."


def _create(settings: Settings, target_seconds: float = 3.0) -> Project:
    return Project.create(
        name="Testprojekt",
        text=TEXT,
        voice="test-stimme",
        engine="dummy",
        sample_rate=24000,
        projects_dir=settings.projects_dir,
        target_seconds=target_seconds,
    )


def test_manifest_roundtrip(settings: Settings) -> None:
    project = _create(settings)
    loaded = Project.load(project.root)
    assert loaded.id == project.id
    assert [c.normalized_text for c in loaded.chunks] == [
        c.normalized_text for c in project.chunks
    ]
    assert loaded.root == project.root


def test_seeds_sind_reproduzierbar() -> None:
    assert derive_seed("abc", 3, 0) == derive_seed("abc", 3, 0)
    assert derive_seed("abc", 3, 0) != derive_seed("abc", 3, 1)
    assert derive_seed("abc", 3, 0) != derive_seed("abc", 4, 0)


def test_reroll_setzt_chunk_zurueck(settings: Settings) -> None:
    project = _create(settings)
    chunk = project.chunks[0]
    chunk.status = ChunkStatus.OK
    chunk.cer = 0.42
    old_seed = chunk.seed

    rerolled = project.reroll(0)
    assert rerolled.seed != old_seed
    assert rerolled.status == ChunkStatus.PENDING
    assert rerolled.cer is None
    assert rerolled.attempts == 1


def test_retext_normalisiert_neu(settings: Settings) -> None:
    project = _create(settings)
    project.retext(0, "Am 3. Mai 2024.")
    assert "dritten Mai" in project.chunks[0].normalized_text
    assert project.chunks[0].status == ChunkStatus.PENDING


def test_fortschritt_und_median(settings: Settings) -> None:
    project = _create(settings)
    for chunk in project.chunks:
        chunk.status = ChunkStatus.OK
        chunk.cer = 0.05
    assert project.is_complete
    assert project.progress == (len(project.chunks), len(project.chunks))
    assert project.median_cer() == 0.05


def test_liste_ist_nach_datum_sortiert(settings: Settings) -> None:
    _create(settings)
    assert len(Project.list_all(settings.projects_dir)) == 1
