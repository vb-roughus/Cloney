"""Routen-Tests gegen die ASGI-App -- ohne Server, ohne Browser, ohne GPU."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from cloney.asr.dummy import DummyASR
from cloney.config import Settings
from cloney.core.project import ChunkStatus, Project
from cloney.core.voices import VoiceStore
from cloney.web.app import create_app

TEXT = "Am 3. Mai 2024 begann es.\n\nDr. Meier sagte z.B. nichts dazu. Dann war Ruhe."


def _client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, DummyASR))


def _create_project(client: TestClient) -> str:
    response = client.post(
        "/projects",
        data={"name": "Testlauf", "text": TEXT, "voice": "test-stimme", "engine": "dummy"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[-1]


def _wait_for_run(client: TestClient, project_id: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if "läuft" not in client.get(f"/projects/{project_id}/status").text:
            return
        time.sleep(0.1)
    raise AssertionError("Renderlauf wurde nicht fertig")


def test_startseite_zeigt_stimmen_und_engines(settings: Settings, voice_store: VoiceStore) -> None:
    response = _client(settings).get("/")
    assert response.status_code == 200
    assert "test-stimme" in response.text
    # Die Lizenz der Gewichte steht in der Oberfläche, nicht nur in der Doku.
    assert "Research &amp; Non-Commercial" in response.text


def test_startseite_ohne_stimme_weist_den_weg(settings: Settings) -> None:
    response = _client(settings).get("/")
    assert "Zuerst eine Stimme anlegen" in response.text


def test_projekt_anlegen_und_rendern(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)

    page = client.get(f"/projects/{project_id}")
    assert page.status_code == 200
    assert "dritten Mai" in page.text  # normalisierte Sprechfassung ist sichtbar

    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    project = Project.load(settings.projects_dir / project_id)
    assert project.is_complete
    assert client.get(f"/projects/{project_id}/output").status_code == 200
    assert client.get(f"/projects/{project_id}/chunks/0/audio").status_code == 200


def test_chunk_neu_wuerfeln_aendert_den_seed(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    before = Project.load(settings.projects_dir / project_id).chunks[0].seed
    row = client.post(f"/projects/{project_id}/chunks/0/reroll")
    assert row.status_code == 200
    assert 'id="chunk-0"' in row.text

    after = Project.load(settings.projects_dir / project_id).chunks[0]
    assert after.seed != before
    assert after.status == ChunkStatus.SYNTHESIZED


def test_chunk_text_aendern_normalisiert_neu(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)

    row = client.post(f"/projects/{project_id}/chunks/0/text", data={"raw_text": "Am 7. Juli."})
    assert row.status_code == 200
    assert "siebten Juli" in row.text


def test_markierten_chunk_durchwinken(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    project = Project.load(settings.projects_dir / project_id)
    project.chunks[0].status = ChunkStatus.NEEDS_REVIEW
    project.save()

    client.post(f"/projects/{project_id}/chunks/0/accept")
    assert Project.load(project.root).chunks[0].status == ChunkStatus.OK


def test_stimme_ueber_die_oberflaeche_anlegen(settings: Settings, reference_wav) -> None:  # noqa: ANN001
    client = _client(settings)
    response = client.post(
        "/voices",
        data={"name": "neue-stimme", "transcript": "Wortlaut der Aufnahme."},
        files={"audio": ("referenz.wav", reference_wav.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert "neue-stimme" in response.text
    assert VoiceStore(settings.voices_dir).exists("neue-stimme")


def test_kurze_referenz_wird_bemaengelt(settings: Settings, tmp_path) -> None:  # noqa: ANN001
    """Die Eingangsprüfung muss warnen, bevor ein ganzes Kapitel gerendert ist."""
    import numpy as np

    from cloney.core.audio import write_wav

    short = tmp_path / "kurz.wav"
    t = np.arange(24000, dtype=np.float32) / 24000
    write_wav(short, (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), 24000)

    response = _client(settings).post(
        "/voices",
        data={"name": "zu-kurz", "transcript": ""},
        files={"audio": ("kurz.wav", short.read_bytes(), "audio/wav")},
    )
    assert "wird der Klon instabil" in response.text


def test_unbekanntes_projekt_gibt_404(settings: Settings) -> None:
    assert _client(settings).get("/projects/gibtsnicht").status_code == 404


def test_leerer_text_wird_abgelehnt(settings: Settings, voice_store: VoiceStore) -> None:
    response = _client(settings).post(
        "/projects",
        data={"name": "Leer", "text": "   ", "voice": "test-stimme", "engine": "dummy"},
    )
    assert response.status_code == 400
