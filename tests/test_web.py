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
    # Gemessen wird gleich mit, der Satz bleibt also nicht ungeprüft stehen.
    assert after.status == ChunkStatus.OK


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


# -- Rückmeldung in der Oberfläche -----------------------------------------


def test_ruhende_statuskarte_hat_keinen_ausloeser(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Ein leeres hx-trigger ist kein 'kein Auslöser': htmx fällt dann auf
    seinen Standard zurück, und der ist bei einem div der Klick. Die Karte
    hätte sich bei jedem Klick in ihr selbst neu geladen und dabei das Ergebnis
    der eigentlichen Anfrage überschrieben."""
    client = _client(settings)
    project_id = _create_project(client)

    page = client.get(f"/projects/{project_id}").text
    assert 'hx-trigger=""' not in page
    assert "every 1500ms" not in page


def test_laufende_statuskarte_fragt_nach(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)

    karte = client.post(f"/projects/{project_id}/run").text
    assert "every 1500ms" in karte
    assert 'data-fertig="0"' in karte
    _wait_for_run(client, project_id)


def test_statuskarte_zeigt_fortschritt(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    assert "<progress" in client.get(f"/projects/{project_id}").text


def test_aktionen_zeigen_dass_sie_arbeiten(settings: Settings, voice_store: VoiceStore) -> None:
    """Ohne Anzeige sieht ein Klick, der eine halbe Minute lädt, aus wie einer,
    der nichts bewirkt hat."""
    client = _client(settings)
    project_id = _create_project(client)

    page = client.get(f"/projects/{project_id}").text
    assert "htmx-indicator" in page
    assert "hx-disabled-elt" in page


def test_projektseite_zeigt_die_referenz(settings: Settings, voice_store: VoiceStore) -> None:
    """Passt der Referenztext nicht zur Aufnahme, ist das die häufigste Ursache
    für unverständliche Ausgabe -- also gehört er sichtbar auf die Seite."""
    client = _client(settings)
    project_id = _create_project(client)

    page = client.get(f"/projects/{project_id}").text
    assert "Dies ist die Referenzaufnahme." in page
    assert "Zeichen/s" in page


def test_fehlende_stimme_wird_als_klartext_gemeldet(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Ein nacktes 'Internal Server Error' im Browser hilft niemandem -- der
    Grund muss mit."""
    import shutil

    client = _client(settings)
    project_id = _create_project(client)
    shutil.rmtree(settings.voices_dir / "test-stimme")

    response = client.post(f"/projects/{project_id}/chunks/0/reroll")
    assert response.status_code == 400
    assert "nicht mehr vorhanden" in response.json()["detail"]


def test_engine_fehler_wird_als_klartext_gemeldet(
    settings: Settings, voice_store: VoiceStore
) -> None:
    from cloney.engines.base import EngineError
    from cloney.web import app as web_app

    client = _client(settings)
    project_id = _create_project(client)

    def kaputt(_name: str, _settings: Settings, _options=None):  # noqa: ANN001, ANN202
        raise EngineError("Modell konnte nicht geladen werden: kein Speicher")

    original = web_app.create_engine
    web_app.create_engine = kaputt
    try:
        response = client.post(f"/projects/{project_id}/chunks/0/reroll")
    finally:
        web_app.create_engine = original

    assert response.status_code == 400
    assert "kein Speicher" in response.json()["detail"]


# -- Regler der Engine ------------------------------------------------------


def _create_project_with(client: TestClient, engine: str) -> str:
    response = client.post(
        "/projects",
        data={"name": "Regler", "text": TEXT, "voice": "test-stimme", "engine": engine},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[-1]


def test_regler_landen_im_manifest(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project_with(client, "f5-de")

    response = client.post(
        f"/projects/{project_id}/options",
        data={"speed": "0.85", "nfe_step": "48", "cfg_strength": "2.2"},
    )
    assert response.status_code == 200

    project = Project.load(settings.projects_dir / project_id)
    assert project.engine_options == {"speed": 0.85, "nfe_step": 48.0, "cfg_strength": 2.2}


def test_ausreisser_werden_gekappt(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project_with(client, "f5-de")

    client.post(f"/projects/{project_id}/options", data={"speed": "99", "unbekannt": "5"})
    project = Project.load(settings.projects_dir / project_id)
    assert project.engine_options["speed"] == 1.5
    assert "unbekannt" not in project.engine_options


def test_teilangabe_laesst_die_uebrigen_stehen(settings: Settings, voice_store: VoiceStore) -> None:
    """Sonst würfe ein einzeln verstellter Regler die anderen stillschweigend
    auf ihren Standard zurück."""
    client = _client(settings)
    project_id = _create_project_with(client, "f5-de")

    client.post(f"/projects/{project_id}/options", data={"speed": "0.8", "nfe_step": "48"})
    client.post(f"/projects/{project_id}/options", data={"speed": "0.9"})

    project = Project.load(settings.projects_dir / project_id)
    assert project.engine_options == {"speed": 0.9, "nfe_step": 48.0}


def test_engine_ohne_regler_zeigt_keine(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project_with(client, "dummy")
    seite = client.get(f"/projects/{project_id}").text
    assert 'type="range"' not in seite


def test_engine_mit_reglern_zeigt_sie(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project_with(client, "f5-de")
    seite = client.get(f"/projects/{project_id}").text
    assert 'name="speed"' in seite
    assert "Sprechtempo" in seite


def test_alles_neu_rendern_merkt_die_saetze_vor(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)
    assert Project.load(settings.projects_dir / project_id).is_complete

    client.post(f"/projects/{project_id}/rerender")
    project = Project.load(settings.projects_dir / project_id)
    assert all(c.status == ChunkStatus.PENDING for c in project.chunks)
    assert all(c.attempts == 1 for c in project.chunks)


# -- Verwalten: Projekte ----------------------------------------------------


def test_projekt_umbenennen(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)

    response = client.post(f"/projects/{project_id}/rename", data={"name": "Kapitel sieben"})
    assert response.status_code == 200
    assert Project.load(settings.projects_dir / project_id).name == "Kapitel sieben"


def test_leerer_name_wird_abgelehnt(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    assert client.post(f"/projects/{project_id}/rename", data={"name": "  "}).status_code == 400


def test_projekt_loeschen_entfernt_auch_den_ton(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)
    root = settings.projects_dir / project_id
    assert (root / "chunks").exists()

    response = client.post(f"/projects/{project_id}/delete", follow_redirects=False)
    assert response.status_code == 303
    assert not root.exists()


def test_projektkennung_darf_nicht_ausbrechen(settings: Settings) -> None:
    """Die Kennung kommt aus der URL. Ohne Prüfung ließe sich beim Löschen aus
    dem Datenverzeichnis herausgreifen."""
    assert _client(settings).post("/projects/..%2F..%2Fetc/delete").status_code in (400, 404)


def test_kopie_uebernimmt_regler_aber_keinen_ton(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project_with(client, "f5-de")
    client.post(f"/projects/{project_id}/options", data={"speed": "0.8"})

    response = client.post(f"/projects/{project_id}/duplicate", follow_redirects=False)
    assert response.status_code == 303
    kopie_id = response.headers["location"].rsplit("/", 1)[-1]

    kopie = Project.load(settings.projects_dir / kopie_id)
    original = Project.load(settings.projects_dir / project_id)
    assert kopie.id != original.id
    assert kopie.engine_options == {"speed": 0.8}
    assert kopie.source_text == original.source_text
    assert all(c.status == ChunkStatus.PENDING for c in kopie.chunks)


# -- Verwalten: erzeugter Ton -----------------------------------------------


def test_ton_eines_satzes_verwerfen_behaelt_den_seed(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Anders als das Neuwürfeln: nur so lässt sich die Wirkung geänderter
    Regler an einem Satz beurteilen, ohne dass zugleich der Zufall wechselt."""
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    vorher = Project.load(settings.projects_dir / project_id).chunks[0]
    assert vorher.audio_file

    client.post(f"/projects/{project_id}/chunks/0/discard")
    nachher = Project.load(settings.projects_dir / project_id).chunks[0]

    assert nachher.seed == vorher.seed
    assert nachher.attempts == vorher.attempts
    assert nachher.audio_file is None
    assert nachher.status == ChunkStatus.PENDING
    assert not (settings.projects_dir / project_id / "chunks" / "chunk_0000.wav").exists()


def test_ton_aller_saetze_verwerfen(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    client.post(f"/projects/{project_id}/discard")
    project = Project.load(settings.projects_dir / project_id)
    assert all(c.audio_file is None for c in project.chunks)
    assert all(
        c.seed == Project.load(settings.projects_dir / project_id).chunks[c.index].seed
        for c in project.chunks
    )
    assert project.output_file is None
    assert not project.output_path.exists()


# -- Verwalten: Stimmen -----------------------------------------------------


def test_wortlaut_aendern_prueft_neu(settings: Settings, voice_store: VoiceStore) -> None:
    """Das Sprechtempo ergibt sich aus Wortlaut und Dauer -- ein geänderter Text
    ändert den Befund, ohne dass die Aufnahme angefasst wurde."""
    client = _client(settings)

    response = client.post("/voices/test-stimme/transcript", data={"transcript": "kurz"})
    assert response.status_code == 200
    assert "Beschriftung" in response.text
    assert VoiceStore(settings.voices_dir).get("test-stimme").transcript == "kurz"


def test_stimme_in_benutzung_wird_nicht_geloescht(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Sonst blieben Projekte mit einem Verweis ins Leere zurück."""
    client = _client(settings)
    _create_project(client)

    response = client.post("/voices/test-stimme/delete")
    assert response.status_code == 409
    assert "wird noch verwendet" in response.json()["detail"]
    assert VoiceStore(settings.voices_dir).exists("test-stimme")


def test_unbenutzte_stimme_wird_geloescht(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    response = client.post("/voices/test-stimme/delete", follow_redirects=False)
    assert response.status_code == 303
    assert not VoiceStore(settings.voices_dir).exists("test-stimme")


def test_referenzaufnahme_ist_anhoerbar(settings: Settings, voice_store: VoiceStore) -> None:
    """Ohne sie zu hören lässt sich nicht beurteilen, ob der Wortlaut stimmt."""
    response = _client(settings).get("/voices/test-stimme/audio")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_einzelnes_neurendern_misst_auch(settings: Settings, voice_store: VoiceStore) -> None:
    """Sonst bliebe der Satz ohne Fehlerrate stehen und der Referenz-Vorspann
    ungeschnitten -- das Abhören einer Reglerstellung ergäbe also ein anderes
    Ergebnis als der vollständige Lauf."""
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    client.post(f"/projects/{project_id}/chunks/0/reroll")
    chunk = Project.load(settings.projects_dir / project_id).chunks[0]
    assert chunk.cer is not None
    assert chunk.status == ChunkStatus.OK


def test_ohne_spracherkennung_bleibt_der_ton_trotzdem(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Ohne ASR gibt es keine Messung -- der erzeugte Ton ist deswegen aber
    nicht verloren."""
    client = TestClient(create_app(settings, asr_factory=None))
    project_id = _create_project(client)

    response = client.post(f"/projects/{project_id}/chunks/0/reroll")
    assert response.status_code == 200
    chunk = Project.load(settings.projects_dir / project_id).chunks[0]
    assert chunk.audio_file is not None
