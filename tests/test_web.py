"""Routen-Tests gegen die ASGI-App -- ohne Server, ohne Browser, ohne GPU."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloney.asr.dummy import DummyASR
from cloney.config import Settings
from cloney.core.audio import duration_seconds, read_wav
from cloney.core.compare import Comparison
from cloney.core.models import ModelStore
from cloney.core.project import ChunkStatus, Project
from cloney.core.voices import VoiceStore
from cloney.engines.dummy import DummyEngine
from cloney.web.app import anzahl, create_app, zeitpunkt

TEXT = "Am 3. Mai 2024 begann es.\n\nDr. Meier sagte z.B. nichts dazu. Dann war Ruhe."


def _client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, DummyASR))


def _create_project(client: TestClient, **werte: str) -> str:
    daten = {"name": "Testlauf", "text": TEXT, "voice": "test-stimme", "engine": "dummy"}
    daten.update(werte)
    response = client.post("/projects", data=daten, follow_redirects=False)
    assert response.status_code == 303, response.text
    return response.headers["location"].rsplit("/", 1)[-1]


def _wait_for_run(client: TestClient, project_id: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if "läuft" not in client.get(f"/projects/{project_id}/status").text:
            return
        time.sleep(0.1)
    raise AssertionError("Renderlauf wurde nicht fertig")


def test_startseite_zaehlt_den_bestand(settings: Settings, voice_store: VoiceStore) -> None:
    """Die Startseite beantwortet 'was steht an?', nicht 'was kann ich anlegen?'."""
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    seite = client.get("/").text
    assert "Übersicht" in seite
    assert "Sätze gerendert" in seite
    assert "Testlauf" in seite
    # Das Anlegen hat die Startseite verlassen -- es steht hinter einem Knopf.
    assert 'action="/projects"' not in seite


def test_startseite_ohne_alles_zeigt_den_einstieg(settings: Settings) -> None:
    seite = _client(settings).get("/").text
    assert "Eine Referenzstimme anlegen" in seite


def test_startseite_ohne_stimme_weist_den_weg(settings: Settings, voice_store: VoiceStore) -> None:
    """Eine Stimme ohne Projekt: die Übersicht steht, der fehlende Schritt auch."""
    client = _client(settings)
    _create_project(client)
    voice_store.delete("test-stimme")
    assert "Zuerst eine Stimme anlegen" in client.get("/").text


def test_zeitstempel_wird_lesbar(settings: Settings, voice_store: VoiceStore) -> None:
    """Gespeichert wird in UTC. In der Liste steht, was auf der Uhr im Raum stand."""
    client = _client(settings)
    _create_project(client)

    assert "+00:00" not in client.get("/projects").text
    # Keine feste Uhrzeit erwarten: umgerechnet wird in die Zone des Rechners,
    # und die ist hier eine andere als in der CI.
    assert re.fullmatch(r"\d\d\.\d\d\.\d{4}, \d\d:\d\d", zeitpunkt("2026-08-30T08:47:01+00:00"))
    # Ein unlesbarer Wert bleibt stehen, statt die Seite zu sprengen.
    assert zeitpunkt("später") == "später"


def test_einer_bekommt_die_einzahl() -> None:
    assert anzahl(1, "Vergleich", "Vergleiche") == "1 Vergleich"
    assert anzahl(0, "Vergleich", "Vergleiche") == "0 Vergleiche"


def test_anlegen_hat_eine_eigene_seite(settings: Settings, voice_store: VoiceStore) -> None:
    """'/projects/new' darf nicht als Projektkennung gelesen werden -- sonst
    stünde dort 404 statt des Formulars."""
    client = _client(settings)

    liste = client.get("/projects")
    assert liste.status_code == 200
    assert 'href="/projects/new"' in liste.text
    assert "<textarea" not in liste.text

    formular = client.get("/projects/new")
    assert formular.status_code == 200
    assert 'action="/projects"' in formular.text
    # Die Lizenz der Gewichte steht in der Oberfläche, nicht nur in der Doku.
    assert "Research &amp; Non-Commercial" in formular.text


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


def test_engine_ohne_regler_zeigt_keine(
    settings: Settings, voice_store: VoiceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alle mitgelieferten Engines bieten Regler an -- eine künftige muss es nicht.

    Deshalb wird für diesen Fall eine reglerlose Engine untergeschoben, statt
    ihn an einer Engine festzumachen, die ihn morgen nicht mehr abbildet.
    """
    from cloney.engines import registry

    schlicht = replace(DummyEngine.info, options=())
    monkeypatch.setitem(registry._INFOS, "dummy", schlicht)

    client = _client(settings)
    project_id = _create_project_with(client, "dummy")
    seite = client.get(f"/projects/{project_id}").text
    assert 'type="range"' not in seite


def test_regler_der_dummy_engine_erreichen_die_engine(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Der Weg einer Reglerstellung: Formular -> Manifest -> Engine -> Tonlänge."""
    client = _client(settings)
    project_id = _create_project_with(client, "dummy")

    client.post(f"/projects/{project_id}/options", data={"speed": "1.5"})
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    schnell = Project.load(settings.projects_dir / project_id)
    assert schnell.engine_options["speed"] == 1.5
    audio, rate = read_wav(schnell.output_path)
    kurz = duration_seconds(audio, rate)

    langsam_id = _create_project_with(client, "dummy")
    client.post(f"/projects/{langsam_id}/options", data={"speed": "0.5"})
    client.post(f"/projects/{langsam_id}/run")
    _wait_for_run(client, langsam_id)
    audio, rate = read_wav(Project.load(settings.projects_dir / langsam_id).output_path)

    assert duration_seconds(audio, rate) > kurz


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


# -- Vergleichslauf ---------------------------------------------------------


def _create_comparison(client: TestClient, **werte: str) -> str:
    daten = {
        "name": "Tempo",
        "text": "Am 3. Mai 2024 begann es.",
        "voice": "test-stimme",
        "engine": "dummy",
        "werte_speed": "0.8, 1.2",
    }
    daten.update(werte)
    response = client.post("/comparisons", data=daten, follow_redirects=False)
    assert response.status_code == 303, response.text
    return response.headers["location"].rsplit("/", 1)[-1]


def _wait_for_comparison(client: TestClient, comparison_id: str, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seite = client.get(f"/comparisons/{comparison_id}/table").text
        if "läuft" not in seite:
            return seite
        time.sleep(0.1)
    raise AssertionError("Vergleichslauf wurde nicht fertig")


def test_vergleich_anlegen_und_rendern(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    comparison_id = _create_comparison(client)

    seite = client.get(f"/comparisons/{comparison_id}")
    assert seite.status_code == 200
    assert "Sprechtempo 0.8" in seite.text
    assert "Sprechtempo 1.2" in seite.text

    client.post(f"/comparisons/{comparison_id}/run")
    tabelle = _wait_for_comparison(client, comparison_id)

    assert "2 von 2 Varianten" in tabelle
    assert "Zeichen/s" in tabelle
    for slug in ("sprechtempo-0-8", "sprechtempo-1-2"):
        assert client.get(f"/comparisons/{comparison_id}/variants/{slug}/audio").status_code == 200


def test_vergleich_ohne_raster_wird_abgelehnt(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    response = client.post(
        "/comparisons",
        data={"name": "Leer", "text": "Hallo.", "voice": "test-stimme", "engine": "dummy"},
    )
    assert response.status_code == 400


def test_tippfehler_in_einer_achse_verwirft_nicht_das_formular(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Sonst käme für einen Zahlendreher in einer von drei Achsen das ganze
    Formular zurück -- mitsamt der Arbeit, die drinsteckt."""
    client = _client(settings)
    comparison_id = _create_comparison(client, werte_speed="0.8, x, 1.2")
    assert "Sprechtempo 0.8" in client.get(f"/comparisons/{comparison_id}").text


def test_reglerfelder_folgen_der_engine(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    felder = client.get("/comparisons/fields", params={"engine": "f5-de"}).text
    assert 'name="werte_nfe_step"' in felder
    assert 'name="werte_pitch"' not in felder


def test_ruhende_vergleichstabelle_hat_keinen_ausloeser(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    comparison_id = _create_comparison(client)
    seite = client.get(f"/comparisons/{comparison_id}").text
    assert 'hx-trigger=""' not in seite
    assert "every 2000ms" not in seite


def test_laufende_vergleichstabelle_fragt_nach(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    comparison_id = _create_comparison(client)
    tabelle = client.post(f"/comparisons/{comparison_id}/run").text
    assert "every 2000ms" in tabelle
    _wait_for_comparison(client, comparison_id)


def test_vergleich_loeschen(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    comparison_id = _create_comparison(client)
    client.post(f"/comparisons/{comparison_id}/delete", follow_redirects=False)
    assert client.get(f"/comparisons/{comparison_id}").status_code == 404


def test_unbekannter_vergleich_gibt_404(settings: Settings) -> None:
    assert _client(settings).get("/comparisons/gibtsnicht").status_code == 404


def test_vergleich_taucht_nicht_in_der_projektliste_auf(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    _create_comparison(client)
    assert "Noch keine Projekte" in client.get("/projects").text


def test_fehlende_stimmaehnlichkeit_ist_ein_hinweis_kein_fehler(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Die Statuskarte muss erklären, warum die Spalte leer bleibt -- und die
    fertige Spur trotzdem anbieten."""
    client = TestClient(create_app(settings, DummyASR, embedder_factory=_kaputte_fabrik))
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    seite = client.get(f"/projects/{project_id}").text
    assert "Ohne Stimmähnlichkeit gerendert" in seite
    assert "speechbrain" in seite
    assert client.get(f"/projects/{project_id}/output").status_code == 200


def _kaputte_fabrik():  # noqa: ANN202
    raise RuntimeError("speechbrain ist nicht installiert")


# -- Vorlage eines bestehenden Projekts ändern ------------------------------


def _configure(client: TestClient, project_id: str, **werte: str):  # noqa: ANN202
    daten = {"text": TEXT, "voice": "test-stimme", "engine": "dummy"}
    daten.update(werte)
    return client.post(f"/projects/{project_id}/configure", data=daten)


def test_text_aendern_behaelt_den_ton_der_gleichen_saetze(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    antwort = _configure(client, project_id, text=TEXT + " Ein neuer Satz kommt dazu.")
    assert antwort.status_code == 200
    assert "Übernommen:" in antwort.text
    # Nach dem Übernehmen steht der Reiter auf den Einstellungen, damit die
    # Rückmeldung dort steht, wo die Änderung ausgelöst wurde.
    kompakt = " ".join(antwort.text.split())
    assert 'id="reiter-einstellungen" checked' in kompakt
    assert 'id="reiter-saetze" checked' not in kompakt

    project = Project.load(settings.projects_dir / project_id)
    assert project.source_text.endswith("Ein neuer Satz kommt dazu.")
    assert any(c.status == ChunkStatus.OK for c in project.chunks)
    assert any(c.status == ChunkStatus.PENDING for c in project.chunks)


def test_stimmwechsel_ueber_die_oberflaeche(settings: Settings, voice_store: VoiceStore) -> None:
    voice_store.add(
        "zweite-stimme", voice_store.get("test-stimme").audio_path, transcript="Wortlaut."
    )
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    _configure(client, project_id, voice="zweite-stimme")

    project = Project.load(settings.projects_dir / project_id)
    assert project.voice == "zweite-stimme"
    assert all(c.status == ChunkStatus.PENDING for c in project.chunks)


def test_aendern_auf_unbekannte_stimme_wird_abgelehnt(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    assert _configure(client, project_id, voice="gibtsnicht").status_code == 400


def test_aendern_auf_leeren_text_wird_abgelehnt(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    assert _configure(client, project_id, text="   ").status_code == 400


def test_aendern_waehrend_eines_laufs_wird_abgelehnt(
    settings: Settings, voice_store: VoiceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sonst zöge der Umbau dem laufenden Renderer die Chunks unter den Füßen weg."""
    from cloney.engines import registry

    class Bedaechtig(DummyEngine):
        def synthesize(self, text, voice, seed):  # noqa: ANN001, ANN202
            time.sleep(0.4)
            return super().synthesize(text, voice, seed)

    monkeypatch.setitem(registry._FACTORIES, "dummy", lambda _s, _o: Bedaechtig())

    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    try:
        assert _configure(client, project_id).status_code == 409
    finally:
        _wait_for_run(client, project_id)


def test_projektseite_zeigt_die_reiter_statt_verschachtelter_klappboxen(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Reine CSS-Reiter: ohne Skript bleibt der gewählte Reiter stehen, auch
    wenn htmx die Statusleiste darüber austauscht."""
    client = _client(settings)
    project_id = _create_project(client)
    seite = client.get(f"/projects/{project_id}").text

    for reiter in ("reiter-saetze", "reiter-einstellungen", "reiter-projekt"):
        assert f'id="{reiter}"' in seite
        assert f'for="{reiter}"' in seite
    # Die Vorlage ist von der Projektseite aus änderbar, nicht nur beim Anlegen.
    assert f'action="/projects/{project_id}/configure"' in seite
    assert 'name="text"' in seite


# -- Trainierte Modelle ------------------------------------------------------


@pytest.fixture
def modelle(settings: Settings, tmp_path: Path) -> ModelStore:
    """Ein eingetragener Stand. Die Dateien sind Attrappen -- geprüft wird, was
    Cloney mit dem Eintrag macht, nicht ob F5 ihn laden kann."""
    store = ModelStore(settings.models_dir)
    ckpt = tmp_path / "model_last.pt"
    vocab = tmp_path / "vocab.txt"
    ckpt.write_bytes(b"kein echtes Modell")
    vocab.write_text("a\nb\n", encoding="utf-8")
    store.add("anna-ft", ckpt, vocab, note="anna, 1 min")
    return store


def test_projekt_gegen_trainierten_stand_anlegen(
    settings: Settings, voice_store: VoiceStore, modelle: ModelStore
) -> None:
    client = _client(settings)
    assert "anna-ft" in client.get("/").text

    project_id = _create_project(client, model="anna-ft")

    assert Project.load(settings.projects_dir / project_id).model == "anna-ft"
    assert "anna-ft" in client.get(f"/projects/{project_id}").text


def test_unbekannter_stand_wird_abgelehnt(settings: Settings, voice_store: VoiceStore) -> None:
    antwort = _client(settings).post(
        "/projects",
        data={
            "name": "Testlauf",
            "text": TEXT,
            "voice": "test-stimme",
            "engine": "dummy",
            "model": "gibt-es-nicht",
        },
    )
    assert antwort.status_code == 400
    assert "gibt-es-nicht" in antwort.text


def test_einzelner_satz_rendert_gegen_denselben_stand(
    settings: Settings,
    voice_store: VoiceStore,
    modelle: ModelStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sonst würfelte ein einzelner Satz gegen den Pretrain und klänge neben
    seinen Nachbarn nach einem anderen Sprecher."""
    import cloney.web.app as web

    gesehen: list[str] = []
    echt = web.create_engine

    def merken(name: str, eigene: Settings, options: dict) -> object:
        gesehen.append(eigene.f5_ckpt_path)
        return echt(name, eigene, options)

    monkeypatch.setattr(web, "create_engine", merken)

    client = _client(settings)
    project_id = _create_project(client, model="anna-ft")
    assert client.post(f"/projects/{project_id}/chunks/0/reroll").status_code == 200

    assert gesehen == [modelle.get("anna-ft").ckpt_path]


def test_fehlender_checkpoint_meldet_sich_beim_rendern(
    settings: Settings, voice_store: VoiceStore, modelle: ModelStore
) -> None:
    """Ein verschobener Checkpoint darf nicht stillschweigend zum Pretrain
    zurückfallen -- das wäre ein anderer Sprecher ohne jeden Hinweis."""
    client = _client(settings)
    project_id = _create_project(client, model="anna-ft")
    Path(modelle.get("anna-ft").ckpt_path).unlink()

    antwort = client.post(f"/projects/{project_id}/chunks/0/reroll")
    assert antwort.status_code == 400
    assert "anna-ft" in antwort.text


def test_modellwechsel_verwirft_den_ton(
    settings: Settings, voice_store: VoiceStore, modelle: ModelStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    assert _configure(client, project_id, model="anna-ft").status_code == 200

    project = Project.load(settings.projects_dir / project_id)
    assert project.model == "anna-ft"
    assert all(chunk.audio_file is None for chunk in project.chunks)


def test_vergleich_stellt_pretrain_gegen_finetune(
    settings: Settings, voice_store: VoiceStore, modelle: ModelStore
) -> None:
    """Die Frage, die ein Finetune aufwirft: hat er etwas gebracht? Ohne den
    Pretrain in derselben Tabelle ist sie nicht zu beantworten."""
    client = _client(settings)
    assert "anna-ft" in client.get("/comparisons").text

    antwort = client.post(
        "/comparisons",
        data={
            "name": "Pretrain gegen Finetune",
            "text": "Am 3. Mai 2024 begann es.",
            "voice": "test-stimme",
            "engine": "dummy",
            "models": ["", "anna-ft"],
        },
        follow_redirects=False,
    )
    assert antwort.status_code == 303, antwort.text
    comparison_id = antwort.headers["location"].rsplit("/", 1)[-1]

    comparison = Comparison.load(settings.comparisons_dir / comparison_id)
    assert [v.model for v in comparison.variants] == ["", "anna-ft"]
    seite = client.get(f"/comparisons/{comparison_id}").text
    assert "Pretrain" in seite and "anna-ft" in seite
