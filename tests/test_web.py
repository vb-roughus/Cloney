"""Routen-Tests gegen die ASGI-App -- ohne Server, ohne Browser, ohne GPU."""

from __future__ import annotations

import io
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from cloney.asr.dummy import DummyASR
from cloney.config import Settings
from cloney.core.audio import duration_seconds, read_wav
from cloney.core.compare import Comparison
from cloney.core.lexicon import Lexicon
from cloney.core.models import ModelStore
from cloney.core.project import ChunkStatus, Project
from cloney.core.voices import VoiceStore
from cloney.engines.dummy import DummyEngine
from cloney.web.app import anzahl, create_app, zeitpunkt
from cloney.web.jobs import ComparisonRunner, JobRunner

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


def test_titel_ist_in_der_satztabelle_erkennbar(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Eine Erkennung, die man nicht sieht, kann man nicht widerlegen -- und
    genau das muss möglich sein, wenn eine Regel eine Zeile falsch einordnet."""
    client = _client(settings)
    project_id = _create_project(client, text="Kapitel 3\n\nEs war einmal ein Satz.")

    seite = client.get(f"/projects/{project_id}").text
    assert "Kapitel drei." in seite
    assert ">Titel<" in seite


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
    felder = client.get(
        "/comparisons/achsen", params={"engine": "f5-de", "voice": "test-stimme"}
    ).text
    assert 'name="werte_nfe_step"' in felder
    assert 'name="werte_pitch"' not in felder


def test_regler_stehen_auf_ihrer_vorgabe(settings: Settings, voice_store: VoiceStore) -> None:
    """Ein leeres Feld beantwortet die Frage nicht, die man beim Anlegen hat:
    was ist der Ausgangspunkt, von dem ich abweiche?"""
    client = _client(settings)

    felder = client.get(
        "/comparisons/achsen", params={"engine": "f5-de", "voice": "test-stimme"}
    ).text

    # Auf Formatierung zu prüfen wäre brüchig -- gemeint ist, welcher Wert
    # vorgewählt ist. Die Vorgaben sind 1.0, 32 und 2.0.
    gewaehlt = re.findall(r'<option value="([^"]+)"\s+selected>', felder)
    assert gewaehlt == ["1", "32", "2"]


def test_gewaehlte_werte_ueberstehen_den_stimmwechsel(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Wer nur die Stimme wechselt, soll nicht sein Raster verlieren."""
    client = _client(settings)

    felder = client.get(
        "/comparisons/achsen",
        params={"engine": "dummy", "voice": "test-stimme", "werte_speed": ["0.7", "1.3"]},
    ).text

    assert felder.count('name="werte_speed"') == 2


def test_ein_weiteres_wertfeld_kommt_auf_anforderung(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)

    feld = client.get("/comparisons/wertfeld", params={"engine": "dummy", "key": "speed"}).text

    assert 'name="werte_speed"' in feld
    assert "data-entfernen" in feld


def test_unbekannter_regler_hat_kein_wertfeld(settings: Settings, voice_store: VoiceStore) -> None:
    antwort = _client(settings).get(
        "/comparisons/wertfeld", params={"engine": "dummy", "key": "gibtesnicht"}
    )

    assert antwort.status_code == 404


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


# -- Ansicht hält sich selbst aktuell ---------------------------------------


def test_satztabelle_laedt_waehrend_des_laufs_nach(
    settings: Settings, voice_store: VoiceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wer beim Rendern zusieht, soll fertige Sätze sehen und anhören können,
    ohne die Seite neu zu laden.

    Der Lauf wird dafür nicht wirklich gestartet, sondern als laufend gemeldet:
    ein Dummy-Lauf ist schneller vorbei, als der Abruf hier ankäme.
    """
    client = _client(settings)
    project_id = _create_project(client)
    monkeypatch.setattr(JobRunner, "is_running", lambda self, key: True)

    tabelle = client.get(f"/projects/{project_id}/table").text

    assert f'hx-get="/projects/{project_id}/table"' in tabelle
    assert "every 2000ms" in tabelle


def test_satztabelle_hoert_auf_nachzuladen(settings: Settings, voice_store: VoiceStore) -> None:
    """Im Ruhezustand wird nicht abgefragt, sondern gehorcht.

    Endlos alle zwei Sekunden nachzuladen wäre die eine falsche Antwort, ein
    leeres hx-trigger die andere: htmx fiele damit auf seinen Standard zurück,
    bei einem div also auf den Klick.
    """
    client = _client(settings)
    project_id = _create_project(client)

    tabelle = client.get(f"/projects/{project_id}/table").text

    assert 'id="chunks"' in tabelle
    assert "every 2000ms" not in tabelle
    assert "lauf-gestartet from:body" in tabelle


def test_satztabelle_kommt_beim_start_eines_laufs_in_gang(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Der Knopf tauscht nur die Statusleiste aus. Ohne dieses Ereignis stünde
    die Tabelle den ganzen Lauf über unverändert da -- und danach auch."""
    client = _client(settings)
    project_id = _create_project(client)

    antwort = client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    assert antwort.headers.get("HX-Trigger") == "lauf-gestartet"


def test_ein_handgriff_an_einer_zeile_frischt_den_zaehler_auf(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Die Statusleiste steht außerhalb der Zeile und wird nicht mit
    ausgetauscht. Ohne das Ereignis bliebe dort eine Zahl stehen, die seit dem
    verworfenen Ton nicht mehr gilt."""
    client = _client(settings)
    project_id = _create_project(client)

    antwort = client.post(f"/projects/{project_id}/chunks/0/reroll")

    assert antwort.headers.get("HX-Trigger") == "satz-geaendert"
    assert "satz-geaendert from:body" in client.get(f"/projects/{project_id}/status").text


def test_fertige_saetze_sind_waehrend_des_laufs_hoerbar(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Der eigentliche Wunsch: nicht bis zum Ende warten müssen."""
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    # Nach dem Lauf steht jeder Satz einzeln bereit -- derselbe Weg, den die
    # Tabelle während des Laufs für die schon fertigen anbietet.
    antwort = client.get(f"/projects/{project_id}/chunks/0/audio")
    assert antwort.status_code == 200
    # Ohne diese Kopfzeile zeigte der Browser nach dem Neuwürfeln den alten Ton.
    assert antwort.headers["cache-control"] == "no-cache"


def test_uebersicht_laedt_nur_bei_laufenden_nach(
    settings: Settings, voice_store: VoiceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(settings)
    _create_project(client)

    ruhig = client.get("/uebersicht")
    assert ruhig.status_code == 200
    assert "hx-trigger" not in ruhig.text

    monkeypatch.setattr(JobRunner, "is_running", lambda self, key: True)
    laufend = client.get("/uebersicht").text
    assert 'hx-get="/uebersicht"' in laufend
    assert "Läuft gerade" in laufend


# -- Aussprache --------------------------------------------------------------


def test_eingetragene_aussprache_landet_in_der_sprechfassung(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Der ganze Zweck: was eingetragen ist, geht so ins Modell."""
    client = _client(settings)
    antwort = client.post("/lexicon", data={"word": "SWIFT", "spoken": "Ssuift"})
    assert antwort.status_code == 200

    project_id = _create_project(client, text="Die SWIFT-Nachricht kam an.")

    project = Project.load(settings.projects_dir / project_id)
    assert "Ssuift" in project.chunks[0].normalized_text
    assert "SWIFT" in project.chunks[0].raw_text  # der Rohtext bleibt, wie er ist


def test_kandidaten_kommen_aus_den_projekttexten(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Wer entscheiden soll, muss sehen, worüber -- samt Vorschlag fürs
    Buchstabieren und dem Projekt, in dem das Wort steht."""
    client = _client(settings)
    _create_project(client, name="Kapitel 1", text="Die USB-Verbindung stand.")

    seite = client.get("/lexicon").text
    assert "USB" in seite
    assert "U-Es-Be" in seite
    assert "Kapitel 1" in seite


def test_eingetragenes_wort_ist_kein_kandidat_mehr(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    _create_project(client, text="Die USB-Verbindung stand.")
    client.post("/lexicon", data={"word": "USB", "spoken": "U-Es-Be"})

    seite = client.get("/lexicon").text
    assert "Kandidaten aus den Projekten" not in seite


def test_eintrag_laesst_sich_aendern(settings: Settings, voice_store: VoiceStore) -> None:
    """Auch das Wort selbst: ein Tippfehler steckt dort genauso oft."""
    client = _client(settings)
    client.post("/lexicon", data={"word": "SWFIT", "spoken": "Ssuift"})

    antwort = client.post("/lexicon/SWFIT/edit", data={"new_word": "SWIFT", "spoken": "Suift"})

    assert antwort.status_code == 200
    lexikon = Lexicon.load(settings.data_dir)
    assert lexikon.entries == {"SWIFT": "Suift"}


def test_aendern_eines_unbekannten_eintrags_meldet_sich(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    antwort = client.post("/lexicon/GIBTESNICHT/edit", data={"new_word": "X", "spoken": "Iks"})
    assert antwort.status_code == 404


def test_eintrag_laesst_sich_entfernen(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    client.post("/lexicon", data={"word": "Journal", "spoken": "Schurnahl"})

    antwort = client.post("/lexicon/Journal/delete")

    assert antwort.status_code == 200
    # Der Platzhalter im Formular nennt dasselbe Wort -- geprüft wird die Liste.
    assert "Noch nichts eingetragen" in antwort.text
    assert client.post("/lexicon/Journal/delete").status_code == 404


def test_leere_sprechweise_wird_abgelehnt(settings: Settings, voice_store: VoiceStore) -> None:
    """Sonst verschwände das Wort, und niemand merkte es bis zum Hören."""
    client = _client(settings)
    assert client.post("/lexicon", data={"word": "SWIFT", "spoken": "  "}).status_code == 400


def test_abspieler_nimmt_die_breite_des_satzes_ein(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """In einer eigenen Spalte von fünfzehn Zeichen Breite ist die Suchleiste ein
    paar Pixel lang -- eine Stelle anzuspringen war Glückssache."""
    client = _client(settings)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    seite = client.get(f"/projects/{project_id}").text

    assert 'class="satzton"' in seite
    assert "spalte-ton" not in seite
    # Der Knopf zum Übernehmen gehört zum Formular, steht aber außerhalb davon,
    # damit er nicht zwischen Satz und Ton gerät.
    assert 'form="satz-0"' in seite


def test_neu_wuerfeln_nimmt_die_aktuelle_aussprache(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Der gemeldete Fall: Eintrag angelegt, Satz neu gewürfelt -- und es klang
    wie vorher. Die Sprechfassung stand seit dem Anlegen im Manifest."""
    client = _client(settings)
    project_id = _create_project(client, text="Die SWIFT-Nachricht kam an.")
    assert "SWIFT" in Project.load(settings.projects_dir / project_id).chunks[0].normalized_text

    client.post("/lexicon", data={"word": "SWIFT", "spoken": "Ssuift"})
    assert client.post(f"/projects/{project_id}/chunks/0/reroll").status_code == 200

    chunk = Project.load(settings.projects_dir / project_id).chunks[0]
    assert "Ssuift" in chunk.normalized_text
    assert "SWIFT" in chunk.raw_text


def test_text_uebernehmen_nimmt_die_aktuelle_aussprache(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client, text="Ein Satz ohne Besonderheit.")
    client.post("/lexicon", data={"word": "Journal", "spoken": "Schurnahl"})

    client.post(f"/projects/{project_id}/chunks/0/text", data={"raw_text": "Im Journal stand es."})

    chunk = Project.load(settings.projects_dir / project_id).chunks[0]
    assert chunk.normalized_text == "Im Schurnahl stand es."


def test_auffrischen_trifft_nur_die_betroffenen_saetze(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Was gleich klingt, behält seinen Ton -- sonst kostete ein einzelner
    Eintrag das ganze Kapitel."""
    client = _client(settings)
    # Zwei Absätze, damit es zwei Sätze werden -- sonst stünde beides in einem
    # Chunk, und "nur die betroffenen" wäre nicht zu zeigen.
    project_id = _create_project(
        client, text="Ein Satz ohne Besonderheit.\n\nDie SWIFT-Nachricht kam an."
    )
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)
    vorher = Project.load(settings.projects_dir / project_id)
    unberuehrt = vorher.chunks[0].seed

    client.post("/lexicon", data={"word": "SWIFT", "spoken": "Ssuift"})
    antwort = client.post(f"/projects/{project_id}/refresh-spoken")

    assert antwort.status_code == 200
    nachher = Project.load(settings.projects_dir / project_id)
    assert nachher.chunks[0].seed == unberuehrt
    assert nachher.chunks[0].audio_file is not None
    assert "Ssuift" in nachher.chunks[1].normalized_text
    assert nachher.chunks[1].status == ChunkStatus.PENDING


def test_auffrischen_ohne_aenderung_sagt_das_auch(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client)

    antwort = client.post(f"/projects/{project_id}/refresh-spoken")

    assert "Keine Sprechfassung hat sich geändert" in antwort.text


def test_saetze_lassen_sich_nach_zustand_aussuchen(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Der Fall aus dem Betrieb: nach einem Lauf die auffälligen durchhören,
    ohne sie aus neunzig unauffälligen herauszusuchen."""
    client = _client(settings)
    project_id = _create_project(client, text="Erster Satz.\n\nZweiter Satz.")
    project = Project.load(settings.projects_dir / project_id)
    project.chunks[0].status = ChunkStatus.OK
    project.chunks[1].status = ChunkStatus.NEEDS_REVIEW
    project.save()

    seite = client.get(f"/projects/{project_id}/table", params={"status": "pruefen"}).text

    assert "Zweiter Satz." in seite
    assert "Erster Satz." not in seite
    assert "1 von 2 Sätzen" in seite


def test_saetze_lassen_sich_durchsuchen(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client, text="Erster Satz.\n\nZweiter Satz.")

    seite = client.get(f"/projects/{project_id}/table", params={"q": "zweiter"}).text

    assert "Zweiter Satz." in seite
    assert "Erster Satz." not in seite


def test_ohne_treffer_steht_da_warum(settings: Settings, voice_store: VoiceStore) -> None:
    """Eine leere Tabelle allein sähe aus wie ein Fehler im Projekt."""
    client = _client(settings)
    project_id = _create_project(client)

    seite = client.get(f"/projects/{project_id}/table", params={"q": "gibtesnicht"}).text

    assert "Kein Satz passt dazu" in seite


def test_der_filter_uebersteht_das_nachladen(
    settings: Settings, voice_store: VoiceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sonst käme nach zwei Sekunden die ungefilterte Liste zurück -- und die
    Auswahl wäre mitten im Durchhören weg."""
    client = _client(settings)
    project_id = _create_project(client)
    monkeypatch.setattr(JobRunner, "is_running", lambda self, key: True)

    seite = client.get(f"/projects/{project_id}/table", params={"status": "pruefen"}).text

    assert f'hx-get="/projects/{project_id}/table?status=pruefen' in seite


def test_rueckfragen_stehen_im_dokument(settings: Settings, voice_store: VoiceStore) -> None:
    """Der Kasten gehört ins Grundgerüst: gefragt wird auf jeder Seite."""
    seite = _client(settings).get("/").text

    assert '<dialog id="rueckfrage"' in seite


def test_loeschen_fragt_zurueck(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)

    seite = client.get(f"/projects/{project_id}").text

    assert 'data-frage="Projekt Testlauf samt erzeugtem Ton löschen?"' in seite
    assert 'data-frage-ja="Löschen"' in seite


def test_keine_browsereigenen_rueckfragen_mehr() -> None:
    """confirm() setzt seinen Kasten an den oberen Fensterrand, weit weg vom
    Knopf. Ein einzelnes zurückgebliebenes onclick fiele im Betrieb erst auf,
    wenn jemand davor steht -- deshalb hier."""
    vorlagen = Path(__file__).resolve().parents[1] / "cloney" / "web" / "templates"

    uebrig = [p.name for p in vorlagen.rglob("*.html") if "confirm(" in p.read_text("utf-8")]

    assert uebrig == []


def test_name_mit_apostroph_bleibt_eine_frage(settings: Settings, voice_store: VoiceStore) -> None:
    """Vorher stand der Name in einem JavaScript-String in einem HTML-Attribut.
    Ein Apostroph darin beendete den String -- und der Knopf löschte danach
    ungefragt. Als Attributwert ist er nur noch Text."""
    client = _client(settings)
    project_id = _create_project(client, name="Rolfs' Kapitel")

    seite = client.get(f"/projects/{project_id}").text

    assert 'data-frage="Projekt Rolfs&#39; Kapitel samt' in seite


def test_stimme_loeschen_fragt_zurueck(settings: Settings, voice_store: VoiceStore) -> None:
    """Der Knopf sitzt im Formular fürs Speichern -- die Frage muss deshalb an
    ihm hängen und nicht am Formular."""
    seite = _client(settings).get("/voices").text

    assert 'data-frage="Stimme test-stimme samt allen Lagen löschen?"' in seite


# -- Emotionslagen ---------------------------------------------------------


def _lage_anlegen(client: TestClient, name: str = "ernst") -> None:
    daten = np.zeros(24000 * 6, dtype=np.float32)
    puffer = io.BytesIO()
    sf.write(puffer, daten, 24000, format="WAV")
    antwort = client.post(
        "/voices/test-stimme/lagen",
        data={"lage": name, "transcript": f"{name.capitalize()} gesprochen."},
        files={"audio": (f"{name}.wav", puffer.getvalue(), "audio/wav")},
    )
    assert antwort.status_code == 200, antwort.text


def test_lage_laesst_sich_ueber_die_oberflaeche_anlegen(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    _lage_anlegen(client)

    seite = client.get("/voices").text

    assert "ernst" in seite
    assert "neutral und 1 weitere" in seite
    assert client.get("/voices/test-stimme/lagen/ernst/audio").status_code == 200


def test_ohne_weitere_lage_steht_kein_chip_in_der_zeile(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Ein Chip verspräche eine Wahl, die es nicht gibt."""
    client = _client(settings)
    project_id = _create_project(client)

    tabelle = client.get(f"/projects/{project_id}/table").text

    assert "chunks/0/lage" not in tabelle


def test_mit_zweiter_lage_steht_der_chip_da(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    _lage_anlegen(client)
    project_id = _create_project(client)

    tabelle = client.get(f"/projects/{project_id}/table").text

    assert f"/projects/{project_id}/chunks/0/lage" in tabelle
    assert ">neutral</button>" in tabelle


def test_die_auswahl_kommt_erst_auf_klick(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    _lage_anlegen(client)
    project_id = _create_project(client)

    auswahl = client.get(f"/projects/{project_id}/chunks/0/lage").text

    assert "neutral" in auswahl
    assert "ernst" in auswahl


def test_gewaehlte_lage_landet_im_manifest(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    _lage_anlegen(client)
    project_id = _create_project(client)

    zeile = client.post(f"/projects/{project_id}/chunks/0/lage", data={"lage": "ernst"}).text

    assert ">ernst</button>" in zeile
    assert Project.load(settings.projects_dir / project_id).chunks[0].lage == "ernst"


def test_lagenwechsel_nimmt_den_ton_zurueck(settings: Settings, voice_store: VoiceStore) -> None:
    """Der vorhandene Ton stammt aus einer anderen Aufnahme."""
    client = _client(settings)
    _lage_anlegen(client)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/run")
    _wait_for_run(client, project_id)

    client.post(f"/projects/{project_id}/chunks/0/lage", data={"lage": "ernst"})

    chunk = Project.load(settings.projects_dir / project_id).chunks[0]
    assert chunk.status == ChunkStatus.PENDING
    assert chunk.audio_file is None


def test_einzelner_satz_laesst_sich_nachrendern(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Nach einem Lagenwechsel der Weg, genau diesen Satz zu hören -- ohne
    Seed und Text anzurühren."""
    client = _client(settings)
    project_id = _create_project(client)
    vorher = Project.load(settings.projects_dir / project_id).chunks[0].seed

    client.post(f"/projects/{project_id}/chunks/0/render")

    chunk = Project.load(settings.projects_dir / project_id).chunks[0]
    assert chunk.audio_file
    assert chunk.seed == vorher


def test_lage_laesst_sich_wieder_entfernen(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    _lage_anlegen(client)

    antwort = client.post("/voices/test-stimme/lagen/ernst/delete", follow_redirects=False)

    assert antwort.status_code == 303
    assert voice_store.lagen("test-stimme") == ["neutral"]


def test_neutral_laesst_sich_nicht_einzeln_loeschen(
    settings: Settings, voice_store: VoiceStore
) -> None:
    antwort = _client(settings).post("/voices/test-stimme/lagen/neutral/delete")

    assert antwort.status_code == 400
    assert "Hauptaufnahme" in antwort.text


# -- Zuschnitt eines Vergleichs --------------------------------------------


def test_vorschau_zeigt_die_zeilen_vor_dem_rendern(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Sonst erfährt man erst nach zwölf Minuten Rechenzeit, was entstanden ist."""
    client = _client(settings)

    seite = client.get(
        "/comparisons/vorschau",
        params={"engine": "dummy", "voice": "test-stimme", "werte_speed": ["0.8", "1.2"]},
    ).text

    assert "2 Varianten" in seite
    assert "Sprechtempo 0.8" in seite
    assert "Sprechtempo 1.2" in seite


def test_vorschau_sagt_wenn_es_nichts_zu_vergleichen_gibt(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)

    seite = client.get(
        "/comparisons/vorschau",
        params={"engine": "dummy", "voice": "test-stimme", "werte_speed": ["1.0"]},
    ).text

    assert "mindestens zwei Varianten" in seite
    assert "unfertig" in seite


def test_die_textprobe_steckt_hinter_einem_knopf(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Sie ist das Längste am Formular und das, was man am seltensten anfasst."""
    seite = _client(settings).get("/comparisons").text

    assert 'data-oeffnet="#textfenster"' in seite
    assert '<dialog id="textfenster" class="seitenfenster">' in seite


def test_die_probe_ist_kein_pflichtfeld(settings: Settings, voice_store: VoiceStore) -> None:
    """Ein Pflichtfeld in einem geschlossenen Fenster kann der Browser nicht
    anspringen -- er lehnte das Absenden dann wortlos ab. Geprüft wird auf dem
    Server."""
    client = _client(settings)
    seite = client.get("/comparisons").text
    fenster = seite[seite.index('<dialog id="textfenster"') :]
    assert "required" not in fenster[: fenster.index("</dialog>")]

    antwort = client.post(
        "/comparisons",
        data={
            "name": "Ohne Text",
            "voice": "test-stimme",
            "engine": "dummy",
            "text": "",
            "werte_speed": ["0.8", "1.2"],
        },
    )
    assert antwort.status_code == 400
    assert "Textprobe ist leer" in antwort.text


def test_vergleich_laesst_sich_aendern(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    comparison_id = _create_comparison(client, werte_speed="0.8, 1.2")

    antwort = client.post(
        f"/comparisons/{comparison_id}/edit",
        data={
            "name": "Neuer Name",
            "voice": "test-stimme",
            "engine": "dummy",
            "text": "Ein Satz.",
            "werte_speed": ["0.8", "1.2", "1.4"],
        },
        follow_redirects=False,
    )

    assert antwort.status_code == 303
    geladen = Comparison.load(settings.comparisons_dir / comparison_id)
    assert geladen.name == "Neuer Name"
    assert len(geladen.variants) == 3


def test_die_maske_zum_aendern_ist_gefuellt(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    comparison_id = _create_comparison(client, werte_speed="0.7, 1.3")

    seite = client.get(f"/comparisons/{comparison_id}/edit").text

    gewaehlt = re.findall(r'<option value="([^"]+)"\s+selected>', seite)
    assert "0.7" in gewaehlt
    assert "1.3" in gewaehlt


def test_vergleich_aendern_waehrend_eines_laufs_wird_abgelehnt(
    settings: Settings, voice_store: VoiceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sonst zöge der Umbau dem laufenden Renderer die Varianten unter den
    Füßen weg."""
    client = _client(settings)
    comparison_id = _create_comparison(client, werte_speed="0.8, 1.2")
    # Vergleiche haben einen eigenen Runner -- JobRunner zu patchen wirkte hier
    # nicht, und der Test wäre grün geworden, ohne etwas zu prüfen.
    monkeypatch.setattr(ComparisonRunner, "is_running", lambda self, key: True)

    antwort = client.post(
        f"/comparisons/{comparison_id}/edit",
        data={
            "name": "X",
            "voice": "test-stimme",
            "engine": "dummy",
            "text": "Ein Satz.",
            "werte_speed": ["0.8", "1.2"],
        },
    )

    assert antwort.status_code == 409


def test_unbekannte_lage_faellt_aus_dem_zuschnitt(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Sie fiele beim Rendern ohnehin auf die Hauptaufnahme zurück -- dann
    stünden zwei Zeilen da, die dasselbe messen."""
    client = _client(settings)

    antwort = client.post(
        "/comparisons",
        data={
            "name": "Lagen",
            "voice": "test-stimme",
            "engine": "dummy",
            "text": "Ein Satz.",
            "werte_speed": ["1.0"],
            "lagen": ["neutral", "gibtesnicht"],
        },
    )

    assert antwort.status_code == 400
    assert "mindestens zwei Varianten" in antwort.text


# -- Einklappen, Markieren, Vorgabe ----------------------------------------


def _mit_lage(client: TestClient) -> None:
    """Der Stimme eine zweite Lage geben -- ohne die gäbe es nichts zu wählen."""
    _lage_anlegen(client)


def test_eingeklappt_bleibt_nur_der_wortlaut(settings: Settings, voice_store: VoiceStore) -> None:
    """Ein Kapitel hat schnell hundert Sätze, und jeder trägt ausgeklappt ein
    Textfeld, zwei Rückschriften, einen Abspieler und vier Knöpfe."""
    client = _client(settings)
    project_id = _create_project(client, text="Der erste Satz.\n\nDer zweite Satz.")

    tabelle = client.get(f"/projects/{project_id}/table", params={"kompakt": 1}).text

    assert "Der erste Satz." in tabelle
    assert "<textarea" not in tabelle
    assert "Neu würfeln" not in tabelle


def test_ein_einzelner_satz_laesst_sich_ausklappen(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    project_id = _create_project(client, text="Der erste Satz.\n\nDer zweite Satz.")

    tabelle = client.get(f"/projects/{project_id}/table", params={"kompakt": 1, "offen": "1"}).text

    # Genau einer trägt sein Textfeld, der andere nicht.
    assert tabelle.count("<textarea") == 1


def test_der_klappzustand_haengt_an_der_adresse(
    settings: Settings, voice_store: VoiceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sonst käme nach zwei Sekunden die ausgeklappte Liste zurück."""
    client = _client(settings)
    project_id = _create_project(client)
    monkeypatch.setattr(JobRunner, "is_running", lambda self, key: True)

    tabelle = client.get(f"/projects/{project_id}/table", params={"kompakt": 1}).text

    # Jinja schreibt das kaufmännische Und als Entität -- der Browser liest es
    # richtig, der Test muss es also so erwarten.
    assert f"/projects/{project_id}/table?status=alle&amp;q=&amp;kompakt=1" in tabelle


def test_unlesbare_klappliste_kippt_die_tabelle_nicht(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Der Wert kommt aus einer Adresse."""
    antwort = _client(settings).get(
        f"/projects/{_create_project(_client(settings))}/table",
        params={"kompakt": 1, "offen": "0,x,zwei"},
    )

    assert antwort.status_code in (200, 404)


def test_markierte_saetze_bekommen_dieselbe_lage(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    _mit_lage(client)
    project_id = _create_project(client, text="Eins.\n\nZwei.\n\nDrei.")

    client.post(
        f"/projects/{project_id}/chunks/lage",
        data={"auswahl": ["0", "2"], "lage": "ernst", "status": "alle", "q": ""},
    )

    project = Project.load(settings.projects_dir / project_id)
    assert [c.lage for c in project.chunks] == ["ernst", "", "ernst"]


def test_die_sammelanwendung_haelt_den_filter(settings: Settings, voice_store: VoiceStore) -> None:
    """Die Antwort ersetzt die ganze Tabelle -- ohne den Filter stünde danach
    die ungefilterte Liste da."""
    client = _client(settings)
    _mit_lage(client)
    project_id = _create_project(client, text="Eins.\n\nZwei.")

    tabelle = client.post(
        f"/projects/{project_id}/chunks/lage",
        data={"auswahl": ["0"], "lage": "ernst", "status": "offen", "q": "eins", "kompakt": "1"},
    ).text

    assert 'value="offen"\n               checked' in tabelle or "offen" in tabelle
    assert "kompakt=1" in tabelle


def test_verborgenes_bleibt_auch_im_stylesheet_verborgen() -> None:
    """Eine eigene display-Angabe überstimmt das hidden-Attribut.

    Im Browser gemessen: die Sammelleiste stand trotz hidden von Anfang an da,
    weil '.sammelleiste { display: flex }' die Regel des Browsers schlägt. Ein
    Test auf das Attribut allein merkt davon nichts -- deshalb hier auf die
    Regel, die es zurückholt.
    """
    css = (Path(__file__).resolve().parents[1] / "cloney/web/static/style.css").read_text("utf-8")

    ohne_kommentare = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    mit_display = set(re.findall(r"\.([a-z-]+)\s*\{[^}]*\bdisplay\s*:", ohne_kommentare))
    with_hidden = set(re.findall(r"\.([a-z-]+)\[hidden\]", ohne_kommentare))

    assert "sammelleiste" in mit_display
    assert "sammelleiste" in with_hidden


def test_die_sammelleiste_startet_verborgen(settings: Settings, voice_store: VoiceStore) -> None:
    """Ohne Markierung hätte sie nichts, worauf sie wirken könnte."""
    client = _client(settings)
    _mit_lage(client)
    project_id = _create_project(client)

    tabelle = client.get(f"/projects/{project_id}/table").text

    assert '<form id="sammelform" class="sammelleiste" hidden' in tabelle


def test_ohne_zweite_lage_keine_sammelleiste(settings: Settings, voice_store: VoiceStore) -> None:
    client = _client(settings)
    project_id = _create_project(client)

    assert 'id="sammelform"' not in client.get(f"/projects/{project_id}/table").text


def test_die_vorgabe_des_projekts_laesst_sich_setzen(
    settings: Settings, voice_store: VoiceStore
) -> None:
    client = _client(settings)
    _mit_lage(client)
    project_id = _create_project(client, text="Eins.\n\nZwei.")

    seite = client.post(f"/projects/{project_id}/lage", data={"lage": "ernst"}).text

    assert "Vorgabe auf" in seite
    project = Project.load(settings.projects_dir / project_id)
    assert project.lage == "ernst"
    assert all(project.lage_of(c) == "ernst" for c in project.chunks)


def test_die_vorgabe_waehrend_eines_laufs_wird_abgelehnt(
    settings: Settings, voice_store: VoiceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(settings)
    _mit_lage(client)
    project_id = _create_project(client)
    monkeypatch.setattr(JobRunner, "is_running", lambda self, key: True)

    antwort = client.post(f"/projects/{project_id}/lage", data={"lage": "ernst"})

    assert antwort.status_code == 409


def test_der_chip_zeigt_die_geltende_lage(settings: Settings, voice_store: VoiceStore) -> None:
    """Auch wenn sie von der Vorgabe kommt und am Satz nichts steht."""
    client = _client(settings)
    _mit_lage(client)
    project_id = _create_project(client)
    client.post(f"/projects/{project_id}/lage", data={"lage": "ernst"})

    tabelle = client.get(f"/projects/{project_id}/table").text

    assert ">ernst</button>" in tabelle


def test_der_picker_fuehrt_zur_vorgabe_zurueck(settings: Settings, voice_store: VoiceStore) -> None:
    """Ohne ihn bliebe ein von Hand gesetzter Satz für immer abgekoppelt."""
    client = _client(settings)
    _mit_lage(client)
    project_id = _create_project(client)

    auswahl = client.get(f"/projects/{project_id}/chunks/0/lage").text

    assert "wie Projekt" in auswahl
