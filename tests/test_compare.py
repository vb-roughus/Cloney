"""Vergleichslauf ohne GPU: Raster, Seeds, Messwerte, Ausfälle.

Möglich wird das dadurch, dass auch DummyEngine Regler anbietet -- der Weg von
der Reglerstellung bis in die erzeugte Tonspur ist damit vollständig prüfbar.
"""

from __future__ import annotations

import pytest

from cloney.asr.dummy import DummyASR
from cloney.config import Settings
from cloney.core.compare import MAX_VARIANTS, Comparison, VariantStatus, build_variants
from cloney.core.project import Project, derive_seed
from cloney.core.voices import VoiceStore
from cloney.engines.dummy import DummyEngine
from cloney.engines.f5_german import F5_INFO
from cloney.pipeline import run_comparison

PROBE = "Am 3. Mai 2024 begann es. Dr. Meier sagte z.B. nichts dazu."


def _engine(options: dict[str, float], model: str = ""):  # noqa: ANN202
    return DummyEngine(speed=options.get("speed", 1.0), pitch=options.get("pitch", 0.0))


def _comparison(settings: Settings, grid: dict[str, list[float]]) -> Comparison:
    return Comparison.create(
        name="Tempo",
        text=PROBE,
        voice="test-stimme",
        engine=DummyEngine.info,
        grid=grid,
        comparisons_dir=settings.comparisons_dir,
    )


# -- Raster ----------------------------------------------------------------


def test_raster_ist_das_kreuzprodukt() -> None:
    variants = build_variants(DummyEngine.info, {"speed": [0.8, 1.0], "pitch": [0, 50]})
    assert len(variants) == 4
    assert [v.options for v in variants] == [
        {"speed": 0.8, "pitch": 0.0},
        {"speed": 0.8, "pitch": 50.0},
        {"speed": 1.0, "pitch": 0.0},
        {"speed": 1.0, "pitch": 50.0},
    ]


def test_beschriftung_nennt_nur_was_sich_aendert() -> None:
    variants = build_variants(DummyEngine.info, {"speed": [0.8, 1.2], "pitch": [50]})
    # Der Grundton ist überall gleich -- er gehört in die Zeile, nicht in die
    # Beschriftung, sonst steht in jeder Spalte dasselbe.
    assert [v.label for v in variants] == ["Sprechtempo 0.8", "Sprechtempo 1.2"]
    assert all(v.options["pitch"] == 50.0 for v in variants)


def test_unbekannte_regler_und_ausreisser_fallen_weg() -> None:
    variants = build_variants(F5_INFO, {"speed": [99.0, -3.0], "erfunden": [1.0]})
    # 99 und -3 landen beide an ihrer Grenze und sind danach je einmal da.
    assert [v.options for v in variants] == [{"speed": 1.5}, {"speed": 0.5}]


def test_doppelte_werte_ergeben_keine_doppelten_zeilen() -> None:
    variants = build_variants(DummyEngine.info, {"speed": [1.0, 1.0, 1.0]})
    assert len(variants) == 1


def test_raster_ohne_werte_wird_abgelehnt(settings: Settings) -> None:
    with pytest.raises(ValueError, match="mindestens zwei Varianten"):
        _comparison(settings, {})


def test_eine_einzelne_variante_ist_kein_vergleich(settings: Settings) -> None:
    """Seit die Regler mit ihrer Vorgabe vorbelegt sind, entsteht das leicht aus
    Versehen: alle Achsen auf einem Wert, und es gäbe nichts zu vergleichen."""
    with pytest.raises(ValueError, match="mindestens zwei Varianten"):
        _comparison(settings, {"speed": [1.0]})


def test_raster_ist_gedeckelt() -> None:
    """Ein volles Kreuzprodukt liefe sonst stundenlang, ohne dass es jemand
    beabsichtigt hätte."""
    variants = build_variants(
        DummyEngine.info,
        {"speed": [0.5, 0.7, 0.9, 1.1, 1.3], "pitch": [0, 20, 40, 60, 80]},
    )
    assert len(variants) == MAX_VARIANTS


# -- Lauf ------------------------------------------------------------------


def test_vergleich_misst_jede_variante(settings: Settings, voice_store: VoiceStore) -> None:
    comparison = _comparison(settings, {"speed": [0.6, 1.4]})

    run_comparison(comparison, settings, voice_store, _engine, DummyASR)

    assert comparison.is_complete
    for variant in comparison.variants:
        assert variant.status == VariantStatus.DONE
        assert variant.median_cer == 0.0
        assert variant.duration_s and variant.duration_s > 0.0
        assert comparison.variant_project(variant.slug).output_path.exists()

    langsam, schnell = comparison.variants
    assert langsam.duration_s > schnell.duration_s
    # Das Sprechtempo der fertigen Spur ist die Zahl, an der "zu schnell" hängt.
    assert comparison.chars_per_second(schnell.slug) > comparison.chars_per_second(langsam.slug)


def test_alle_varianten_teilen_dieselben_seeds(settings: Settings, voice_store: VoiceStore) -> None:
    """Sonst unterschieden sich zwei Zeilen in zwei Dingen zugleich -- Regler
    und Zufall -- und die Tabelle beantwortete nicht mehr die gestellte Frage."""
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})

    run_comparison(comparison, settings, voice_store, _engine, DummyASR)

    projects = [comparison.variant_project(v.slug) for v in comparison.variants]
    assert projects[0].id != projects[1].id
    assert [c.seed for c in projects[0].chunks] == [c.seed for c in projects[1].chunks]
    assert projects[0].chunks[0].seed == derive_seed(comparison.id, 0, 0)


def test_vergleich_wuerfelt_nicht_nach(settings: Settings, voice_store: VoiceStore) -> None:
    """Ein Wiederholungsversuch vergäbe einen neuen Seed und verwischte genau
    das, was gemessen werden soll."""
    comparison = _comparison(settings, {"speed": [0.9, 1.1]})
    strenge = settings.model_copy(update={"max_retries": 3, "cer_threshold": -1.0})

    run_comparison(comparison, strenge, voice_store, _engine, DummyASR)

    for variante in comparison.variants:
        project = comparison.variant_project(variante.slug)
        assert all(c.attempts == 0 for c in project.chunks)


def test_gescheiterte_variante_bricht_den_vergleich_nicht_ab(
    settings: Settings, voice_store: VoiceStore
) -> None:
    from cloney.engines.base import EngineError

    def launisch(options: dict[str, float], model: str = ""):  # noqa: ANN202
        if options.get("speed") == 0.6:
            raise EngineError("Modell nicht geladen")
        return _engine(options)

    comparison = _comparison(settings, {"speed": [0.6, 1.4]})
    run_comparison(comparison, settings, voice_store, launisch, DummyASR)

    kaputt, heil = comparison.variants
    assert kaputt.status == VariantStatus.FAILED
    assert "Modell nicht geladen" in kaputt.error
    assert heil.status == VariantStatus.DONE
    assert heil.median_cer == 0.0


def test_fortsetzen_rendert_nur_das_offene(settings: Settings, voice_store: VoiceStore) -> None:
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)
    vorher = comparison.variant_project(comparison.variants[0].slug).chunks[0].audio_file

    def verweigert(options: dict[str, float], model: str = ""):  # noqa: ANN202
        raise AssertionError("Eine fertige Variante darf nicht noch einmal laufen")

    run_comparison(comparison, settings, voice_store, verweigert, DummyASR)
    assert comparison.variant_project(comparison.variants[0].slug).chunks[0].audio_file == vorher


# -- Auswertung und Manifest -----------------------------------------------


def test_spitzenreiter_je_spalte(settings: Settings, voice_store: VoiceStore) -> None:
    """Bewusst je Spalte statt als Gesamtnote: die Gewichtung von Fehlerrate
    gegen Ähnlichkeit kann niemand belegen."""
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)

    a, b = comparison.variants
    a.median_cer, a.median_similarity = 0.05, 0.70
    b.median_cer, b.median_similarity = 0.02, 0.65
    assert comparison.best_cer() == {b.slug}
    assert comparison.best_similarity() == {a.slug}


def test_gleichstand_ueberall_markiert_niemanden(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Eine Auszeichnung, die jede Zeile trägt, sagt nichts aus -- und eine, die
    bei Gleichstand die erste Zeile trifft, sagt etwas Falsches."""
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)

    assert [v.median_cer for v in comparison.variants] == [0.0, 0.0]
    assert comparison.best_cer() == set()


def test_ohne_messwerte_gibt_es_keinen_spitzenreiter(settings: Settings) -> None:
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    assert comparison.best_cer() == set()
    assert comparison.best_similarity() == set()


def test_manifest_uebersteht_den_neustart(settings: Settings, voice_store: VoiceStore) -> None:
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)

    geladen = Comparison.load(comparison.root)
    assert geladen.is_complete
    assert geladen.text == PROBE
    assert geladen.variants[0].options == {"speed": 0.8}
    assert Comparison.list_all(settings.comparisons_dir)[0].id == comparison.id


def test_kennung_darf_nicht_aus_dem_datenverzeichnis_zeigen(settings: Settings) -> None:
    with pytest.raises(ValueError, match="Ungültige Vergleichskennung"):
        Comparison.resolve(settings.comparisons_dir, "../../etc")


def test_loeschen_raeumt_die_varianten_mit_weg(settings: Settings, voice_store: VoiceStore) -> None:
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)
    root = comparison.root

    comparison.delete()
    assert not root.exists()
    assert Comparison.list_all(settings.comparisons_dir) == []


def test_varianten_liegen_beim_vergleich_nicht_bei_den_projekten(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Ein Vergleich mit acht Varianten dürfte die Projektliste nicht fluten."""
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)

    assert Project.list_all(settings.projects_dir) == []
    assert len(list(comparison.variants_dir.iterdir())) == 2


# -- Modelle gegeneinander --------------------------------------------------


def test_modelle_werden_zu_einer_achse_des_vergleichs() -> None:
    """Die Frage nach einem Finetune lautet: ist er besser als der Pretrain?
    Ein leerer Name steht für den Pretrain."""
    variants = build_variants(DummyEngine.info, {"speed": [1.0]}, models=["", "anna-ft"])

    assert [v.model for v in variants] == ["", "anna-ft"]
    assert [v.label for v in variants] == ["Pretrain", "anna-ft"]


def test_modelle_und_regler_ergeben_das_kreuzprodukt() -> None:
    variants = build_variants(DummyEngine.info, {"speed": [0.8, 1.2]}, models=["", "anna-ft"])

    assert len(variants) == 4
    assert variants[0].label == "Pretrain · Sprechtempo 0.8"
    assert variants[-1].label == "anna-ft · Sprechtempo 1.2"


def test_ein_einzelnes_modell_ist_keine_achse() -> None:
    """Sonst stünde in jeder Beschriftung derselbe Name."""
    variants = build_variants(DummyEngine.info, {"speed": [0.8, 1.2]}, models=["anna-ft"])

    assert all(v.model == "anna-ft" for v in variants)
    assert variants[0].label == "Sprechtempo 0.8"


def test_doppelt_genannter_stand_ergibt_eine_zeile() -> None:
    """Zwei gleiche Zeilen kosteten nur Rechenzeit -- dieselbe Regel wie bei
    doppelten Reglerwerten. Im Web ist der Pretrain ankreuzbar, in der CLI
    mehrfach nennbar; beide Wege führen hierher."""
    variants = build_variants(DummyEngine.info, {}, models=["", "anna-ft", ""])

    assert [v.label for v in variants] == ["Pretrain", "anna-ft"]


def test_vergleich_nur_ueber_modelle_braucht_keine_regler() -> None:
    variants = build_variants(DummyEngine.info, {}, models=["", "anna-ft"])

    assert [v.label for v in variants] == ["Pretrain", "anna-ft"]


def test_das_modell_erreicht_die_engine(settings: Settings, voice_store: VoiceStore) -> None:
    """Ohne diese Durchreichung würde jede Variante gegen denselben Stand
    rendern und die Tabelle wäre eine Zeile Zufall je Modell."""
    gesehen: list[str] = []

    def merkend(options: dict[str, float], model: str = ""):  # noqa: ANN202
        gesehen.append(model)
        return _engine(options)

    comparison = Comparison.create(
        name="Stände",
        text=PROBE,
        voice="test-stimme",
        engine=DummyEngine.info,
        grid={"speed": [1.0]},
        comparisons_dir=settings.comparisons_dir,
        models=["", "anna-ft"],
    )
    run_comparison(comparison, settings, voice_store, merkend, DummyASR)

    assert gesehen == ["", "anna-ft"]


# -- Emotionslage als Achse -------------------------------------------------


def test_lagen_werden_zur_achse(settings: Settings) -> None:
    """Dieselben Sätze gegen verschiedene Aufnahmen derselben Stimme."""
    variants = build_variants(DummyEngine.info, {"speed": [1.0]}, lagen=["neutral", "ernst"])

    assert [v.lage for v in variants] == ["neutral", "ernst"]
    assert [v.label for v in variants] == ["neutral", "ernst"]


def test_lage_und_regler_kreuzen_sich(settings: Settings) -> None:
    variants = build_variants(DummyEngine.info, {"speed": [0.8, 1.2]}, lagen=["neutral", "ernst"])

    assert len(variants) == 4
    assert {(v.lage, v.options["speed"]) for v in variants} == {
        ("neutral", 0.8),
        ("neutral", 1.2),
        ("ernst", 0.8),
        ("ernst", 1.2),
    }


def test_eine_einzelne_lage_ist_keine_achse(settings: Settings) -> None:
    """Sie stünde sonst in jeder Beschriftung, ohne etwas zu unterscheiden."""
    variants = build_variants(DummyEngine.info, {"speed": [0.8, 1.2]}, lagen=["ernst"])

    assert all(v.lage == "ernst" for v in variants)
    assert all("ernst" not in v.label for v in variants)


def test_doppelt_genannte_lage_ergibt_keine_zweite_zeile(settings: Settings) -> None:
    variants = build_variants(
        DummyEngine.info, {"speed": [1.0]}, lagen=["ernst", "ernst", "neutral"]
    )

    assert len(variants) == 2


def test_die_lage_landet_auf_allen_saetzen(settings: Settings, voice_store: VoiceStore) -> None:
    """Verglichen wird eine Aufnahme gegen eine andere, nicht ein Satz gegen
    den nächsten -- die Lage gehört deshalb zur Variante."""
    comparison = Comparison.create(
        name="Lagen",
        text=PROBE,
        voice="test-stimme",
        engine=DummyEngine.info,
        grid={"speed": [1.0]},
        comparisons_dir=settings.comparisons_dir,
        lagen=["neutral", "ernst"],
    )

    project = comparison.prepare(comparison.variants[1].slug, DummyEngine.info, 8.0)

    assert {c.lage for c in project.chunks} == {"ernst"}


def test_slugs_bleiben_eindeutig(settings: Settings) -> None:
    """Mit drei Achsen werden Beschriftungen lang, und zwei können hinter der
    Kürzung auf vierzig Zeichen gleich aussehen."""
    variants = build_variants(
        DummyEngine.info,
        {"speed": [0.8, 1.2]},
        models=[
            "ein-sehr-langer-name-fuer-einen-trainierten-stand-eins",
            "ein-sehr-langer-name-fuer-einen-trainierten-stand-zwei",
        ],
    )

    slugs = [v.slug for v in variants]
    assert len(set(slugs)) == len(slugs)


# -- Zuschnitt ändern -------------------------------------------------------


def _geaendert(comparison: Comparison, **abweichung) -> dict[str, int]:
    argumente = {
        "name": comparison.name,
        "text": comparison.text,
        "voice": comparison.voice,
        "engine": DummyEngine.info,
        "grid": {"speed": [0.8, 1.2]},
    }
    argumente.update(abweichung)
    return comparison.reconfigure(**argumente)


def test_ein_nachgetragener_wert_kostet_nur_die_neue_zeile(
    settings: Settings, voice_store: VoiceStore
) -> None:
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)

    bericht = _geaendert(comparison, grid={"speed": [0.8, 1.2, 1.4]})

    assert bericht == {"behalten": 2, "neu": 1, "entfernt": 0}
    assert [v.is_done for v in comparison.variants] == [True, True, False]


def test_ein_anderer_text_verwirft_alle_ergebnisse(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Die Zahlen einer Zeile gelten für genau diese Probe."""
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)

    bericht = _geaendert(comparison, text="Ein ganz anderer Satz.")

    assert bericht == {"behalten": 0, "neu": 2, "entfernt": 2}
    assert not any(v.is_done for v in comparison.variants)


def test_weggefallene_varianten_nehmen_ihren_ton_mit(
    settings: Settings, voice_store: VoiceStore
) -> None:
    """Er gehört zu einer Zeile, die es nicht mehr gibt."""
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})
    run_comparison(comparison, settings, voice_store, _engine, DummyASR)
    ordner = comparison.variant_root(comparison.variants[1].slug)
    assert ordner.exists()

    _geaendert(comparison, grid={"speed": [0.8, 1.4]})

    assert not ordner.exists()


def test_aendern_auf_eine_einzige_variante_wird_abgelehnt(
    settings: Settings, voice_store: VoiceStore
) -> None:
    comparison = _comparison(settings, {"speed": [0.8, 1.2]})

    with pytest.raises(ValueError, match="mindestens zwei Varianten"):
        _geaendert(comparison, grid={"speed": [1.0]})

    # Und der Vergleich steht unverändert da, statt halb geändert.
    assert len(Comparison.load(comparison.root).variants) == 2
