"""Vorbereiten und Starten eines Finetunes -- ohne GPU, ohne F5-TTS.

Trainiert wird von F5s eigenen Skripten. Geprüft wird deshalb genau das, was
Cloney beiträgt und was falsch sein kann: das Eingabeformat, das Vokabular, die
Ordner und die berechneten Parameter.
"""

from __future__ import annotations

import csv
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cloney.asr.base import Transcript
from cloney.core.audio import write_wav
from cloney.core.dataset import build_dataset
from cloney.core.finetune import (
    FRAMES_PER_SECOND,
    FinetuneError,
    TrainingPlan,
    check_prepared,
    install_vocab,
    plan_training,
    prepare_command,
    write_f5_metadata,
)

SR = 24000


class FesterASR:
    def transcribe(self, audio, sample_rate, language="de") -> Transcript:  # noqa: ANN001
        return Transcript("Am 3. Mai 2024 begann alles ganz harmlos hier.")

    def close(self) -> None:
        return None


def _datensatz(tmp_path: Path):  # noqa: ANN202
    t = np.arange(int(6.0 * SR), dtype=np.float32) / SR
    huelle = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    rede = (0.3 * huelle * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    stille = np.zeros(int(0.6 * SR), dtype=np.float32)
    write_wav(tmp_path / "lesung.wav", np.concatenate([rede, stille, rede]), SR)
    return build_dataset("Anna liest", [tmp_path / "lesung.wav"], FesterASR(), tmp_path / "ds")


# -- Das Eingabeformat ------------------------------------------------------


def test_metadata_bekommt_kopfzeile_und_absolute_pfade(tmp_path: Path) -> None:
    """prepare_csv_wavs.py verlangt beides ausdrücklich. Cloneys eigene
    metadata.csv führt relative Pfade ohne Kopfzeile, weil das für alles andere
    handlicher ist."""
    dataset = _datensatz(tmp_path)

    ziel = write_f5_metadata(dataset, tmp_path / "f5")

    zeilen = list(csv.reader(ziel.open(encoding="utf-8"), delimiter="|"))
    assert zeilen[0] == ["audio_file", "text"]
    assert len(zeilen) == len(dataset.utterances) + 1
    for zeile in zeilen[1:]:
        assert Path(zeile[0]).is_absolute()
        assert Path(zeile[0]).exists()


def test_metadata_uebernimmt_die_sprechfassung(tmp_path: Path) -> None:
    dataset = _datensatz(tmp_path)
    ziel = write_f5_metadata(dataset, tmp_path / "f5")

    zeilen = list(csv.reader(ziel.open(encoding="utf-8"), delimiter="|"))
    assert "dritten Mai" in zeilen[1][1]


# -- Das Vokabular ----------------------------------------------------------


def test_vokabular_des_pretrains_ersetzt_das_von_f5(tmp_path: Path) -> None:
    """F5 kopiert im Finetune-Zweig sein eigenes, fest eingetragenes Vokabular --
    das des englisch-chinesischen Basismodells. Beim Weitertrainieren eines
    deutschen Modells hätte die Embedding-Matrix dann eine andere Größe als die
    geladenen Gewichte."""
    daten = tmp_path / "data"
    daten.mkdir()
    (daten / "vocab.txt").write_text("falsch\n", encoding="utf-8")
    pretrain = tmp_path / "deutsch-vocab.txt"
    pretrain.write_text("richtig\n", encoding="utf-8")

    install_vocab(pretrain, daten)

    assert (daten / "vocab.txt").read_text(encoding="utf-8") == "richtig\n"


def test_fehlendes_vokabular_wird_benannt(tmp_path: Path) -> None:
    with pytest.raises(FinetuneError, match="Vokabular des Pretrains"):
        install_vocab(tmp_path / "gibtsnicht.txt", tmp_path / "data")


def test_unvollstaendige_vorbereitung_wird_erkannt(tmp_path: Path) -> None:
    """Sonst bricht erst das Training ab -- nach dem Laden des Modells."""
    daten = tmp_path / "data"
    daten.mkdir()
    (daten / "raw.arrow").write_bytes(b"")

    with pytest.raises(FinetuneError, match="duration.json"):
        check_prepared(daten)


# -- Die berechneten Parameter ----------------------------------------------


def _plan(tmp_path: Path, **kw) -> TrainingPlan:  # noqa: ANN003
    dataset = _datensatz(tmp_path)
    return TrainingPlan(
        dataset_name=dataset.root.name,
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "ckpts",
        pretrain_ckpt=tmp_path / "modell.safetensors",
        vocab_path=tmp_path / "vocab.txt",
        total_seconds=dataset.total_seconds,
        **kw,
    )


def test_frames_werden_in_hoerbare_dauer_uebersetzt(tmp_path: Path) -> None:
    """'batch_size_per_gpu = 1600' sagt niemandem etwas. Sekunden Ton je
    Schritt schon -- und daraus ergibt sich die Zahl der Schritte."""
    plan = _plan(tmp_path, batch_frames=1600, epochs=10)

    assert plan.seconds_per_step == pytest.approx(1600 / FRAMES_PER_SECOND)
    assert plan.seconds_per_step == pytest.approx(17.07, abs=0.1)
    assert plan.steps_per_epoch == max(1, round(plan.total_seconds / plan.seconds_per_step))
    assert plan.total_steps == plan.steps_per_epoch * 10


def test_leerer_datensatz_ergibt_keine_division_durch_null(tmp_path: Path) -> None:
    plan = _plan(tmp_path, batch_frames=1600)
    leer = TrainingPlan(
        dataset_name=plan.dataset_name,
        data_dir=plan.data_dir,
        checkpoint_dir=plan.checkpoint_dir,
        pretrain_ckpt=plan.pretrain_ckpt,
        vocab_path=plan.vocab_path,
        total_seconds=0.0,
    )
    assert leer.steps_per_epoch == 1


# -- Der Aufruf -------------------------------------------------------------


def test_trainingsbefehl_traegt_die_tragenden_angaben(tmp_path: Path) -> None:
    plan = _plan(tmp_path, batch_frames=1200, epochs=42)
    befehl = plan.command()

    # Gestartet über den eigenen Starter, der die Worker-Zahl setzt; die
    # Argumente sind unverändert die von F5s finetune_cli.
    assert "cloney.core.train_launcher" in befehl
    assert "--finetune" in befehl
    # Ohne 'custom' nähme F5 sein eigenes Vokabular.
    assert befehl[befehl.index("--tokenizer") + 1] == "custom"
    assert befehl[befehl.index("--tokenizer_path") + 1] == str(plan.vocab_path)
    assert befehl[befehl.index("--pretrain") + 1] == str(plan.pretrain_ckpt)
    # Frames, nicht Beispiele.
    assert befehl[befehl.index("--batch_size_type") + 1] == "frame"
    assert befehl[befehl.index("--batch_size_per_gpu") + 1] == "1200"
    assert befehl[befehl.index("--epochs") + 1] == "42"


def test_aufwaermen_ist_fuer_ein_finetune_bemessen(tmp_path: Path) -> None:
    """F5s Standard von 20000 Aufwärmschritten ist für einen Lauf von Grund auf
    gedacht. Ein Finetune auf eine Stimme wäre vorbei, bevor die Lernrate oben
    angekommen ist."""
    befehl = _plan(tmp_path).command()
    assert int(befehl[befehl.index("--num_warmup_updates") + 1]) <= 1000
    assert int(befehl[befehl.index("--save_per_updates") + 1]) <= 5000


def _mit_dauer(sekunden: float, epochs: int = 100) -> TrainingPlan:
    return TrainingPlan(
        dataset_name="anna",
        data_dir=Path("d"),
        checkpoint_dir=Path("c"),
        pretrain_ckpt=Path("m"),
        vocab_path=Path("v"),
        total_seconds=sekunden,
        batch_frames=1600,
        epochs=epochs,
    )


def test_aufwaermen_verschlingt_nicht_den_ganzen_lauf() -> None:
    """0.6 Minuten Material ergeben rund 200 Schritte. Mit den festen 200
    Aufwärmschritten wäre die Lernrate genau dann oben, wenn das Training
    endet -- der Lauf hätte nie bei der Ziellernrate trainiert."""
    kurz = _mit_dauer(36.0)

    assert kurz.total_steps == pytest.approx(200, abs=20)
    assert kurz.warmup < kurz.total_steps // 5
    assert kurz.warmup >= 10


def test_bei_ausreichendem_material_gilt_die_obergrenze() -> None:
    lang = _mit_dauer(3600.0)
    assert lang.warmup == 200
    assert lang.save_interval == 1000


def test_kurze_laeufe_bekommen_trotzdem_zwischenstaende() -> None:
    """Am Ende sichert F5 ohnehin einmal -- aber gerade die Zwischenstände
    zeigen, ob längeres Training noch etwas bringt."""
    kurz = _mit_dauer(36.0)
    assert kurz.save_interval < kurz.total_steps
    assert kurz.last_interval < kurz.save_interval


def test_vorbereitungsbefehl_uebergibt_die_datei_nicht_den_ordner(tmp_path: Path) -> None:
    """Der Parameter heißt 'inp_dir', das Skript prüft aber auf die Endung .csv
    und liest die Tonpfade unverändert aus der Tabelle. Dem Namen zu folgen
    endet in 'ValueError: input must be a .csv file'."""
    befehl = prepare_command(tmp_path / "f5" / "metadata.csv", tmp_path / "raus")

    assert "f5_tts.train.datasets.prepare_csv_wavs" in befehl
    assert befehl[3].endswith("metadata.csv")
    assert befehl[4] == str(tmp_path / "raus")


def test_vorbereitung_umgeht_die_pruefung_auf_das_emilia_vokabular(tmp_path: Path) -> None:
    """Der Finetune-Zweig von prepare_csv_wavs.py prüft mit einem assert auf
    <f5>/data/Emilia_ZH_EN_pinyin/vocab.txt -- eine Datei aus den
    Trainingsdaten, die bei einer Installation über pip nicht mitkommt. Der
    Aufruf bricht dort ab, bevor er irgendetwas tut.

    '--pretrain' steuert im Skript ausschließlich die Herkunft des Vokabulars,
    sonst ist beides identisch. Das richtige legt install_vocab danach hin.
    """
    assert "--pretrain" in prepare_command(tmp_path / "rein", tmp_path / "raus")


def test_ordnername_kommt_vom_datensatz_nicht_von_der_anzeige(tmp_path: Path) -> None:
    """Der Name wird bei F5 zu einem Ordnernamen. 'Anna liest' mit Leerzeichen
    taugt dafür nicht."""
    dataset = _datensatz(tmp_path)
    plan = plan_training(
        dataset, tmp_path / "m.safetensors", tmp_path / "v.txt", f5_dir=tmp_path / "f5"
    )

    assert plan.dataset_name == "anna-liest"
    assert " " not in plan.dataset_name
    # F5 sucht unter data/<name>_<tokenizer> und schreibt nach ckpts/<name>.
    assert plan.data_dir == tmp_path / "f5" / "data" / "anna-liest_custom"
    assert plan.checkpoint_dir == tmp_path / "f5" / "ckpts" / "anna-liest"


def test_knappes_material_wird_erkannt(tmp_path: Path) -> None:
    """F5s eigene Angabe lautet 10 bis 100 Stunden, die dokumentierten Erfolge
    einzelner Sprecher liegen bei zwölf Stunden. Wer mit Minuten antritt, soll
    das wissen, bevor er auf ein Ergebnis wartet."""
    plan = _plan(tmp_path, batch_frames=1600)
    assert plan.knappes_material

    viel = TrainingPlan(
        dataset_name="anna",
        data_dir=tmp_path / "d",
        checkpoint_dir=tmp_path / "c",
        pretrain_ckpt=tmp_path / "m.safetensors",
        vocab_path=tmp_path / "v.txt",
        total_seconds=3600.0,
    )
    assert not viel.knappes_material


def test_f5_wurzel_laesst_sich_vorgeben(tmp_path: Path) -> None:
    """Ohne installiertes F5-TTS gäbe es sonst keine Pfade -- und wer F5 als
    Auscheckstand statt über pip hat, käme nicht weiter."""
    from cloney.core.finetune import checkpoint_dir_for, data_dir_for

    assert data_dir_for("anna", root=tmp_path) == tmp_path / "data" / "anna_custom"
    assert checkpoint_dir_for("anna", root=tmp_path) == tmp_path / "ckpts" / "anna"


# -- Vom Checkpoint zum auswählbaren Modell ---------------------------------


def test_checkpoints_werden_gefunden_juengster_zuerst(tmp_path: Path) -> None:
    """model_last.pt steht vorn -- es ist der Stand, den man zuerst hören will."""
    from cloney.core.models import find_checkpoints

    for name in ("model_1000.pt", "model_12000.pt", "model_4000.pt", "model_last.pt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "pretrained_base.pt").write_bytes(b"")

    gefunden = [p.name for p in find_checkpoints(tmp_path)]
    assert gefunden == ["model_last.pt", "model_12000.pt", "model_4000.pt", "model_1000.pt"]


def test_modell_zeigt_die_engine_auf_den_trainierten_stand(tmp_path: Path) -> None:
    """Kein neuer Weg in die Engine, sondern derselbe wie bisher: sie liest
    Checkpoint und Vokabular aus der Konfiguration."""
    from cloney.config import Settings
    from cloney.core.models import ModelStore, settings_for

    ckpt = tmp_path / "model_last.pt"
    ckpt.write_bytes(b"")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\n", encoding="utf-8")

    store = ModelStore(tmp_path / "models")
    modell = store.add("anna-ft", ckpt, vocab, note="anna, 62 min")

    settings = Settings(data_dir=tmp_path / "data")
    angepasst = settings_for(modell, settings)
    assert angepasst.f5_ckpt_path == str(ckpt)
    assert angepasst.f5_vocab_path == str(vocab)
    # Ohne Modell bleibt alles, wie es ist.
    assert settings_for(None, settings) is settings


def test_verschobener_checkpoint_wird_benannt(tmp_path: Path) -> None:
    """Sonst startet ein Lauf und scheitert erst beim Laden des Modells."""
    from cloney.config import Settings
    from cloney.core.models import ModelError, ModelStore, settings_for

    ckpt = tmp_path / "model_last.pt"
    ckpt.write_bytes(b"")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\n", encoding="utf-8")
    modell = ModelStore(tmp_path / "models").add("anna-ft", ckpt, vocab)
    ckpt.unlink()

    with pytest.raises(ModelError, match="verschoben oder gelöscht"):
        settings_for(modell, Settings(data_dir=tmp_path / "data"))


def test_eintrag_loeschen_laesst_den_checkpoint_liegen(tmp_path: Path) -> None:
    """Er gehört F5, ist Gigabyte groß, und ihn beim Aufräumen einer Liste
    mitzulöschen wäre eine böse Überraschung."""
    from cloney.core.models import ModelStore

    ckpt = tmp_path / "model_last.pt"
    ckpt.write_bytes(b"x")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\n", encoding="utf-8")
    store = ModelStore(tmp_path / "models")
    store.add("anna-ft", ckpt, vocab)

    store.delete("anna-ft")
    assert not store.exists("anna-ft")
    assert ckpt.exists()


# -- Der Pretrain muss die EMA-Struktur tragen ------------------------------


def test_inferenz_export_wird_erkannt() -> None:
    """Trainer.load_checkpoint ruft ema_model.load_state_dict, bevor der Zweig
    greift, der einen nackten Export behandelt. Der Wrapper kennt nur
    'ema_model....' samt 'initted' und 'step' -- ein Inferenz-Export scheitert
    dort mit einer seitenlangen Liste fehlender Schlüssel."""
    from cloney.core.finetune import needs_ema_wrapper

    assert needs_ema_wrapper(["transformer.proj_out.weight", "transformer.proj_out.bias"])
    assert not needs_ema_wrapper(["initted", "step", "ema_model.transformer.proj_out.weight"])


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ersetzt torch durch eine Attrappe, die über pickle schreibt und liest.

    Zu prüfen ist die Struktur, die Cloney ablegt -- nicht Torchs
    Serialisierung. Torch nur dafür zu installieren kostete die CI ein
    Vielfaches der Laufzeit für nichts.
    """
    modul = types.ModuleType("torch")
    modul.tensor = lambda wert: ("tensor", wert)
    modul.save = lambda obj, pfad: Path(pfad).write_bytes(pickle.dumps(obj))
    modul.load = lambda pfad, **kwargs: pickle.loads(Path(pfad).read_bytes())
    monkeypatch.setitem(sys.modules, "torch", modul)


def test_wrapper_ergaenzt_praefix_und_buchfuehrung(fake_torch: None) -> None:
    from cloney.core.finetune import EMA_BUCHFUEHRUNG, ema_wrapped

    gewrappt = ema_wrapped({"transformer.proj_out.weight": "gewicht"})

    assert set(EMA_BUCHFUEHRUNG) <= set(gewrappt)
    assert "ema_model.transformer.proj_out.weight" in gewrappt
    assert "transformer.proj_out.weight" not in gewrappt


def test_pretrain_wird_nur_bei_bedarf_umgeschrieben(fake_torch: None, tmp_path: Path) -> None:
    """Die offiziellen F5-Checkpoints tragen die Struktur bereits. Sie zu
    kopieren kostete über ein Gigabyte für nichts."""
    import torch

    from cloney.core.finetune import ema_wrapped, write_trainer_pretrain

    passend = tmp_path / "schon_gut.pt"
    torch.save({"ema_model_state_dict": ema_wrapped({"transformer.w": "gewicht"})}, passend)

    assert write_trainer_pretrain(passend, tmp_path / "kopie.pt") == passend
    assert not (tmp_path / "kopie.pt").exists()


def test_nackter_export_wird_ladbar_geschrieben(fake_torch: None, tmp_path: Path) -> None:
    import torch

    from cloney.core.finetune import write_trainer_pretrain

    quelle = tmp_path / "model_420000.pt"
    torch.save({"transformer.proj_out.weight": "gewicht"}, quelle)

    ziel = write_trainer_pretrain(quelle, tmp_path / "pretrain_model_420000.pt")

    assert ziel != quelle
    inhalt = torch.load(ziel, map_location="cpu", weights_only=True)
    # Genau die Struktur, die Trainer.load_checkpoint erwartet.
    assert set(inhalt) == {"ema_model_state_dict"}
    assert {"initted", "step"} <= set(inhalt["ema_model_state_dict"])
    assert "ema_model.transformer.proj_out.weight" in inhalt["ema_model_state_dict"]


# -- Fortsetzen oder neu beginnen -------------------------------------------


def test_leerer_ordner_hat_keine_staende(tmp_path: Path) -> None:
    from cloney.core.finetune import vorhandene_staende

    assert vorhandene_staende(tmp_path / "gibt-es-nicht") == []
    assert vorhandene_staende(tmp_path) == []


def test_vorhandene_staende_werden_gefunden(tmp_path: Path) -> None:
    """F5 entscheidet allein nach den Dateien im Ordner, woher es lädt. Wer das
    nicht weiß, glaubt vom Pretrain aus zu beginnen, und setzt in Wahrheit fort."""
    from cloney.core.finetune import vorhandene_staende

    for name in ("model_last.pt", "model_4000.pt", "beiwerk.txt"):
        (tmp_path / name).write_bytes(b"x")

    assert [p.name for p in vorhandene_staende(tmp_path)] == ["model_4000.pt", "model_last.pt"]


def test_staende_werden_verschoben_statt_geloescht(tmp_path: Path) -> None:
    """Ein Trainingslauf kostet Stunden und Gigabyte. Wer neu beginnt, soll das
    Bisherige nicht verlieren."""
    from cloney.core.finetune import staende_beiseite, vorhandene_staende

    (tmp_path / "model_last.pt").write_bytes(b"gewichte")
    (tmp_path / "model_4000.pt").write_bytes(b"gewichte")

    ziel = staende_beiseite(tmp_path)

    assert ziel is not None and ziel.parent == tmp_path
    assert sorted(p.name for p in ziel.iterdir()) == ["model_4000.pt", "model_last.pt"]
    # Der Ordner ist für F5 jetzt leer -- nur so kommt der Pretrain zum Zug.
    assert vorhandene_staende(tmp_path) == []


def test_ohne_staende_gibt_es_nichts_beiseitezulegen(tmp_path: Path) -> None:
    from cloney.core.finetune import staende_beiseite

    assert staende_beiseite(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


# -- Worker des DataLoaders --------------------------------------------------


@pytest.mark.parametrize(
    ("kerne", "erwartet"),
    [(1, 1), (2, 1), (4, 3), (8, 7), (9, 8), (32, 8)],
)
def test_worker_zahl_richtet_sich_nach_den_kernen(kerne: int, erwartet: int) -> None:
    """F5 legt sechzehn an, auch auf acht Kernen. Einer bleibt hier für den
    Hauptprozess frei -- er füttert die GPU, und wartet er, wartet sie mit."""
    from cloney.core.train_launcher import worker_count

    assert worker_count(kerne) == erwartet


def test_ohne_f5_wird_nichts_gepatcht() -> None:
    """Ein Trainingslauf ist zu teuer, um an einer Bequemlichkeit zu scheitern:
    geht der Patch nicht, läuft das Training mit F5s Vorgabe weiter."""
    from cloney.core.train_launcher import patch_trainer

    assert patch_trainer(4) is False


def test_patch_setzt_die_worker_zahl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gegen ein eingeschleustes Ersatzmodul -- F5 liegt hier nicht vor."""
    import sys
    import types

    gesehen: dict[str, object] = {}

    class Trainer:
        def train(self, train_dataset, num_workers=16, **kwargs):  # noqa: ANN001, ANN202
            gesehen["workers"] = num_workers
            gesehen.update(kwargs)

    modul = types.ModuleType("f5_tts.model.trainer")
    modul.Trainer = Trainer
    monkeypatch.setitem(sys.modules, "f5_tts", types.ModuleType("f5_tts"))
    monkeypatch.setitem(sys.modules, "f5_tts.model", types.ModuleType("f5_tts.model"))
    monkeypatch.setitem(sys.modules, "f5_tts.model.trainer", modul)

    from cloney.core.train_launcher import patch_trainer

    assert patch_trainer(3) is True
    Trainer().train("datensatz", resumable_with_seed=666)

    assert gesehen["workers"] == 3
    # Was F5 sonst übergibt, muss durchkommen -- sonst ginge das Fortsetzen kaputt.
    assert gesehen["resumable_with_seed"] == 666


def test_der_lauf_geht_ueber_den_starter() -> None:
    """Sonst käme F5s Vorgabe von sechzehn Worker-Prozessen zum Zug."""
    from cloney.core.finetune import TrainingPlan

    plan = TrainingPlan(
        dataset_name="anna",
        data_dir=Path("data"),
        checkpoint_dir=Path("ckpts"),
        pretrain_ckpt=Path("model.pt"),
        vocab_path=Path("vocab.txt"),
    )
    befehl = plan.command()

    assert "cloney.core.train_launcher" in befehl
    assert "f5_tts.train.finetune_cli" not in befehl
