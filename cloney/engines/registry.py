"""Auflösung von Engine-Namen zu Instanzen."""

from __future__ import annotations

from collections.abc import Callable

from cloney.config import Settings
from cloney.engines.base import EngineInfo, TTSEngine
from cloney.engines.dummy import DummyEngine
from cloney.engines.f5_german import F5_INFO, F5GermanEngine
from cloney.engines.higgs import HIGGS_INFO, HiggsEngine


def _make_higgs(settings: Settings) -> TTSEngine:
    return HiggsEngine(
        base_url=settings.higgs_base_url,
        model=settings.higgs_model,
        timeout_s=settings.higgs_timeout_s,
        reference_mode=settings.higgs_reference_mode,
        temperature=settings.higgs_temperature,
        top_k=settings.higgs_top_k,
        max_new_tokens=settings.higgs_max_new_tokens,
    )


def _make_f5(settings: Settings) -> TTSEngine:
    return F5GermanEngine(
        model_config=settings.f5_model_config,
        repo_id=settings.f5_repo_id,
        ckpt_filename=settings.f5_ckpt_filename,
        vocab_filename=settings.f5_vocab_filename,
        ckpt_path=settings.f5_ckpt_path,
        vocab_path=settings.f5_vocab_path,
        device=settings.f5_device,
        nfe_step=settings.f5_nfe_step,
        cfg_strength=settings.f5_cfg_strength,
        speed=settings.f5_speed,
    )


_FACTORIES: dict[str, Callable[[Settings], TTSEngine]] = {
    "dummy": lambda _: DummyEngine(),
    "higgs": _make_higgs,
    "f5-de": _make_f5,
}

_INFOS: dict[str, EngineInfo] = {
    "dummy": DummyEngine.info,
    "higgs": HIGGS_INFO,
    "f5-de": F5_INFO,
}


def available_engines() -> list[EngineInfo]:
    return list(_INFOS.values())


def engine_info(name: str) -> EngineInfo:
    try:
        return _INFOS[name]
    except KeyError:
        available = ", ".join(_INFOS)
        raise ValueError(f"Unbekannte Engine '{name}'. Verfügbar: {available}") from None


def create_engine(name: str, settings: Settings) -> TTSEngine:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        available = ", ".join(_FACTORIES)
        raise ValueError(f"Unbekannte Engine '{name}'. Verfügbar: {available}") from None
    return factory(settings)
