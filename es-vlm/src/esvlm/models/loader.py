"""Load one model per process. Offline evals must fail fast if weights miss."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esvlm.models.registry import ModelSpec, get_spec, resolve_local_dir
from esvlm.types import DeviceSpec
from esvlm.utils.device import detect_device, torch_device, torch_dtype


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    processor: Any
    device: DeviceSpec
    source: str


ADAPTERS = {
    "qwen_vl": "esvlm.models.adapters.qwen_vl",
    "gemma4": "esvlm.models.adapters.gemma4",
    "molmo2": "esvlm.models.adapters.molmo2",
    "internvl": "esvlm.models.adapters.internvl",
}


def load_model(
    key: str,
    *,
    models_dir: Path | None = None,
    revision: str | None = None,
    prefer_device: str | None = None,
    offline: bool = False,
) -> LoadedModel:
    spec = get_spec(key)
    if revision and revision != spec.revision:
        spec = ModelSpec(
            key=spec.key,
            hf_id=spec.hf_id,
            revision=revision,
            env=spec.env,
            adapter=spec.adapter,
            on_cluster=spec.on_cluster,
            notes=spec.notes,
        )
    device = detect_device(prefer=prefer_device)
    local = resolve_local_dir(spec, models_dir)
    source = str(local) if local is not None else spec.hf_id
    if offline and local is None:
        raise FileNotFoundError(
            f"offline load requested but {spec.key} has no config.json under "
            f"{models_dir}. Fetch with slurm/fetch_weights.sbatch or set "
            "ESVLM_MODELS_DIR."
        )
    adapter = _import_adapter(spec.adapter)
    model, processor = adapter.load(
        source,
        revision=spec.revision if local is None else None,
        device=torch_device(device),
        dtype=torch_dtype(device),
        trust_remote_code=True,
    )
    return LoadedModel(spec=spec, model=model, processor=processor, device=device, source=source)


def residual_modules(loaded: LoadedModel) -> list[Any]:
    adapter = _import_adapter(loaded.spec.adapter)
    return adapter.residual_modules(loaded.model)


def _import_adapter(name: str) -> Any:
    import importlib

    if name not in ADAPTERS:
        raise KeyError(name)
    return importlib.import_module(ADAPTERS[name])
