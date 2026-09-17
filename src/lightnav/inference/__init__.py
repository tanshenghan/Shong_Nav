"""Inference core: model loading, HF / in-process vLLM engines, frame buffer, ViT cache.

Names are resolved lazily so importing this package does not pull in torch or
transformers until an engine is actually built.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "InferenceConfig": ("lightnav.inference.config", "InferenceConfig"),
    "build_engine": ("lightnav.inference.engine", "build_engine"),
    "VLNInferenceEngine": ("lightnav.inference.engine", "VLNInferenceEngine"),
    "VitResult": ("lightnav.inference.engine", "VitResult"),
    "ModelBundle": ("lightnav.inference.model", "ModelBundle"),
    "BetaModel": ("lightnav.inference.beta_model", "BetaModel"),
    "BetaModelConfig": ("lightnav.inference.beta_model", "BetaModelConfig"),
    "BetaTrainBatch": ("lightnav.inference.beta_model", "BetaTrainBatch"),
    "load_beta_model": ("lightnav.inference.beta_model", "load_beta_model"),
    "NavigationPolicy": ("lightnav.inference.policies", "NavigationPolicy"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
