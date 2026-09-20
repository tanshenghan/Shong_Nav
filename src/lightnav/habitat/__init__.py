"""Habitat evaluation client: remote env transport, velocity policy, reporting and runner.

Submodules are imported lazily so that ``import lightnav.habitat`` does not
pull in pyzmq or the inference stack until a name is actually used.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "RemoteEnvClient": "lightnav.habitat.remote_env",
    "TrajVocabVLNCEPolicy": "lightnav.habitat.policy",
    "extract_instruction": "lightnav.habitat.policy",
    "select_action_waypoint": "lightnav.habitat.policy",
    "make_json_safe": "lightnav.habitat.results",
    "print_objectnav_summary": "lightnav.habitat.results",
    "print_vlnce_summary": "lightnav.habitat.results",
    "HabitatEvalConfig": "lightnav.habitat.runner",
    "resolve_action_decoder": "lightnav.habitat.runner",
    "run_habitat_eval": "lightnav.habitat.runner",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
