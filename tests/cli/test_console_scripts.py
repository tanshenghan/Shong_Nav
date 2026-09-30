from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SCRIPTS = {
    "lightnav-predict": "lightnav.cli.predict:main",
    "lightnav-serve": "lightnav.serving.ws_server:main",
    "lightnav-ws-client": "lightnav.cli.ws_client:main",
    "lightnav-eval-habitat": "lightnav.cli.eval_habitat:main",
    "lightnav-eval-merge": "lightnav.cli.eval_merge:main",
    "lightnav-render": "lightnav.cli.render:main",
}


def test_pyproject_declares_supported_console_scripts():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert data["project"]["scripts"] == EXPECTED_SCRIPTS


@pytest.mark.parametrize("target", sorted(EXPECTED_SCRIPTS.values()))
def test_every_console_script_target_resolves_to_a_callable(target):
    module_name, func_name = target.split(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, func_name))


def test_pytest_config_selects_the_cpu_suite():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    ini = data["tool"]["pytest"]["ini_options"]
    assert ini["testpaths"] == ["tests", "data_pipeline/R2R/tests", "data_pipeline/vlnverse/tests"]
    assert ini["asyncio_mode"] == "auto"
    assert any(m.startswith("gpu") for m in ini["markers"])
    assert any(m.startswith("video") for m in ini["markers"])


def test_habitat_cli_preserves_stop_confirmation_protocol():
    from lightnav.cli.eval_habitat import build_parser, config_from_args

    cfg = config_from_args(build_parser().parse_args([
        "--model_path", "/model", "--action_output", "fm", "--backend", "hf",
        "--beta_checkpoint", "/checkpoint", "--stop_threshold", "0.7",
        "--stop_confirm_steps", "2",
    ]))
    assert cfg.stop_confirm_steps == 2 and cfg.stop_threshold == 0.7
