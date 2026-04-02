import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fuyao_readmes_reference_existing_paths():
    """Fuyao docs should point to the files that actually exist in this repo."""
    expected_files = [
        PROJECT_ROOT / "fuyao_examples/search_r1/search_r1_qwen3_4b.yaml",
        PROJECT_ROOT / "fuyao_examples/code_dapo/code_dapo_qwen3_4b.yaml",
        PROJECT_ROOT / "fuyao_examples/tracking_patch.py",
        PROJECT_ROOT / "fuyao_docs/adaptation_boundary.md",
        PROJECT_ROOT / "fuyao_docs/scenario_matrix.md",
        PROJECT_ROOT / "fuyao_docs/upstream_upgrade_checklist.md",
        PROJECT_ROOT / "areal_fuyao.dockerfile",
    ]

    for path in expected_files:
        assert path.exists(), f"Missing expected Fuyao artifact: {path}"

    fuyao_examples_readme = (PROJECT_ROOT / "fuyao_examples/README.md").read_text()
    assert "fuyao_examples/search_r1/search_r1_qwen3_4b.yaml" in fuyao_examples_readme
    assert "fuyao_examples/code_dapo/code_dapo_qwen3_4b.yaml" in fuyao_examples_readme

    fuyao_docs_readme = (PROJECT_ROOT / "fuyao_docs/README.md").read_text()
    assert "adaptation_boundary.md" in fuyao_docs_readme
    assert "scenario_matrix.md" in fuyao_docs_readme
    assert "upstream_upgrade_checklist.md" in fuyao_docs_readme


def test_fuyao_run_script_help_lists_supported_run_types():
    """The Fuyao launcher should expose the supported run types via --help."""
    result = subprocess.run(
        ["bash", "fuyao_examples/fuyao_areal_run.sh", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "math_rlvr, search_r1, code_dapo" in result.stdout
    assert "--skip-deploy" in result.stdout
    assert "RETRIEVAL_ENDPOINT" in result.stdout


def test_math_reward_fn_uses_worker_and_falls_back_in_threaded_env(monkeypatch):
    fake_logger = types.SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    fake_areal_utils = types.ModuleType("areal.utils")
    fake_areal_utils.logging = types.SimpleNamespace(
        getLogger=lambda *_args, **_kwargs: fake_logger
    )

    class Worker:
        def __init__(self):
            self.calls = 0

        def verify(self, completions, answer):
            self.calls += 1
            if self.calls == 1:
                return 1.0
            raise ValueError(
                "Math-Verify 'parse' function doesn't support threaded environment"
            )

    worker = Worker()
    fake_reward_mod = types.ModuleType("areal.reward")
    fake_reward_mod.get_math_verify_worker = lambda: worker

    monkeypatch.setitem(sys.modules, "areal.utils", fake_areal_utils)
    monkeypatch.setitem(sys.modules, "areal.reward", fake_reward_mod)
    sys.modules.pop("fuyao_examples.reward", None)
    reward_module = importlib.import_module("fuyao_examples.reward")

    monkeypatch.setattr(
        reward_module,
        "_verify_without_timeout",
        lambda completions, answer: 0.5,
    )

    assert reward_module.math_reward_fn("", "x", [], [], "y") == pytest.approx(1.0)
    assert reward_module.math_reward_fn("", "x", [], [], "y") == pytest.approx(0.5)
