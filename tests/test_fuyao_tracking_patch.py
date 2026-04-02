import importlib
import sys
import types


class DummyStatsLogger:
    def __init__(self):
        self.calls = []

    def commit(self, epoch, step, global_step, data):
        self.calls.append((epoch, step, global_step, data))
        return data


def test_apply_tracking_patch_adds_deepinsight_metrics(monkeypatch):
    """Tracking patch should duplicate mapped metrics without removing originals."""
    fake_logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    fake_logging_module = types.SimpleNamespace(
        getLogger=lambda *_args, **_kwargs: fake_logger
    )
    fake_areal = types.ModuleType("areal")
    fake_areal_utils = types.ModuleType("areal.utils")
    fake_areal_utils.logging = fake_logging_module
    fake_areal.utils = fake_areal_utils

    monkeypatch.setitem(sys.modules, "areal", fake_areal)
    monkeypatch.setitem(sys.modules, "areal.utils", fake_areal_utils)
    fake_module = types.ModuleType("areal.utils.stats_logger")
    fake_module.StatsLogger = DummyStatsLogger
    monkeypatch.setitem(sys.modules, "areal.utils.stats_logger", fake_module)

    sys.modules.pop("fuyao_examples.tracking_patch", None)
    tracking_patch = importlib.import_module("fuyao_examples.tracking_patch")

    tracking_patch.apply_tracking_patch()

    logger = DummyStatsLogger()
    payload = {
        "ppo_actor/task_reward/avg": 1.25,
        "ppo_actor/update/actor_loss/avg": 0.4,
        "unmapped_metric": 7.0,
    }

    logger.commit(0, 0, 0, payload)

    committed = logger.calls[0][3]
    assert committed["ppo_actor/task_reward/avg"] == 1.25
    assert committed["deepinsight_algorithm/reward"] == 1.25
    assert committed["deepinsight_algorithm/policy_loss"] == 0.4
    assert committed["unmapped_metric"] == 7.0
