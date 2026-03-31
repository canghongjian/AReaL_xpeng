"""Fuyao reward function wrappers.

Wraps areal's gsm8k_reward_fn with threaded-env fallback for math_verify.
"""

from math_verify.grader import verify as math_verify_grader
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify.parser import parse as math_verify_parse

from areal.utils import logging

logger = logging.getLogger("FuyaoReward")

_THREADED_ENV_ERROR = (
    "Math-Verify 'parse' function doesn't support threaded environment"
)

_GOLD_TARGET = (
    ExprExtractionConfig(try_extract_without_anchor=True),
    LatexExtractionConfig(),
)
_PRED_TARGET = (
    ExprExtractionConfig(try_extract_without_anchor=True),
    LatexExtractionConfig(),
)
_PRECISION = 6


def _verify_without_timeout(response: str, ground_truth: str) -> float:
    extracted_predictions = math_verify_parse(response, _PRED_TARGET, parsing_timeout=None)
    extracted_golds = math_verify_parse(ground_truth, _GOLD_TARGET, parsing_timeout=None)
    if not extracted_golds or not extracted_predictions:
        return 0.0
    matched = any(
        math_verify_grader(gold, pred, float_rounding=_PRECISION, timeout_seconds=None)
        for gold in extracted_golds
        for pred in extracted_predictions
    )
    return 1.0 if matched else 0.0


def math_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs) -> float:
    """Math reward with threaded-env fallback.

    Drop-in replacement for areal.reward.gsm8k.gsm8k_reward_fn.
    """
    try:
        from areal.reward import get_math_verify_worker

        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception as exc:
        if _THREADED_ENV_ERROR in str(exc):
            return _verify_without_timeout(str(completions), str(answer))
        logger.warning("Exception in math_reward_fn", exc_info=True)
        return 0.0
