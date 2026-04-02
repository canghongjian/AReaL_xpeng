"""Reward functions for Search-R1 training."""

from __future__ import annotations

import re
import string
from typing import Any

# ---------------------------------------------------------------------------
# Answer normalisation (from GEM qa_em.py)
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Lower-case, remove articles / punctuation, collapse whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))



def em_check(prediction: str, golden_answers: list[str] | str) -> bool:
    """Exact-match check after normalisation."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    return any(
        normalize_answer(str(ans)) == normalized_prediction for ans in golden_answers
    )


# ---------------------------------------------------------------------------
# Answer extraction (from GEM parsing.py)
# ---------------------------------------------------------------------------


def extract_last_tagged_answer(model_response: str) -> str | None:
    """Extract the last ``<answer>...</answer>`` content from *model_response*."""
    idx = model_response.rfind("<answer>")
    if idx < 0:
        return None
    end_idx = len(model_response)
    while True:
        next_end_tag_idx = model_response.rfind("</answer>", 0, end_idx)
        if next_end_tag_idx < idx:
            break
        end_idx = next_end_tag_idx
    return model_response[idx + len("<answer>") : end_idx].strip()


def search_r1_reward(
    completions: str,
    golden_answers: list[str] | str,
    **_kwargs: Any,
) -> float:
    """Reward exact-match answers only."""
    model_answer = extract_last_tagged_answer(completions)
    if model_answer is not None and em_check(model_answer, golden_answers):
        return 1.0
    return 0.0
