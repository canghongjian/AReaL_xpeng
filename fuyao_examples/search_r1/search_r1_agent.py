"""Search-R1 agent for AReaL proxy-mode training.

Replicates the ROLL Search-R1 interaction loop:
  1. Format question with prompt + tool instruction
  2. LLM generates with stop=["</search>"]
  3. Parse <search> tag -> call retriever HTTP API
  4. Inject results as tool-response user message
  5. Repeat up to max_turns / max_tool_uses
  6. Compute reward via exact-match only
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import aiohttp

from fuyao_examples.search_r1.reward import search_r1_reward

APPLY_PROMPT_TEMPLATE = (
    "For any question, always reason through your thought process using:\n"
    "<think> your reasoning here </think>\n"
    "Then, provide your final answer using:\n"
    "<answer> your answer here </answer>\n\n"
    "Question: {question}\n"
)

TOOL_INSTRUCTION = (
    "You have access to a search engine to help answer questions.\n\n"
    "Additional instructions:\n"
    "- If your initial reasoning in <think> shows you lack some knowledge, "
    "explain what you need to find next inside a new <think> block.\n"
    "- Then issue a search query using:\n"
    "  <search> your query here </search>\n"
    "- The search engine will provide results inside:\n"
    "  <information> ... </information>\n"
    "- You may repeat the <think> and <search> steps as many times as needed.\n"
    "- When you are ready, give your final answer in:\n"
    "  <answer> your answer here </answer>"
)

DEFAULT_SYSTEM_PROMPT = "You're a helpful assistant."

MAX_TOOL_USES_EXCEEDED_MSG = (
    "\n\nReached the maximum number of tool use. "
    "Please output final answer directly."
)


def _build_tool_response(observation: str) -> str:
    """Format tool observations to match the ROLL Search-R1 trajectory shape."""
    return f"<tool_response>\n\n<information>{observation}</information>\n\n</tool_response>"


async def call_retriever(
    query: str,
    search_url: str,
    topk: int = 3,
    timeout: int = 5,
) -> str:
    """POST to the retriever service and format results."""
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                search_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                result = json.loads(await resp.text())["result"]
                return _passages_to_string(result)
    except Exception as exc:
        return f"[SearchTool Error: {exc}]"



def _passages_to_string(result: list[dict]) -> str:
    """Format retriever results identically to ROLL XpengSearchTool."""
    parts: list[str] = []
    for idx, doc_item in enumerate(result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        parts.append(f"Doc {idx + 1}(Title: {title}) {text}")
    return "\n".join(parts)



def _parse_search_query(text: str) -> str | None:
    """Extract the first ``<search>...</search>`` content."""
    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _restore_search_stop_tag(text: str) -> str:
    """Restore ``</search>`` only when the backend stripped the stop token."""
    if "<search>" in text and "</search>" not in text:
        return text + "</search>"
    return text



def _build_user_prompt(question: str) -> str:
    """Build the initial Search-R1 user prompt from a raw question."""
    return f"{APPLY_PROMPT_TEMPLATE.format(question=question)}\n{TOOL_INSTRUCTION}"


class SearchR1Agent:
    """Proxy-mode Search-R1 agent.

    Parameters are injected via ``workflow_kwargs`` in the training config.
    """

    def __init__(
        self,
        search_url: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        input_key: str = "question",
        topk: int = 3,
        max_turns: int = 10,
        max_tool_uses: int = 2,
        question_key: str = "question",
        answer_key: str = "golden_answers",
        **kwargs: Any,
    ):
        self.search_url = search_url or os.environ.get(
            "RETRIEVAL_ENDPOINT", os.environ.get("SEARCH_URL", "")
        )
        self.system_prompt = system_prompt
        self.input_key = input_key
        self.topk = topk
        self.max_turns = max_turns
        self.max_tool_uses = max_tool_uses
        self.question_key = question_key
        self.answer_key = answer_key
        self.extra_kwargs = kwargs

    async def run(self, data: dict, **extra_kwargs: Any) -> float:
        """Run one episode and return a scalar reward."""
        from openai import AsyncOpenAI

        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        question = data[self.question_key]
        golden_answers = data[self.answer_key]
        if self.input_key == self.question_key:
            user_content = _build_user_prompt(question)
        else:
            user_content = data[self.input_key]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        tool_use_attempts = 0
        successful_tool_uses = 0

        for _turn in range(self.max_turns):
            completion = await client.chat.completions.create(
                messages=messages,
                model="default",
                stop=["</search>"],
                **self._gen_kwargs(),
            )

            assistant_text = completion.choices[0].message.content or ""
            full_assistant_text = _restore_search_stop_tag(assistant_text)
            search_query = _parse_search_query(full_assistant_text)

            if search_query is not None and tool_use_attempts < self.max_tool_uses:
                messages.append({"role": "assistant", "content": full_assistant_text})

                tool_use_attempts += 1
                search_results = await call_retriever(
                    query=search_query,
                    search_url=self.search_url,
                    topk=self.topk,
                )
                if "SearchTool Error" not in search_results:
                    successful_tool_uses += 1

                observation = search_results
                if tool_use_attempts >= self.max_tool_uses:
                    observation += MAX_TOOL_USES_EXCEEDED_MSG

                messages.append(
                    {"role": "user", "content": _build_tool_response(observation)}
                )
            else:
                messages.append({"role": "assistant", "content": assistant_text})
                break

            if "<answer>" in assistant_text and "</answer>" in assistant_text:
                break

        full_trajectory_text = "\n".join(message["content"] for message in messages)
        reward = search_r1_reward(
            completions=full_trajectory_text,
            golden_answers=golden_answers,
            tool_use_attempts=tool_use_attempts,
            successful_tool_uses=successful_tool_uses,
        )
        from areal import workflow_context
        from areal.utils import stats_tracker

        scope = workflow_context.stat_scope()
        tracker = stats_tracker.get(scope)
        tracker.scalar(reward=reward)
        tracker.scalar(tool_use_attempts=float(tool_use_attempts))
        tracker.scalar(successful_tool_uses=float(successful_tool_uses))
        tracker.scalar(
            tool_use_rate=float(tool_use_attempts > 0),
            tool_success_rate=(
                float(successful_tool_uses / tool_use_attempts)
                if tool_use_attempts > 0
                else 0.0
            ),
            answer_after_search_rate=float(
                tool_use_attempts > 0 and "<answer>" in full_trajectory_text
            ),
            answer_without_search_rate=float(
                tool_use_attempts == 0 and "<answer>" in full_trajectory_text
            ),
        )
        return reward

    def _gen_kwargs(self) -> dict[str, Any]:
        """Generation kwargs passed to chat.completions.create."""
        kwargs: dict[str, Any] = {}
        for key in ("temperature", "top_p", "max_completion_tokens"):
            if key in self.extra_kwargs:
                kwargs[key] = self.extra_kwargs[key]
        return kwargs
