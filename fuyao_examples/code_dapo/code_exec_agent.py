"""Code Execution (DAPO) agent for AReaL proxy mode.

Agent loop:
  1. LLM generates text with stop=["</code>"]
  2. Parse <code>...</code> and execute Python code
  3. Inject output as user message
  4. Repeat until \\boxed{answer} or max_turns
  5. Return reward (math verify)
"""

import asyncio
import os
import re
import subprocess
import tempfile
from typing import Any

from openai import AsyncOpenAI

from fuyao_examples.reward import math_verify_with_fallback

from areal import workflow_context
from areal.utils import logging, stats_tracker

logger = logging.getLogger("CodeExecAgent")

CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
MAX_OUTPUT_CHARS = 1024
PYTHON_CODE_TOOL_INSTRUCTION = (
    "Initially, when solving a question, you would need to think step by step, "
    "without the ability to use code for calculation. "
    "Now, you have the capability to write code to use the code interpreter for calculation. "
    "The code will be executed by a sandbox, and the result can be returned to enhance "
    "your reasoning process. your calculation while still maintaining the reasoning process. "
    "The thinking process can have multiple code snippets. Each code snippet is wrapped "
    "with: <code>...</code>, and should be executable. "
    "Details: "
    "1. Identify sections where code execution could speed up the reasoning process or make "
    "the calculation more accurate. "
    "2. Replace the manual calculation steps with code snippets and the corresponding "
    "interpreter's execution results. "
    "3. Keep the logical flow of the reasoning process intact, including any failed "
    "exploration attempts that were part of the initial process. "
    "4. The code snippets should be complete scripts, including necessary imports, and "
    "should not contain markdown symbols like <python>...</python>. "
    "5. Outputs in the code snippets must explicitly call the print function. "
    "6. Execution results should match the model's output exactly, with no extra or "
    "missing tokens."
)
MAX_TOOL_USES_EXHAUSTED_PROMPT = (
    "Reached the maximum number of tool use. Please output final answer directly.\n"
    "Ensure that your response includes the format of '\\boxed{answer}', e.g. \\boxed{A}."
)


def _restore_code_stop_tag(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("</code>"):
        return text
    trailing = text[len(stripped):]
    return f"{stripped}</code>{trailing}"


def _extract_code(text: str) -> str | None:
    """Extract code from <code>...</code> or unclosed <code> (stop_strings strips closing tag)."""
    m = CODE_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    idx = text.rfind("<code>")
    if idx != -1:
        code = text[idx + len("<code>"):]
        return code.strip() if code.strip() else None
    return None


def _extract_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


async def _execute_code_local(code: str, timeout: int = 10) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = os.path.join(tmpdir, "run.py")
        with open(code_file, "w") as f:
            f.write(code)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["python3", code_file],
                capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
            )
            output = result.stdout
            if result.returncode != 0:
                error_lines = (result.stderr or "").strip().split("\n")[-5:]
                output = (output + "\n[ERROR]\n" + "\n".join(error_lines)).strip()
        except subprocess.TimeoutExpired:
            output = f"[ERROR] Code execution timed out after {timeout}s"
        except Exception as e:
            output = f"[ERROR] {type(e).__name__}: {e}"
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated to {MAX_OUTPUT_CHARS} chars)"
    return output.strip()


class CodeExecAgent:
    """Code DAPO agent for AReaL proxy mode.

    Not a RolloutWorkflow — AReaL auto-wraps this in OpenAIProxyWorkflow.
    """

    def __init__(
        self,
        system_prompt: str = "",
        code_timeout: int = 10,
        max_turns: int = 10,
        max_tool_uses: int = 5,
        stop: list[str] | None = None,
        temperature: float = 0.99,
        max_completion_tokens: int = 4096,
        **kwargs,
    ):
        self.system_prompt = system_prompt
        self.code_timeout = code_timeout
        self.max_turns = max_turns
        self.max_tool_uses = max_tool_uses
        self.stop = stop or ["</code>"]
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    async def run(self, data: dict, **extra_kwargs: Any) -> float:
        """Run one code execution episode. Returns scalar reward."""
        import httpx

        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        http_client: httpx.AsyncClient | None = extra_kwargs.get("http_client")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        # Extract problem and answer
        question = data.get("prompt", data.get("question", ""))
        solution = data.get("solution", data.get("answer", ""))
        ground_truth = _extract_boxed(solution) or solution

        # Build initial messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = f"{question}\n{PYTHON_CODE_TOOL_INSTRUCTION}"
        if "\\boxed" not in question:
            user_content += "\nEnsure that your response includes the format of '\\boxed{answer}', e.g. \\boxed{A}."
        messages.append({"role": "user", "content": user_content})

        tool_use_count = 0
        tool_use_success = 0
        accumulated_text = ""

        for turn in range(self.max_turns):
            # Call LLM via proxy
            completion = await client.chat.completions.create(
                model="default",
                messages=messages,
                stop=self.stop,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
            )

            assistant_text = completion.choices[0].message.content or ""
            accumulated_text += assistant_text

            # Check for boxed answer
            boxed_answer = _extract_boxed(accumulated_text)
            if boxed_answer is not None:
                messages.append({"role": "assistant", "content": assistant_text})
                break

            # Check for code
            code = _extract_code(assistant_text)
            if code is not None and tool_use_count < self.max_tool_uses:
                tool_use_count += 1
                # Restore the stop tag if the inference backend stripped it.
                full_assistant_text = _restore_code_stop_tag(assistant_text)
                messages.append({"role": "assistant", "content": full_assistant_text})

                exec_output = await _execute_code_local(code, self.code_timeout)
                if not exec_output.startswith("[ERROR]"):
                    tool_use_success += 1

                follow_up_prompt = "Continue. Put your final answer in \\boxed{}."
                if tool_use_count >= self.max_tool_uses:
                    follow_up_prompt = MAX_TOOL_USES_EXHAUSTED_PROMPT
                messages.append({
                    "role": "user",
                    "content": f"<output>\n{exec_output}\n</output>\n{follow_up_prompt}",
                })
                accumulated_text = (
                    accumulated_text.rstrip() + f"\n<output>\n{exec_output}\n</output>\n"
                )
            else:
                messages.append({"role": "assistant", "content": assistant_text})
                if code is not None and tool_use_count >= self.max_tool_uses:
                    break
                messages.append({
                    "role": "user",
                    "content": "Please provide your final answer in \\boxed{}.",
                })

        # Compute reward
        reward = math_verify_with_fallback(accumulated_text, ground_truth)

        # Report stats
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=reward,
            tool_use_count=tool_use_count,
            tool_use_success=tool_use_success / max(tool_use_count, 1),
            num_turns=min(turn + 1, self.max_turns),
        )
        return reward
