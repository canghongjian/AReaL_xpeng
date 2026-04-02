import asyncio
import copy
import importlib
import sys
import types

import pytest


class RecordingTracker:
    def __init__(self):
        self.calls = []

    def scalar(self, **kwargs):
        self.calls.append(kwargs)


def _merge_scalar_calls(tracker: RecordingTracker) -> dict[str, float]:
    merged = {}
    for call in tracker.calls:
        merged.update(call)
    return merged


def _install_fake_openai(monkeypatch, responses):
    response_iter = iter(responses)
    create_calls = []

    def make_response(text):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=text),
                )
            ]
        )

    async def create(**kwargs):
        create_calls.append(kwargs)
        return make_response(next(response_iter))

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )
    return create_calls


def test_search_agent_handles_stop_strings_and_tool_loop(monkeypatch):
    """Search agent should restore a stripped stop tag and only reward the answer."""
    tracker = RecordingTracker()
    create_calls = _install_fake_openai(
        monkeypatch,
        [
            "<think>Need search</think><search>capital of france",
            "<answer>Paris</answer>",
        ],
    )

    fake_areal = types.ModuleType("areal")
    fake_areal.workflow_context = types.SimpleNamespace(stat_scope=lambda: "fuyao/search")
    fake_areal_utils = types.ModuleType("areal.utils")
    fake_areal_utils.stats_tracker = types.SimpleNamespace(get=lambda _scope: tracker)
    fake_areal.utils = fake_areal_utils

    monkeypatch.setitem(sys.modules, "areal", fake_areal)
    monkeypatch.setitem(sys.modules, "areal.utils", fake_areal_utils)

    search_module = importlib.import_module("fuyao_examples.search_r1.search_r1_agent")

    async def fake_retriever(query, search_url, topk=3, timeout=5):
        assert query == "capital of france"
        assert search_url == "http://retriever"
        assert topk == 3
        assert timeout == 5
        return "Doc 1(Title: France) Paris is the capital."

    monkeypatch.setattr(search_module, "call_retriever", fake_retriever)

    agent = search_module.SearchR1Agent(
        search_url="http://retriever",
        max_turns=3,
        max_tool_uses=2,
    )
    reward = asyncio.run(
        agent.run({"question": "Capital of France?", "golden_answers": ["Paris"]})
    )

    assert reward == pytest.approx(1.0)
    assert create_calls[0]["stop"] == ["</search>"]
    assert create_calls[1]["messages"][2]["content"].endswith("</search>")
    assert create_calls[1]["messages"][2]["content"].count("</search>") == 1

    scalars = _merge_scalar_calls(tracker)
    assert scalars["reward"] == pytest.approx(1.0)
    assert scalars["tool_use_attempts"] == 1.0
    assert scalars["successful_tool_uses"] == 1.0
    assert scalars["answer_after_search_rate"] == 1.0


def test_search_agent_does_not_duplicate_existing_search_stop_tag(monkeypatch):
    """Search agent should not append a second closing search tag."""
    tracker = RecordingTracker()
    create_calls = _install_fake_openai(
        monkeypatch,
        [
            "<think>Need search</think><search>capital of france</search>",
            "<answer>Paris</answer>",
        ],
    )

    fake_areal = types.ModuleType("areal")
    fake_areal.workflow_context = types.SimpleNamespace(stat_scope=lambda: "fuyao/search")
    fake_areal_utils = types.ModuleType("areal.utils")
    fake_areal_utils.stats_tracker = types.SimpleNamespace(get=lambda _scope: tracker)
    fake_areal.utils = fake_areal_utils

    monkeypatch.setitem(sys.modules, "areal", fake_areal)
    monkeypatch.setitem(sys.modules, "areal.utils", fake_areal_utils)

    sys.modules.pop("fuyao_examples.search_r1.search_r1_agent", None)
    search_module = importlib.import_module("fuyao_examples.search_r1.search_r1_agent")

    async def fake_retriever(query, search_url, topk=3, timeout=5):
        assert query == "capital of france"
        assert search_url == "http://retriever"
        assert topk == 3
        assert timeout == 5
        return "Doc 1(Title: France) Paris is the capital."

    monkeypatch.setattr(search_module, "call_retriever", fake_retriever)

    agent = search_module.SearchR1Agent(
        search_url="http://retriever",
        max_turns=3,
        max_tool_uses=2,
    )
    reward = asyncio.run(
        agent.run({"question": "Capital of France?", "golden_answers": ["Paris"]})
    )

    assert reward == pytest.approx(1.0)
    assert create_calls[1]["messages"][2]["content"] == (
        "<think>Need search</think><search>capital of france</search>"
    )


def test_search_agent_can_use_configured_input_key(monkeypatch):
    """Search agent should preserve the configured system prompt and input key."""
    tracker = RecordingTracker()
    create_calls = _install_fake_openai(
        monkeypatch,
        ["<answer>Paris</answer>"],
    )

    fake_areal = types.ModuleType("areal")
    fake_areal.workflow_context = types.SimpleNamespace(stat_scope=lambda: "fuyao/search")
    fake_areal_utils = types.ModuleType("areal.utils")
    fake_areal_utils.stats_tracker = types.SimpleNamespace(get=lambda _scope: tracker)
    fake_areal.utils = fake_areal_utils

    monkeypatch.setitem(sys.modules, "areal", fake_areal)
    monkeypatch.setitem(sys.modules, "areal.utils", fake_areal_utils)

    sys.modules.pop("fuyao_examples.search_r1.search_r1_agent", None)
    search_module = importlib.import_module("fuyao_examples.search_r1.search_r1_agent")

    agent = search_module.SearchR1Agent(
        search_url="http://retriever",
        system_prompt="custom system",
        input_key="prompt",
        max_turns=1,
    )
    reward = asyncio.run(
        agent.run(
            {
                "question": "Capital of France?",
                "prompt": "custom dataset prompt",
                "golden_answers": ["Paris"],
            }
        )
    )

    assert reward == pytest.approx(1.0)
    assert create_calls[0]["messages"][0]["content"] == "custom system"
    assert create_calls[0]["messages"][1]["content"] == "custom dataset prompt"


def test_search_reward_ignores_tool_success_bonus():
    from fuyao_examples.search_r1.reward import search_r1_reward

    reward = search_r1_reward(
        completions=(
            "<think>Need search</think>"
            "<search>capital of france</search>"
            "<tool_response><information>Paris is the capital.</information></tool_response>"
        ),
        golden_answers=["London"],
        tool_use_attempts=1,
        successful_tool_uses=1,
    )

    assert reward == pytest.approx(0.0)


def test_code_agent_handles_unclosed_code_stop_and_reports_stats(monkeypatch):
    """Code agent should append the stripped stop tag and continue after tool output."""
    tracker = RecordingTracker()
    create_calls = []
    reward_inputs = {}

    def make_response(text):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=text),
                )
            ]
        )

    response_iter = iter(["<code>print(42)", "The answer is \\boxed{42}"])

    async def create(**kwargs):
        create_calls.append(kwargs)
        return make_response(next(response_iter))

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    fake_logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    fake_logging_module = types.SimpleNamespace(getLogger=lambda *_args, **_kwargs: fake_logger)
    fake_areal = types.ModuleType("areal")
    fake_areal.workflow_context = types.SimpleNamespace(stat_scope=lambda: "fuyao/code")
    fake_areal_utils = types.ModuleType("areal.utils")
    fake_areal_utils.logging = fake_logging_module
    fake_areal_utils.stats_tracker = types.SimpleNamespace(get=lambda _scope: tracker)
    fake_areal.utils = fake_areal_utils

    fake_reward_module = types.ModuleType("fuyao_examples.reward")
    fake_reward_module.math_verify_with_fallback = lambda *_args, **_kwargs: 0.0

    monkeypatch.setitem(sys.modules, "areal", fake_areal)
    monkeypatch.setitem(sys.modules, "areal.utils", fake_areal_utils)
    monkeypatch.setitem(sys.modules, "fuyao_examples.reward", fake_reward_module)
    sys.modules.pop("fuyao_examples.code_dapo.code_exec_agent", None)
    code_exec_agent = importlib.import_module("fuyao_examples.code_dapo.code_exec_agent")

    monkeypatch.setattr(code_exec_agent, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        code_exec_agent,
        "workflow_context",
        types.SimpleNamespace(stat_scope=lambda: "fuyao/code"),
    )
    monkeypatch.setattr(
        code_exec_agent,
        "stats_tracker",
        types.SimpleNamespace(get=lambda _scope: tracker),
    )

    async def fake_execute_code_local(code, timeout=10):
        assert code == "print(42)"
        assert timeout == 10
        return "42"

    def fake_math_verify_with_fallback(response, ground_truth):
        reward_inputs["response"] = response
        reward_inputs["ground_truth"] = ground_truth
        return 1.0

    monkeypatch.setattr(code_exec_agent, "_execute_code_local", fake_execute_code_local)
    monkeypatch.setattr(
        code_exec_agent,
        "math_verify_with_fallback",
        fake_math_verify_with_fallback,
    )

    agent = code_exec_agent.CodeExecAgent(max_turns=3, max_tool_uses=2)
    reward = asyncio.run(agent.run({"prompt": "Use code", "solution": "\\boxed{42}"}))

    assert reward == 1.0
    assert create_calls[0]["stop"] == ["</code>"]
    first_user_prompt = create_calls[0]["messages"][0]["content"]
    assert "Now, you have the capability to write code" in first_user_prompt
    assert "Each code snippet is wrapped with: <code>...</code>" in first_user_prompt
    assert create_calls[1]["messages"][1]["content"] == "<code>print(42)</code>"
    assert "<output>\n42\n</output>" in reward_inputs["response"]
    assert reward_inputs["ground_truth"] == "42"

    scalars = _merge_scalar_calls(tracker)
    assert scalars["reward"] == 1.0
    assert scalars["tool_use_count"] == 1
    assert scalars["tool_use_success"] == 1.0
    assert scalars["num_turns"] == 2


def test_code_agent_does_not_duplicate_existing_code_stop_tag(monkeypatch):
    """Code agent should not append a second closing code tag."""
    tracker = RecordingTracker()
    create_calls = []
    reward_inputs = {}

    def make_response(text):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=text),
                )
            ]
        )

    response_iter = iter(["<code>print(42)</code>", "The answer is \\boxed{42}"])

    async def create(**kwargs):
        create_calls.append(kwargs)
        return make_response(next(response_iter))

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    fake_logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    fake_logging_module = types.SimpleNamespace(getLogger=lambda *_args, **_kwargs: fake_logger)
    fake_areal = types.ModuleType("areal")
    fake_areal.workflow_context = types.SimpleNamespace(stat_scope=lambda: "fuyao/code")
    fake_areal_utils = types.ModuleType("areal.utils")
    fake_areal_utils.logging = fake_logging_module
    fake_areal_utils.stats_tracker = types.SimpleNamespace(get=lambda _scope: tracker)
    fake_areal.utils = fake_areal_utils

    fake_reward_module = types.ModuleType("fuyao_examples.reward")
    fake_reward_module.math_verify_with_fallback = lambda *_args, **_kwargs: 0.0

    monkeypatch.setitem(sys.modules, "areal", fake_areal)
    monkeypatch.setitem(sys.modules, "areal.utils", fake_areal_utils)
    monkeypatch.setitem(sys.modules, "fuyao_examples.reward", fake_reward_module)
    sys.modules.pop("fuyao_examples.code_dapo.code_exec_agent", None)
    code_exec_agent = importlib.import_module("fuyao_examples.code_dapo.code_exec_agent")

    monkeypatch.setattr(code_exec_agent, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        code_exec_agent,
        "workflow_context",
        types.SimpleNamespace(stat_scope=lambda: "fuyao/code"),
    )
    monkeypatch.setattr(
        code_exec_agent,
        "stats_tracker",
        types.SimpleNamespace(get=lambda _scope: tracker),
    )

    async def fake_execute_code_local(code, timeout=10):
        assert code == "print(42)"
        assert timeout == 10
        return "42"

    def fake_math_verify_with_fallback(response, ground_truth):
        reward_inputs["response"] = response
        reward_inputs["ground_truth"] = ground_truth
        return 1.0

    monkeypatch.setattr(code_exec_agent, "_execute_code_local", fake_execute_code_local)
    monkeypatch.setattr(
        code_exec_agent,
        "math_verify_with_fallback",
        fake_math_verify_with_fallback,
    )

    agent = code_exec_agent.CodeExecAgent(max_turns=3, max_tool_uses=2)
    reward = asyncio.run(agent.run({"prompt": "Use code", "solution": "\\boxed{42}"}))

    assert reward == 1.0
    assert create_calls[1]["messages"][1]["content"] == "<code>print(42)</code>"
    assert reward_inputs["response"].count("</code>") == 1


def test_code_agent_matches_roll_max_tool_use_turn_behavior(monkeypatch):
    """After one tool use, the next prompt should force a direct final answer."""
    tracker = RecordingTracker()
    create_calls = []
    reward_inputs = {}

    def make_response(text):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=text),
                )
            ]
        )

    response_iter = iter(["<code>print(42)", "<code>print(43)</code>"])

    async def create(**kwargs):
        create_calls.append(copy.deepcopy(kwargs))
        return make_response(next(response_iter))

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    fake_logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    fake_logging_module = types.SimpleNamespace(getLogger=lambda *_args, **_kwargs: fake_logger)
    fake_areal = types.ModuleType("areal")
    fake_areal.workflow_context = types.SimpleNamespace(stat_scope=lambda: "fuyao/code")
    fake_areal_utils = types.ModuleType("areal.utils")
    fake_areal_utils.logging = fake_logging_module
    fake_areal_utils.stats_tracker = types.SimpleNamespace(get=lambda _scope: tracker)
    fake_areal.utils = fake_areal_utils

    fake_reward_module = types.ModuleType("fuyao_examples.reward")
    fake_reward_module.math_verify_with_fallback = lambda *_args, **_kwargs: 0.0

    monkeypatch.setitem(sys.modules, "areal", fake_areal)
    monkeypatch.setitem(sys.modules, "areal.utils", fake_areal_utils)
    monkeypatch.setitem(sys.modules, "fuyao_examples.reward", fake_reward_module)
    sys.modules.pop("fuyao_examples.code_dapo.code_exec_agent", None)
    code_exec_agent = importlib.import_module("fuyao_examples.code_dapo.code_exec_agent")

    monkeypatch.setattr(code_exec_agent, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        code_exec_agent,
        "workflow_context",
        types.SimpleNamespace(stat_scope=lambda: "fuyao/code"),
    )
    monkeypatch.setattr(
        code_exec_agent,
        "stats_tracker",
        types.SimpleNamespace(get=lambda _scope: tracker),
    )

    async def fake_execute_code_local(code, timeout=10):
        assert code == "print(42)"
        assert timeout == 10
        return "42"

    def fake_math_verify_with_fallback(response, ground_truth):
        reward_inputs["response"] = response
        reward_inputs["ground_truth"] = ground_truth
        return 0.0

    monkeypatch.setattr(code_exec_agent, "_execute_code_local", fake_execute_code_local)
    monkeypatch.setattr(
        code_exec_agent,
        "math_verify_with_fallback",
        fake_math_verify_with_fallback,
    )

    agent = code_exec_agent.CodeExecAgent(max_turns=3, max_tool_uses=1)
    reward = asyncio.run(agent.run({"prompt": "Use code", "solution": "\\boxed{42}"}))

    assert reward == 0.0
    assert len(create_calls) == 2
    assert (
        "Reached the maximum number of tool use. Please output final answer directly."
        in create_calls[1]["messages"][-1]["content"]
    )
    assert "Please provide your final answer in \\boxed{}." not in reward_inputs["response"]

    scalars = _merge_scalar_calls(tracker)
    assert scalars["tool_use_count"] == 1
    assert scalars["num_turns"] == 2


def test_fuyao_reward_uses_no_timeout_parser_in_threaded_env(monkeypatch):
    reward_module = importlib.import_module("fuyao_examples.reward")

    parse_calls = []
    grader_calls = []

    def fake_parse(text, target, parsing_timeout=None):
        parse_calls.append((text, parsing_timeout))
        return ["42"] if "42" in text else []

    def fake_grader(gold, pred, float_rounding=None, timeout_seconds=None):
        grader_calls.append((gold, pred, float_rounding, timeout_seconds))
        return gold == pred

    monkeypatch.setattr(reward_module, "math_verify_parse", fake_parse)
    monkeypatch.setattr(reward_module, "math_verify_grader", fake_grader)

    assert reward_module.math_verify_with_fallback("\\boxed{42}", "42") == pytest.approx(1.0)
    assert reward_module.math_reward_fn(None, "\\boxed{42}", None, None, "42") == pytest.approx(1.0)
    assert all(timeout is None for _text, timeout in parse_calls)
    assert all(timeout_seconds is None for *_rest, timeout_seconds in grader_calls)
