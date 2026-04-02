#!/usr/bin/env python3
"""Test if ArealOpenAI respects extra_body chat_template_kwargs."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from transformers import AutoTokenizer

MODEL = "/publicdata/huggingface.co/Qwen/Qwen3-4B"


async def test():
    tok = AutoTokenizer.from_pretrained(MODEL)

    # Mock engine that captures what it receives
    captured_input_ids = []

    async def mock_agenerate(req):
        captured_input_ids.append(list(req.input_ids))
        # Return a minimal response
        resp = MagicMock()
        resp.output_tokens = [tok.eos_token_id]
        resp.output_logprobs = [0.0]
        resp.output_versions = [0]
        resp.output_len = 1
        resp.input_tokens = list(req.input_ids) + [tok.eos_token_id]
        resp.input_len = len(req.input_ids)
        resp.stop_reason = "stop"
        resp.latency = 0.1
        resp.ttft = 0.1
        resp.tokenizer = tok
        resp.processor = None
        resp.routed_experts = None
        resp.input_images = None
        return resp

    engine = MagicMock()
    engine.agenerate = mock_agenerate
    engine.get_version = MagicMock(return_value=0)

    from areal.experimental.openai.client import ArealOpenAI

    messages = [
        {"role": "system", "content": "You can write code."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    # Test 1: Without extra_body (default)
    client1 = ArealOpenAI(engine=engine, tokenizer=tok)
    try:
        await client1.chat.completions.create(
            model="default",
            messages=messages,
        )
    except Exception:
        pass
    ids1 = captured_input_ids[-1] if captured_input_ids else []
    text1 = tok.decode(ids1)
    print("Test 1: Default (no extra_body)")
    print(f"  Has <think>: {'<think>' in text1}")
    print(f"  Tail: {repr(text1[-60:])}")
    print()

    # Test 2: With extra_body enable_thinking=False
    captured_input_ids.clear()
    client2 = ArealOpenAI(engine=engine, tokenizer=tok)
    try:
        await client2.chat.completions.create(
            model="default",
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception:
        pass
    ids2 = captured_input_ids[-1] if captured_input_ids else []
    text2 = tok.decode(ids2)
    print("Test 2: extra_body={chat_template_kwargs: {enable_thinking: False}}")
    print(f"  Has <think>: {'<think>' in text2}")
    print(f"  Tail: {repr(text2[-60:])}")
    print()

    print(f"Token diff: {len(ids2) - len(ids1)}")


asyncio.run(test())
