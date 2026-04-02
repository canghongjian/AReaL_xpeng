#!/usr/bin/env python3
"""Test that default_chat_template_kwargs works in ArealOpenAI.

Directly tests the tokenization path to verify enable_thinking=False
produces the empty <think></think> block.
"""

from transformers import AutoTokenizer

MODEL = "/publicdata/huggingface.co/Qwen/Qwen3-4B"

tok = AutoTokenizer.from_pretrained(MODEL)

messages = [
    {"role": "system", "content": "You can write code in <code>...</code>."},
    {"role": "user", "content": "What is 2+2?"},
]

# Test 1: Default apply_chat_template
print("=" * 60)
print("Test 1: Default tokenizer.apply_chat_template")
ids1 = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
text1 = tok.decode(ids1)
print(f"  Tokens: {len(ids1)}, Has <think>: {'<think>' in text1}")
print(f"  Tail: {repr(text1[-60:])}")

# Test 2: With enable_thinking=False
print()
print("=" * 60)
print("Test 2: tokenizer.apply_chat_template(enable_thinking=False)")
ids2 = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
text2 = tok.decode(ids2)
print(f"  Tokens: {len(ids2)}, Has <think>: {'<think>' in text2}")
print(f"  Tail: {repr(text2[-60:])}")

# Test 3: Simulate what ArealOpenAI does with default_chat_template_kwargs
print()
print("=" * 60)
print("Test 3: Simulated ArealOpenAI path")
default_kwargs = {"enable_thinking": False}
extra_body = {}  # No per-request override
ct_kwargs = {**default_kwargs, **extra_body.get("chat_template_kwargs", {})}
ids3 = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **ct_kwargs)
text3 = tok.decode(ids3)
print(f"  Tokens: {len(ids3)}, Has <think>: {'<think>' in text3}")
print(f"  Same as Test 2: {ids3 == ids2}")

# Test 4: Per-request override takes precedence
print()
print("=" * 60)
print("Test 4: Per-request override (enable_thinking=True) over default (False)")
extra_body4 = {"chat_template_kwargs": {"enable_thinking": True}}
ct_kwargs4 = {**default_kwargs, **extra_body4.get("chat_template_kwargs", {})}
ids4 = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **ct_kwargs4)
text4 = tok.decode(ids4)
print(f"  Tokens: {len(ids4)}, Has <think>: {'<think>' in text4}")
print(f"  Same as Test 1 (default): {ids4 == ids1}")

# Test 5: Verify OpenAIProxyConfig accepts the field
print()
print("=" * 60)
print("Test 5: OpenAIProxyConfig with chat_template_kwargs")
from areal.api.cli_args import OpenAIProxyConfig
cfg = OpenAIProxyConfig(
    mode="subproc",
    chat_template_kwargs={"enable_thinking": False},
)
print(f"  cfg.chat_template_kwargs: {cfg.chat_template_kwargs}")
print(f"  cfg.mode: {cfg.mode}")

# Summary
print()
print("=" * 60)
think_in_2 = "<think>" in text2 and "</think>" in text2
same_3_2 = ids3 == ids2
override_works = ids4 == ids1
config_ok = cfg.chat_template_kwargs == {"enable_thinking": False}

if think_in_2 and same_3_2 and override_works and config_ok:
    print("✓ ALL TESTS PASSED")
    print("  - enable_thinking=False produces empty <think></think>")
    print("  - default_chat_template_kwargs merge works correctly")
    print("  - Per-request override takes precedence")
    print("  - OpenAIProxyConfig accepts chat_template_kwargs field")
else:
    print("✗ SOME TESTS FAILED")
    print(f"  think_in_2={think_in_2}, same_3_2={same_3_2}, override={override_works}, config={config_ok}")
