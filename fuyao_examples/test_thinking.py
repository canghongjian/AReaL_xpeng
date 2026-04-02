#!/usr/bin/env python3
"""Test enable_thinking behavior with Qwen3 tokenizer.

Tests:
1. Default apply_chat_template → does it add <think>?
2. enable_thinking=False → adds empty <think></think>?
3. ArealOpenAI with extra_body → does it pass through?
"""

from transformers import AutoTokenizer

MODEL = "/publicdata/huggingface.co/Qwen/Qwen3-4B"

tok = AutoTokenizer.from_pretrained(MODEL)

messages = [
    {"role": "system", "content": "You can write code in <code>...</code>."},
    {"role": "user", "content": "What is 2+2?"},
]

print("=" * 60)
print("Test 1: Default (no enable_thinking)")
text1 = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"  Tail: {repr(text1[-60:])}")
print(f"  Has <think>: {'<think>' in text1}")
print()

print("=" * 60)
print("Test 2: enable_thinking=True")
text2 = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
print(f"  Tail: {repr(text2[-60:])}")
print(f"  Has <think>: {'<think>' in text2}")
print()

print("=" * 60)
print("Test 3: enable_thinking=False")
text3 = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(f"  Tail: {repr(text3[-60:])}")
print(f"  Has <think>: {'<think>' in text3}")
print(f"  Has empty think block: {'<think>\\n\\n</think>' in text3}")
print()

print("=" * 60)
print("Test 4: via chat_template_kwargs dict")
kwargs = {"enable_thinking": False}
text4 = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
print(f"  Tail: {repr(text4[-60:])}")
print(f"  Same as Test 3: {text3 == text4}")
print()

print("=" * 60)
print("Test 5: Token count comparison")
ids_default = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
ids_no_think = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
print(f"  Default tokens: {len(ids_default)}")
print(f"  enable_thinking=False tokens: {len(ids_no_think)}")
print(f"  Difference: {len(ids_no_think) - len(ids_default)} tokens")
