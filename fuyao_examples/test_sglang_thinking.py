#!/usr/bin/env python3
"""Quick test: launch SGLang server, send prompts, check thinking and <code> behavior.

Usage:
    python fuyao_examples/test_sglang_thinking.py

This script:
1. Starts SGLang server on a free port (GPU 7)
2. Sends the same prompt with/without enable_thinking
3. Checks if model uses <code> tag
4. Shuts down server
"""

import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL = "/publicdata/huggingface.co/Qwen/Qwen3-4B"
GPU_ID = 7  # Use last GPU to avoid conflicts
PORT = 39877

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within '\\boxed{}', e.g. \\boxed{A}. "
    "You have the capability to write code to use the code interpreter for calculation. "
    "The code will be executed by a sandbox, and the result can be returned to enhance your reasoning process. "
    "Each code snippet is wrapped with: <code>...</code>, and should be executable. "
    "The code snippets should be complete scripts, including necessary imports. "
    "Outputs in the code snippets must explicitly call the print function."
)

QUESTION = (
    "Find the sum of all positive integers n such that n^2 + 12n - 2007 is a perfect square.\n"
    "Ensure that your response includes the format of '\\boxed{answer}', e.g. \\boxed{A}."
)


def start_server():
    print(f"Starting SGLang server on GPU {GPU_ID}, port {PORT}...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--port", str(PORT),
            "--dtype", "bfloat16",
            "--mem-fraction-static", "0.8",
            "--context-length", "8192",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait for server ready
    for i in range(120):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i+1}s")
                return proc
        except Exception:
            pass
        time.sleep(1)
    print("Server failed to start!")
    proc.kill()
    sys.exit(1)


def send_chat(messages, stop=None, enable_thinking=None, max_tokens=2048):
    """Send chat completion request to SGLang server."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.99,
        "top_p": 0.99,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop

    # SGLang uses chat_template_kwargs in the request body
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    finish_reason = data["choices"][0]["finish_reason"]
    usage = data.get("usage", {})
    return content, finish_reason, usage


def test():
    messages_default = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]

    print("\n" + "=" * 60)
    print("Test 1: Default (no enable_thinking), stop=['</code>']")
    print("=" * 60)
    content1, reason1, usage1 = send_chat(messages_default, stop=["</code>"])
    has_think1 = "<think>" in content1
    has_code1 = "<code>" in content1
    print(f"  finish_reason: {reason1}")
    print(f"  has_think: {has_think1}")
    print(f"  has_code: {has_code1}")
    print(f"  tokens: {usage1}")
    print(f"  content[:300]: {content1[:300]}")

    print("\n" + "=" * 60)
    print("Test 2: enable_thinking=False, stop=['</code>']")
    print("=" * 60)
    content2, reason2, usage2 = send_chat(messages_default, stop=["</code>"], enable_thinking=False)
    has_think2 = "<think>" in content2
    has_code2 = "<code>" in content2
    print(f"  finish_reason: {reason2}")
    print(f"  has_think: {has_think2}")
    print(f"  has_code: {has_code2}")
    print(f"  tokens: {usage2}")
    print(f"  content[:300]: {content2[:300]}")

    print("\n" + "=" * 60)
    print("Test 3: enable_thinking=True, stop=['</code>']")
    print("=" * 60)
    content3, reason3, usage3 = send_chat(messages_default, stop=["</code>"], enable_thinking=True)
    has_think3 = "<think>" in content3
    has_code3 = "<code>" in content3
    print(f"  finish_reason: {reason3}")
    print(f"  has_think: {has_think3}")
    print(f"  has_code: {has_code3}")
    print(f"  tokens: {usage3}")
    print(f"  content[:300]: {content3[:300]}")

    print("\n" + "=" * 60)
    print("Test 4: enable_thinking=False, no stop (let it finish)")
    print("=" * 60)
    content4, reason4, usage4 = send_chat(messages_default, enable_thinking=False, max_tokens=4096)
    has_think4 = "<think>" in content4
    has_code4 = "<code>" in content4
    has_boxed4 = "\\boxed" in content4
    print(f"  finish_reason: {reason4}")
    print(f"  has_think: {has_think4}")
    print(f"  has_code: {has_code4}")
    print(f"  has_boxed: {has_boxed4}")
    print(f"  tokens: {usage4}")
    print(f"  content[:300]: {content4[:300]}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Test 1 (default):            think={has_think1}, code={has_code1}")
    print(f"  Test 2 (no_think+stop):      think={has_think2}, code={has_code2}")
    print(f"  Test 3 (think+stop):         think={has_think3}, code={has_code3}")
    print(f"  Test 4 (no_think, no_stop):  think={has_think4}, code={has_code4}")


if __name__ == "__main__":
    proc = start_server()
    try:
        test()
    finally:
        print("\nShutting down server...")
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
        print("Done.")
