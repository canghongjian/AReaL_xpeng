#!/usr/bin/env python3
"""Quick test: check what triggers stop in Test 2."""

import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL = "/publicdata/huggingface.co/Qwen/Qwen3-4B"
GPU_ID = 7
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
        [sys.executable, "-m", "sglang.launch_server",
         "--model-path", MODEL, "--port", str(PORT),
         "--dtype", "bfloat16", "--mem-fraction-static", "0.8",
         "--context-length", "8192"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for i in range(120):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i+1}s")
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    sys.exit(1)


def send_chat(messages, stop=None, enable_thinking=None, max_tokens=4096):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.99,
        "top_p": 0.99,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    resp = requests.post(f"http://localhost:{PORT}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"], data["choices"][0]["finish_reason"]


def test():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]

    print("\n" + "=" * 60)
    print("Test A: enable_thinking=False, stop=['</code>'] — full content")
    print("=" * 60)
    content, reason = send_chat(messages, stop=["</code>"], enable_thinking=False)
    print(f"finish_reason: {reason}")
    print(f"length: {len(content)} chars")
    print(f"has <code>: {'<code>' in content}")
    print(f"last 200 chars: {repr(content[-200:])}")
    print()

    # Check if </code> appears as substring in the content
    # (it shouldn't if it was the stop trigger, but let's check context)
    print("Checking for 'code' mentions in content:")
    for i, line in enumerate(content.split('\n')):
        if 'code' in line.lower():
            print(f"  line {i}: {line[:120]}")

    print("\n" + "=" * 60)
    print("Test B: enable_thinking=False, stop=['</code>'] — different question")
    print("=" * 60)
    messages2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 123 * 456?\nEnsure that your response includes the format of '\\boxed{answer}'."},
    ]
    content2, reason2 = send_chat(messages2, stop=["</code>"], enable_thinking=False)
    print(f"finish_reason: {reason2}")
    print(f"length: {len(content2)} chars")
    print(f"has <code>: {'<code>' in content2}")
    print(f"last 200 chars: {repr(content2[-200:])}")

    print("\n" + "=" * 60)
    print("Test C: enable_thinking=False, no stop — same simple question")
    print("=" * 60)
    content3, reason3 = send_chat(messages2, enable_thinking=False)
    print(f"finish_reason: {reason3}")
    print(f"length: {len(content3)} chars")
    print(f"has <code>: {'<code>' in content3}")
    if '<code>' in content3:
        idx = content3.find('<code>')
        print(f"  code context: ...{content3[idx:idx+200]}...")


if __name__ == "__main__":
    proc = start_server()
    try:
        test()
    finally:
        print("\nShutting down...")
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
