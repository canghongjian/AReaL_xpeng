"""Search R1 Agentic RL training entry point (proxy mode).

Usage:
    python fuyao_examples/search_r1/train_search_r1.py \
        --config fuyao_examples/search_r1/search_r1_qwen3_4b.yaml
"""

import os
import sys

from datasets import load_dataset

from fuyao_examples.configs import AgenticConfig
from fuyao_examples.tracking_patch import apply_tracking_patch

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config


def load_search_dataset(
    path: str, question_key: str, answer_key: str, split: str = "train"
):
    """Load a search QA dataset from parquet/json files or HuggingFace."""
    if os.path.isdir(path):
        import glob

        parquet_files = glob.glob(os.path.join(path, "*.parquet"))
        json_files = glob.glob(os.path.join(path, "*.json")) + glob.glob(
            os.path.join(path, "*.jsonl")
        )
        if parquet_files:
            ds = load_dataset("parquet", data_files=parquet_files, split="train")
        elif json_files:
            ds = load_dataset("json", data_files=json_files, split="train")
        else:
            ds = load_dataset(path, split=split)
    else:
        ds = load_dataset(path, split=split)

    # Normalize column names, preserve prompt if available
    def normalize(sample):
        question = sample.get(question_key, "")
        answer = sample.get(answer_key, "")
        result = {"question": question, "golden_answers": answer}
        if "prompt" in sample and sample["prompt"]:
            result["prompt"] = sample["prompt"]
        return result

    ds = ds.map(normalize)
    return ds


def main(args):
    config, _ = load_expr_config(args, AgenticConfig)

    train_dataset = load_search_dataset(
        config.train_dataset.path,
        question_key="question",
        answer_key="golden_answers",
        split="train",
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = load_search_dataset(
            config.valid_dataset.path,
            question_key=getattr(config, "_val_question_key", "problem"),
            answer_key=getattr(config, "_val_answer_key", "answer"),
            split="test",
        )

    # Resolve retrieval endpoint
    retrieval_endpoint = config.retrieval_endpoint
    if not retrieval_endpoint:
        retrieval_endpoint = os.environ.get("RETRIEVAL_ENDPOINT", "")
    if not retrieval_endpoint:
        print(
            "ERROR: retrieval_endpoint not set. "
            "Set via config or RETRIEVAL_ENDPOINT env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    apply_tracking_patch()

    # Agent workflow kwargs
    workflow_kwargs = dict(
        search_url=retrieval_endpoint,
        system_prompt=config.system_prompt,
        input_key=config.input_key,
        answer_key=config.answer_key,
        max_turns=config.max_turns,
        max_tool_uses=config.max_tool_uses,
        temperature=config.gconfig.temperature,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        trainer.train(
            # Not a RolloutWorkflow — AReaL auto-wraps in OpenAIProxyWorkflow
            workflow="fuyao_examples.search_r1.search_r1_agent.SearchR1Agent",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="fuyao_examples.search_r1.search_r1_agent.SearchR1Agent",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
