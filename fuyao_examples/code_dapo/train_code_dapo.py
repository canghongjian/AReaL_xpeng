"""Code DAPO Agentic RL training entry point (proxy mode).

Usage:
    python fuyao_examples/code_dapo/train_code_dapo.py \
        --config fuyao_examples/code_dapo/code_dapo_qwen3_4b.yaml
"""

import sys

from datasets import load_dataset

from fuyao_examples.configs import AgenticConfig
from fuyao_examples.tracking_patch import apply_tracking_patch

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config


def load_dapo_math_dataset(path: str, split: str = "train"):
    """Load dapo_math_17k dataset for code execution RL."""
    ds = load_dataset("parquet", data_dir=path, split="train")

    def process(sample):
        return {
            "prompt": sample["prompt"],
            "solution": sample["solution"],
        }

    ds = ds.map(process)
    cols_to_remove = [c for c in ds.column_names if c not in ("prompt", "solution")]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)
    return ds


def main(args):
    config, _ = load_expr_config(args, AgenticConfig)

    train_dataset = load_dapo_math_dataset(config.train_dataset.path)
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = load_dapo_math_dataset(config.valid_dataset.path)

    apply_tracking_patch()

    # Agent workflow kwargs — passed to CodeExecAgent.__init__
    workflow_kwargs = dict(
        system_prompt=config.system_prompt,
        code_timeout=config.code_timeout,
        max_turns=config.max_turns,
        max_tool_uses=config.max_tool_uses,
        stop=list(config.gconfig.stop) if config.gconfig.stop else ["</code>"],
        temperature=config.gconfig.temperature,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        trainer.train(
            # Not a RolloutWorkflow — AReaL auto-wraps in OpenAIProxyWorkflow
            workflow="fuyao_examples.code_dapo.code_exec_agent.CodeExecAgent",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="fuyao_examples.code_dapo.code_exec_agent.CodeExecAgent",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
