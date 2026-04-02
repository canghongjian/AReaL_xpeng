# 上游升级检查清单

当 AReaL upstream 更新后，Fuyao 适配层优先做兼容性核对，不直接同步改造代码。

## 先检查哪些边界

1. `PPOTrainer.train(...)` 是否还支持字符串形式的 `workflow` / `eval_workflow`
1. OpenAI Proxy 路径是否仍支持 duck-typed agent
1. `GRPOConfig` 继承扩展后的字段解析是否保持兼容
1. `StatsLogger.commit(...)` 的签名和调用时机是否变化
1. `stats_tracker` 与 `workflow_context` 的统计接口是否变化
1. 推理后端对 `stop` / `stop_strings` 的返回行为是否变化

## 要看的 Fuyao 文件

- `fuyao_examples/fuyao_areal_run.sh`
- `fuyao_examples/configs.py`
- `fuyao_examples/tracking_patch.py`
- `fuyao_examples/math/train_math_rlvr.py`
- `fuyao_examples/search_r1/train_search_r1.py`
- `fuyao_examples/search_r1/search_r1_agent.py`
- `fuyao_examples/code_dapo/train_code_dapo.py`
- `fuyao_examples/code_dapo/code_exec_agent.py`

## 最小回归命令

优先跑旁路适配层自己的快速回归：

```bash
uv run pytest \
  tests/test_fuyao_artifacts.py \
  tests/test_fuyao_tracking_patch.py \
  tests/test_vllm_agentic_stop.py
```

如果这组测试失败，先修 Fuyao 适配层，不急着碰上游。

## 升级后要同步的文档

- `fuyao_examples/README.md`
- `fuyao_docs/scenario_matrix.md`
- `fuyao_docs/adaptation_boundary.md`
- `fuyao_docs/lessons_learned.md`

## 何时才考虑修改上游

- 旁路适配层已经无法接住上游变更
- 现有 patch 方案开始明显脆弱
- 该问题本身就是上游通用 bug，而不是 Fuyao 私有需求

满足这些条件时，再进入上游改动评审，并补对应测试。
