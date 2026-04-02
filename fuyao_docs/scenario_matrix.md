# Fuyao 场景矩阵

当前 Fuyao 适配围绕 3 个训练场景展开，入口、配置和外部依赖如下。

## 场景总览

| 场景      | 入口                                          | 配置                                               | 模型定位          | 数据                      | 奖励                                               | 外部服务   |
| --------- | --------------------------------------------- | -------------------------------------------------- | ----------------- | ------------------------- | -------------------------------------------------- | ---------- |
| Math RLVR | `fuyao_examples/math/train_math_rlvr.py`      | `fuyao_examples/math/qwen3_4b_rlvr.yaml`           | Qwen3 4B Instruct | `dapo_math_17k`           | `fuyao_examples.reward.math_reward_fn`             | 无         |
| Search R1 | `fuyao_examples/search_r1/train_search_r1.py` | `fuyao_examples/search_r1/search_r1_qwen3_4b.yaml` | Qwen3 4B Base     | NQ Search / HotpotQA      | `fuyao_examples.search_r1.reward.search_r1_reward` | 检索服务   |
| Code DAPO | `fuyao_examples/code_dapo/train_code_dapo.py` | `fuyao_examples/code_dapo/code_dapo_qwen3_4b.yaml` | Qwen3 4B Instruct | `dapo_math_17k_processed` | `fuyao_examples.reward.math_verify_with_fallback`  | 可选 execd |

## 运行矩阵

| 场景      | `--run-type` | 默认执行方式      | 关键环境变量          | 关键 stop token |
| --------- | ------------ | ----------------- | --------------------- | --------------- |
| Math RLVR | `math_rlvr`  | 直接 RLVRWorkflow | `SWANLAB_API_KEY`     | 常规生成参数    |
| Search R1 | `search_r1`  | Proxy 模式 Agent  | `RETRIEVAL_ENDPOINT`  | `</search>`     |
| Code DAPO | `code_dapo`  | Proxy 模式 Agent  | `EXECD_ENDPOINT` 可选 | `</code>`       |

## 维护要点

### Math RLVR

- 重点关注数据适配和 reward 兼容，不需要外部工具服务。
- 适合用来验证基础训练链路是否仍然可跑。

### Search R1

- 重点关注检索接口契约、stop token 处理、工具调用统计和 `RETRIEVAL_ENDPOINT` 注入。
- 这是最容易被 OpenAI Proxy 或推理后端行为差异影响的场景。
- 读指标时优先看：
  - `rollout/tool_use_attempts`
  - `rollout/successful_tool_uses`
  - `rollout/tool_use_rate`
  - `rollout/tool_success_rate`
  - `ppo_actor/seq_len/avg`
- 如果和 ROLL 对比，优先对齐到：
  - `tool_use_attempts ↔ env/.../tool_use_counter`
  - `successful_tool_uses ↔ env/.../tool_success_counter`
  - `seq_len ↔ prompt_length + non_prompt_length`

### Code DAPO

- 重点关注 `</code>` 截断、代码执行输出回填、`\\boxed{}` 抽取和工具成功率统计。
- 本地 subprocess 是默认路径，远程 execd 是附加能力，不应反向绑死主流程。
- 读指标时优先看：
  - `rollout/tool_use_count`
  - `rollout/tool_use_success`
  - `ppo_actor/task_reward/avg`
  - `ppo_actor/seq_len/avg`
- 注意 `rollout/tool_use_success` 是成功率，不是成功次数；和 ROLL 对比时只能近似映射到 `tool_success_counter / tool_use_counter`。

## 建议验证顺序

1. 先跑 Math RLVR，确认配置、日志和基础训练链路没断。
1. 再跑 Search R1，确认检索服务和 Proxy 回路稳定。
1. 最后跑 Code DAPO，确认 stop token 和多轮代码执行逻辑稳定。
