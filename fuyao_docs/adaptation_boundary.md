# Fuyao 旁路适配边界

目标是复用 AReaL 上游能力，同时把 Fuyao 特有逻辑控制在独立目录，降低后续同步 upstream 的冲突成本。

## 设计原则

- 优先复用上游 `areal/`、`examples/`、`docs/` 的既有能力，不在同名模块里平铺改造。
- Fuyao 特有入口、配置、Agent、数据适配和补丁优先落在 `fuyao_examples/`。
- Fuyao 设计说明、踩坑记录和升级策略优先落在 `fuyao_docs/`。
- 只有当上游缺少必要扩展点，且无法通过配置、动态导入、包装器或 monkey patch 解决时，才允许修改 `areal/`。

## 目录归属

| 路径                                | 归属         | 当前职责                                           |
| ----------------------------------- | ------------ | -------------------------------------------------- |
| `fuyao_examples/fuyao_areal_run.sh` | Fuyao 适配层 | 统一启动入口，管理场景切换和沙盒部署               |
| `fuyao_examples/math/`              | Fuyao 适配层 | Math RLVR 配置和训练入口                           |
| `fuyao_examples/search_r1/`         | Fuyao 适配层 | Search R1 的 Agent、Workflow、配置和训练入口       |
| `fuyao_examples/code_dapo/`         | Fuyao 适配层 | Code DAPO 的 Agent、Workflow、配置和训练入口       |
| `fuyao_examples/dataset/`           | Fuyao 适配层 | 数据格式归一化和 AReaL 数据接口适配                |
| `fuyao_examples/tracking_patch.py`  | Fuyao 适配层 | DeepInsight 指标映射，避免直接改上游 `StatsLogger` |
| `fuyao_docs/`                       | Fuyao 适配层 | 设计、经验、矩阵和升级检查文档                     |
| `areal/`                            | 上游核心     | 训练框架主逻辑，默认只复用不修改                   |

## 当前依赖的上游扩展点

- `PPOTrainer.train(...)` 的 `workflow` / `eval_workflow` 动态导入能力
- OpenAI Proxy 路径对 duck-typed agent 的兼容
- `GRPOConfig` 继承扩展为 `AgenticConfig`
- `StatsLogger.commit(...)` 可通过外部 patch 注入额外指标
- `stats_tracker` 和 `workflow_context` 可在工作流外层上报自定义统计

## 允许修改上游的条件

满足以下任一条件时，才考虑改 `areal/`：

1. 适配层无法通过包装或动态导入接入，必须新增通用扩展点。
1. 上游本身存在 bug，且该 bug 影响 Fuyao 之外的正常使用场景。
1. 当前 monkey patch 或旁路实现已经引入明显维护成本，继续外置的复杂度高于一次受控上游改动。

一旦修改上游，必须同时补三件事：

- 在 `fuyao_docs/upstream_upgrade_checklist.md` 里记录变更点
- 为该变更补最小回归测试
- 在提交说明里标注“为什么不能继续留在适配层”

## 每次新增 Fuyao 功能前的自检

- 这项逻辑是否只服务于 Fuyao 场景？
- 是否已经存在可复用的上游接口？
- 能否放在 `fuyao_examples/` 下，以入口、配置、包装器或 patch 的形式实现？
- 如果需要碰上游，是否已经写清楚不可外置的原因？
