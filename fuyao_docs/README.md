# Fuyao AReaL 文档

AReaL 框架在 Fuyao 平台上的适配、设计与实践文档。

## 核心文档

| 文档                                           | 内容                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------- |
| [areal_fuyao_vision.md](areal_fuyao_vision.md) | **AReaL_Fuyao 终局态设计** — 三层训练模式、与 Forge/Composer 2 的对齐、实施路径 |
| [lessons_learned.md](lessons_learned.md)       | **实践踩坑记录** — 从 RLVR 到 Agentic 的完整实践路径和关键发现                  |

## 维护与升级

| 文档                                                           | 内容                                                                      |
| -------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [adaptation_boundary.md](adaptation_boundary.md)               | **旁路适配边界** — 哪些逻辑放在 `fuyao_examples/`，哪些情况下才允许动上游 |
| [scenario_matrix.md](scenario_matrix.md)                       | **场景矩阵** — 三个训练场景的入口、配置、依赖、外部服务和验证方式         |
| [upstream_upgrade_checklist.md](upstream_upgrade_checklist.md) | **升级检查清单** — 上游 AReaL 升级时要逐项核对的兼容点和最小回归命令      |

## AReaL 入门

初次接触 AReaL 框架建议按顺序阅读：

| 文档                                                                       | 内容                                                     |
| -------------------------------------------------------------------------- | -------------------------------------------------------- |
| [getting_started/architecture.md](getting_started/architecture.md)         | 整体六层架构：Controller / Workflow / Engine / RPC       |
| [getting_started/yaml_config.md](getting_started/yaml_config.md)           | YAML 配置全字段说明：资源分配 / 超参 / 调度策略          |
| [getting_started/trainer_workflow.md](getting_started/trainer_workflow.md) | PPOTrainer 训练循环与 Workflow 执行细节                  |
| [getting_started/proxy_system.md](getting_started/proxy_system.md)         | Proxy 体系：ArealOpenAI / ProxyServer / Gateway 三种模式 |

## 参考来源

| 文档                          | 位置                                                               |
| ----------------------------- | ------------------------------------------------------------------ |
| Agentic RL 框架评估报告       | `learn/rl_training_frameworks/agentic_rl_frameworks_survey.md`     |
| Forge (MiniMax) 解读          | `learn/rl_training_frameworks/forge_minimax_faithful.md`           |
| Composer 2 (Cursor) 解读      | `learn/papers/composer2/composer2_deep_dive.md`                    |
| 架构对比 (Evals/Harbor/Forge) | `learn/rl_training_frameworks/agentic_architectures_comparison.md` |
