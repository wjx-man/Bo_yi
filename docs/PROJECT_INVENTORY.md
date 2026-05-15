# 爱恩斯坦棋项目文件与数据总览

本文档用于说明项目中每个主要源文件、脚本、测试文件和训练产物的作用。最终推荐版本为：

- 最终模型：`artifacts/checkpoints/policy_value_v4_full_best_loss.pt`
- 最终智能体：`full-neural-mcts`
- 推荐推理参数：`--full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample`

## 一、项目顶层文件

| 路径 | 说明 |
|---|---|
| `README.md` | 项目快速说明，包含运行 GUI、自我对弈数据生成、训练、评估、绘制曲线等常用命令。 |
| `main.py` | 本地 GUI 启动入口。支持人类、随机 AI、基础 MCTS、策略价值网络、root neural-MCTS、full neural-MCTS。 |
| `online_match.py` | 在线对局房间/服务端相关入口，用于和在线客户端完成完整比赛流程。 |

## 二、核心规则与对局模块

| 路径 | 说明 |
|---|---|
| `einstein_chess/engine.py` | 爱恩斯坦棋规则引擎。定义棋盘、棋子、走法、骰子、合法动作、胜负判断、默认/随机初始摆放。 |
| `einstein_chess/players.py` | 玩家接口。包含 `PlayerAgent` 抽象类、`HumanPlayer`、`RandomAIPlayer`。所有 AI 都通过该接口接入比赛。 |
| `einstein_chess/match.py` | 完整比赛运行器。负责 15 分钟包干计时、开局布阵、回合推进、非法走法判负、超时判负、日志记录。 |
| `einstein_chess/ui.py` | Tkinter 图形界面。支持人机/机机对弈、手动选择初始棋子摆放、AI 走子展示。 |
| `einstein_chess/online_match_client.py` | 在线比赛客户端协议工具。负责解析服务端状态、发送布局、发送走法、转换网络消息和内部棋局对象。 |
| `einstein_chess/__init__.py` | 包导出入口。导出核心类、智能体和训练常量；在线模块采用可选导入，避免离线训练/评估被在线文件影响。 |

## 三、智能体模块

| 路径 | 说明 |
|---|---|
| `einstein_chess/agents/mcts.py` | 基础 MCTS 智能体。使用 UCT 选择动作，随机/启发式 rollout 到终局或最大步数。用于早期自我对弈数据生成。 |
| `einstein_chess/agents/policy_value.py` | 纯策略价值网络智能体。加载 PyTorch checkpoint，直接根据策略头选择合法动作。 |
| `einstein_chess/agents/neural_mcts.py` | 第一版 neural-MCTS。使用网络策略先验做根节点 PUCT，走一步后用 value 评估叶子。速度快，但不是完整多层树。 |
| `einstein_chess/agents/full_neural_mcts.py` | 最终版完整 neural-MCTS。支持多层 PUCT 搜索、骰子 chance node、叶子 value 评估和回传。 |
| `einstein_chess/agents/__init__.py` | 智能体模块导出入口。 |

## 四、训练与编码模块

| 路径 | 说明 |
|---|---|
| `einstein_chess/training/state_encoder.py` | 状态编码。将局面编码为 `15 x 5 x 5` 张量，包括红棋通道、蓝棋通道、当前玩家、骰子点数和合法位置辅助通道。 |
| `einstein_chess/training/action_codec.py` | 动作编码。将动作编码为 18 维动作空间：6 个棋子编号乘以 3 个方向。 |
| `einstein_chess/training/self_play.py` | 自我对弈数据生成核心逻辑。支持 `mcts`、`neural-mcts`、`full-neural-mcts`，支持默认/随机开局。 |
| `einstein_chess/training/dataset.py` | NPZ 数据集读取器，将 `states/policies/values` 转换为 PyTorch Dataset。 |
| `einstein_chess/training/model.py` | PyTorch 策略价值网络。支持旧版 CNN 和 ResNet 结构；最终模型使用旧版 CNN 结构。 |
| `einstein_chess/training/__init__.py` | 训练模块导出入口。 |

## 五、命令脚本

| 路径 | 说明 |
|---|---|
| `scripts/generate_self_play.py` | 生成自我对弈 NPZ 数据。可选择 MCTS、root neural-MCTS、full neural-MCTS。 |
| `scripts/inspect_dataset.py` | 检查 NPZ 数据集是否合法，输出样本数、胜负分布、骰子分布、动作分布、策略分布是否归一等。 |
| `scripts/merge_datasets.py` | 合并多个 NPZ 自我对弈数据集，生成更大的训练集。 |
| `scripts/train_policy_value.py` | 训练策略价值网络。支持 best loss / best top1 checkpoint、CSV 训练日志、LR scheduler、early stopping、ResNet 参数。 |
| `scripts/evaluate_agents.py` | 批量评估智能体胜率。支持随机开局、红蓝互换、CSV 总结和逐局日志输出。 |
| `scripts/plot_training_log.py` | 根据训练 CSV 日志绘制 loss、top1、value MAE 曲线。 |
| `scripts/online_client.py` | 在线随机/基础客户端。 |
| `scripts/online_human_client.py` | 在线人类客户端。 |
| `scripts/online_model_client.py` | 在线模型客户端。支持 `policy`、`neural-mcts`、`full-neural-mcts`。 |

## 六、测试文件

| 路径 | 说明 |
|---|---|
| `tests/test_engine.py` | 测试规则引擎：棋盘、走法、吃子、胜负、骰子候选棋子等。 |
| `tests/test_match.py` | 测试比赛运行器：计时、非法动作、错误布局、随机布局模式。 |
| `tests/test_mcts.py` | 测试基础 MCTS 合法走子和比赛运行。 |
| `tests/test_neural_mcts_agent.py` | 测试 root neural-MCTS 和 full neural-MCTS 的合法动作、策略分布、chance node。 |
| `tests/test_policy_value_agent.py` | 测试策略价值智能体加载 checkpoint 和输出合法动作。 |
| `tests/test_policy_value_training.py` | 测试模型、loss、Residual CNN 前向传播是否兼容。 |
| `tests/test_training_data.py` | 测试状态编码、动作编码、自我对弈数据生成、NPZ 数据格式。 |
| `tests/test_merge_datasets.py` | 测试 NPZ 数据集合并。 |
| `tests/test_reporting_scripts.py` | 测试训练曲线绘制和评估 CSV 输出。 |
| `tests/test_main_cli.py` | 测试 GUI CLI 能正确构建各种玩家。 |
| `tests/test_online_match_client.py` | 测试在线协议工具和 TCP 对战流程。 |

## 七、artifacts 数据目录

### 1. `artifacts/data/*.npz`

所有 `.npz` 自我对弈数据均包含以下字段：

| 字段 | 形状 | 说明 |
|---|---|---|
| `states` | `(N, 15, 5, 5)` | 状态编码张量。 |
| `policies` | `(N, 18)` | MCTS 搜索访问次数归一化后的策略标签。 |
| `values` | `(N,)` | 从当前行动方视角记录最终收益：胜为 `+1`，负为 `-1`，未分胜负为 `0`。 |
| `players` | `(N,)` | 当前行动方：红方 `1`，蓝方 `-1`。 |
| `dice_rolls` | `(N,)` | 当前回合骰子点数。 |
| `action_ids` | `(N,)` | 实际选择动作的 18 维动作编号。 |
| `game_ids` | `(N,)` | 样本所属对局编号。 |
| `turn_indices` | `(N,)` | 样本所属回合编号。 |
| `winners` | `(N,)` | 对局胜者：红方 `1`，蓝方 `-1`，无胜者 `0`。 |

当前主要数据集：

| 路径 | 对局数 | 样本数 | 来源 | 用途 |
|---|---:|---:|---|---|
| `artifacts/data/self_play_10g_50s.npz` | 10 | 168 | 基础 MCTS，小规模调试数据。 | 测试脚本和流程验证。 |
| `artifacts/data/self_play_100g_50s.npz` | 100 | 1752 | 基础 MCTS。 | 早期数据试验。 |
| `artifacts/data/self_play_500g_100s.npz` | 500 | 8702 | 基础 MCTS，100 simulations。 | v1 初始策略价值网络训练。 |
| `artifacts/data/self_play_neural_mcts_500g_80s.npz` | 500 | 8528 | v1 网络指导 root neural-MCTS。 | v2 混合数据来源之一。 |
| `artifacts/data/self_play_mixed_v2_1000g.npz` | 1000 | 17230 | 基础 MCTS + root neural-MCTS 数据合并。 | v2 训练集。 |
| `artifacts/data/self_play_neural_mcts_v2_random_1000g_80s.npz` | 1000 | 18795 | v2 网络 + root neural-MCTS + 随机开局。 | v3 新增数据。 |
| `artifacts/data/self_play_mixed_v3_random.npz` | 2000 | 36025 | v2 混合数据 + v2 随机开局数据。 | v3 训练集。 |
| `artifacts/data/self_play_full_neural_mcts_v3_random_1000g_80s.npz` | 1000 | 17558 | v3 网络 + full neural-MCTS + 随机开局。 | v4_full 新增数据。 |
| `artifacts/data/self_play_mixed_v4_full.npz` | 3000 | 53583 | v3 训练集 + full neural-MCTS 新数据。 | 最终 v4_full 训练集。 |

最终训练集为：

```text
artifacts/data/self_play_mixed_v4_full.npz
```

### 2. `artifacts/checkpoints/*.pt`

| 路径 | 说明 |
|---|---|
| `policy_value_500g_100s.pt` | 早期基础 MCTS 数据训练的 checkpoint。 |
| `policy_value_latest.pt` | 初期训练的 latest checkpoint。 |
| `policy_value_best_loss.pt` | 初期训练按验证 loss 选择的 checkpoint，常称 v1。 |
| `policy_value_best_top1.pt` | 初期训练按验证 top1 选择的 checkpoint。 |
| `policy_value_v2_latest.pt` | v2 训练 latest checkpoint。 |
| `policy_value_v2_best_loss.pt` | v2 按验证 loss 选择的 checkpoint。 |
| `policy_value_v2_best_top1.pt` | v2 按验证 top1 选择的 checkpoint。 |
| `policy_value_v3_latest.pt` | v3 训练 latest checkpoint。 |
| `policy_value_v3_best_loss.pt` | v3 主力 checkpoint，曾作为 full neural-MCTS 搜索器验证基础。 |
| `policy_value_v3_best_top1.pt` | v3 按验证 top1 选择的 checkpoint。 |
| `policy_value_resnet_v4_latest.pt` | ResNet 实验 latest checkpoint。 |
| `policy_value_resnet_v4_best_loss.pt` | ResNet 实验 best loss checkpoint；对战结果弱于 v3，未作为最终模型。 |
| `policy_value_resnet_v4_best_top1.pt` | ResNet 实验 best top1 checkpoint；未作为最终模型。 |
| `policy_value_v4_full_latest.pt` | v4_full 训练 latest checkpoint。 |
| `policy_value_v4_full_best_loss.pt` | 最终推荐 checkpoint。 |
| `policy_value_v4_full_best_top1.pt` | v4_full 按验证 top1 选择的 checkpoint，备用。 |
| `policy_value_smoke.pt` | 冒烟测试 checkpoint，用于验证训练流程。 |

最终模型为：

```text
artifacts/checkpoints/policy_value_v4_full_best_loss.pt
```

### 3. `artifacts/logs/*.csv`

训练日志：

| 路径 | 说明 |
|---|---|
| `train_500g_100s.csv` / `train_500g_100s_gpu.csv` | v1 训练日志。 |
| `train_v2_mixed.csv` | v2 训练日志。 |
| `train_v3_random.csv` | v3 训练日志。 |
| `train_resnet_v4.csv` | ResNet v4 实验训练日志。 |
| `train_v4_full.csv` | 最终 v4_full 训练日志。 |
| `train_smoke.csv` | 冒烟测试日志。 |

评估日志：

| 路径 | 说明 |
|---|---|
| `eval_v3_best_loss_red_vs_random.csv` / `eval_v3_best_loss_blue_vs_random.csv` | v3 对随机 AI 的红蓝双边评估。 |
| `eval_v3_best_loss_red_vs_mcts50.csv` / `eval_v3_best_loss_blue_vs_mcts50.csv` | v3 对基础 MCTS50 的红蓝双边评估。 |
| `eval_v3_red_vs_v2_blue.csv` / `eval_v2_red_vs_v3_blue.csv` | v3 与 v2 在默认布局附近的互换评估。 |
| `eval_random_v3_red_vs_v2_blue.csv` / `eval_random_v2_red_vs_v3_blue.csv` | v3 与 v2 在随机布局下的互换评估。 |
| `eval_random_resnet_v4_red_vs_v3_blue.csv` / `eval_random_v3_red_vs_resnet_v4_blue.csv` | ResNet v4 与 v3 的互换评估，证明 ResNet 版本较弱。 |
| `eval_random_full_v3_red_vs_root_v3_blue.csv` / `eval_random_root_v3_red_vs_full_v3_blue.csv` | full neural-MCTS 与 root neural-MCTS 的互换评估。 |
| `eval_random_full_v4_loss_red_vs_full_v3_blue.csv` / `eval_random_full_v3_red_vs_full_v4_loss_blue.csv` | 最终 v4_full 与 v3 的互换评估。 |
| `*_games.csv` | 与对应 summary CSV 配套的逐局结果，记录每一局胜者、原因、步数和剩余时间等。 |

### 4. `artifacts/figures/`

用于保存训练曲线图片。通过以下命令生成：

```bash
python scripts/plot_training_log.py artifacts/logs/train_v4_full.csv --output-dir artifacts/figures --prefix train_v4_full
```

会生成 loss、policy top1、value MAE 曲线，可放入最终 PDF 报告。

## 八、可忽略的生成文件

以下文件或目录是 Python/测试工具自动生成的缓存，不属于源代码和训练数据交付重点：

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `.gitkeep` 仅用于保留空目录。

