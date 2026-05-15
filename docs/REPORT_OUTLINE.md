# 爱恩斯坦棋智能体设计说明文档写作大纲

本文档用于指导最终 PDF 报告写作。每一节都说明了应当写什么内容、可以引用哪些代码文件、应插入哪些图表，以及适合放入报告正文的表格或文字。

最终推荐版本：

```text
最终模型：artifacts/checkpoints/policy_value_v4_full_best_loss.pt
最终智能体：full-neural-mcts
最终训练集：artifacts/data/self_play_mixed_v4_full.npz
推荐参数：simulations=80, depth=12, chance_mode=sample
```

图表目录：

```text
artifacts/figures/report/
```

图表索引：

```text
artifacts/figures/report/FIGURE_INDEX.md
```

## 1. 项目概述

### 1.1 项目背景

本节应介绍爱恩斯坦棋的基本规则和智能体开发背景。需要说明：

- 爱恩斯坦棋是一个 5x5 方格棋盘游戏。
- 红蓝双方各有 6 枚编号为 1-6 的棋子。
- 每回合由骰子决定可走棋子。
- 若对应编号棋子已被移出，则选择最接近骰子点数的可用棋子。
- 红方只能向右、向下、右下移动；蓝方只能向左、向上、左上移动。
- 棋子可以吃掉目标格上的任意棋子，包括己方棋子。
- 胜利条件是率先到达对方出发区角点，或吃光对方棋子。
- 对局只有胜负，没有和棋。

可以写成：

> 爱恩斯坦棋具有规则简单、状态空间较小、但含有骰子随机性的特点。与围棋、五子棋等确定性棋类不同，智能体不仅需要根据当前棋盘判断局面，还必须结合当前骰子点数进行受限动作选择。因此，本项目采用“规则引擎 + MCTS + 策略价值网络 + 自我对弈迭代”的方式实现具备完整对弈能力的智能程序。

### 1.2 任务目标

本节应对照提交要求说明项目目标：

- 自主实现完整规则引擎。
- 实现可稳定运行的智能程序。
- 支持人机对弈、机机对战和在线中间平台对战。
- 支持 15 分钟包干计时。
- 支持开局棋子布局选择。
- 支持自我对弈数据生成、神经网络训练和模型评估。
- 不引用外部开源完整策略模块。

### 1.3 最终成果

本节列出最终产物：

```text
最终模型：policy_value_v4_full_best_loss.pt
最终智能体：full-neural-mcts
最终搜索方式：多层 PUCT + 骰子 chance node + 叶子 value 回传
最终数据集：self_play_mixed_v4_full.npz
最终运行入口：scripts/competition_client.py
```

建议插入图表：

```text
artifacts/figures/report/training_pipeline.png
```

图注可写：

> 图 1 展示了项目从规则引擎、基础 MCTS、自我对弈数据生成，到策略价值网络、root neural-MCTS、full neural-MCTS 和最终模型 v4_full 的完整训练流程。

## 2. 游戏规则与系统功能

### 2.1 规则引擎实现

本节说明规则如何映射到代码。需要写：

- 棋盘使用 5x5 二维数组。
- 红方棋子编码为正数 `1..6`。
- 蓝方棋子编码为负数 `-1..-6`。
- 空格编码为 `0`。
- 当前玩家、骰子点数、胜者、回合数都在 `EinsteinGame` 中维护。

对应代码：

```text
einstein_chess/engine.py
```

重点说明的函数/类：

```text
EinsteinGame
PlayerColor
Move
GameSnapshot
get_legal_moves()
apply_move()
get_candidate_numbers()
random_layout_for()
```

### 2.2 比赛运行器与计时

本节说明完整比赛流程：

- 比赛开始前双方提交布局。
- 每方拥有 15 分钟包干时间。
- 每步调用智能体 `choose_move`。
- 如果超时、非法动作或布局错误，则判负。
- 记录每一步的回合编号、颜色、骰子、合法动作数、实际动作、耗时和剩余时间。

对应代码：

```text
einstein_chess/match.py
```

可以写：

> `MatchRunner` 将规则引擎和智能体接口连接起来，保证任何智能体都必须通过统一的合法动作列表进行选择，从而避免模型绕过规则直接修改棋盘。

### 2.3 本地 GUI 与在线平台

本节说明项目支持的运行方式：

- GUI 人机对弈：`main.py`
- 批量评估：`scripts/evaluate_agents.py`
- 在线中间平台：`online_match.py`
- 参赛客户端：`scripts/competition_client.py`

在线平台协议使用 JSON Lines，每条消息为一行 JSON。客户端主要发送：

```json
{"type":"layout","order":[1,2,3,4,5,6]}
{"type":"move","piece_number":4,"to":[2,1]}
{"type":"pass"}
```

可引用：

```text
AI_CLIENT_PROTOCOL.md
scripts/competition_client.py
```

## 3. 项目文件结构

本节建议用表格展示核心文件。可以从以下内容整理：

| 模块 | 文件 | 作用 |
|---|---|---|
| 规则引擎 | `einstein_chess/engine.py` | 棋盘、规则、走法、胜负判断 |
| 比赛流程 | `einstein_chess/match.py` | 计时、完整比赛、非法行为处理 |
| 玩家接口 | `einstein_chess/players.py` | 人类、随机 AI、智能体接口 |
| 基础 MCTS | `einstein_chess/agents/mcts.py` | UCT 搜索与 rollout |
| root neural-MCTS | `einstein_chess/agents/neural_mcts.py` | 根节点 PUCT + value 评估 |
| full neural-MCTS | `einstein_chess/agents/full_neural_mcts.py` | 多层 PUCT + chance node |
| 策略价值网络 | `einstein_chess/training/model.py` | PyTorch 网络与 loss |
| 状态编码 | `einstein_chess/training/state_encoder.py` | 15 通道状态编码 |
| 动作编码 | `einstein_chess/training/action_codec.py` | 18 维动作空间 |
| 自我对弈 | `einstein_chess/training/self_play.py` | 生成 NPZ 训练数据 |
| 训练脚本 | `scripts/train_policy_value.py` | 训练策略价值网络 |
| 评估脚本 | `scripts/evaluate_agents.py` | 批量胜率评估 |
| 参赛客户端 | `scripts/competition_client.py` | 接入中间对战平台 |

可引用文档：

```text
docs/PROJECT_INVENTORY.md
```

## 4. 状态空间建模与动作定义

### 4.1 状态空间定义

本节应说明一个局面包含：

```text
board
current_player
dice_roll
winner
legal_moves
turn_index
```

解释：

- `board` 表示棋盘上所有棋子位置。
- `current_player` 表示当前行动方。
- `dice_roll` 表示当前骰子点数。
- `legal_moves` 由规则引擎根据当前棋盘和骰子计算。
- `turn_index` 用于记录回合顺序。

### 4.2 状态编码

本节必须详细写。网络输入为：

```text
15 x 5 x 5
```

通道说明：

| 通道 | 含义 |
|---|---|
| 0-5 | 红方 1-6 号棋子位置 |
| 6-11 | 蓝方 1-6 号棋子位置 |
| 12 | 当前行动方，红方为 1，蓝方为 -1 |
| 13 | 骰子点数，取 `dice_roll / 6` |
| 14 | 合法候选棋子位置辅助通道 |

对应代码：

```text
einstein_chess/training/state_encoder.py
```

建议插图：

```text
artifacts/figures/report/state_action_encoding.png
```

### 4.3 动作编码

动作空间为：

```text
6 个棋子编号 x 3 个方向 = 18 个动作
```

动作编号公式：

```text
action_id = (piece_number - 1) * 3 + direction_index
```

红方三个方向：

```text
向下、向右、右下
```

蓝方三个方向：

```text
向上、向左、左上
```

对应代码：

```text
einstein_chess/training/action_codec.py
```

建议插图：

```text
artifacts/figures/report/final_dataset_action_distribution.png
artifacts/figures/report/final_dataset_dice_distribution.png
```

图表说明：

- 动作分布图用于说明最终训练数据覆盖了 18 个动作编号。
- 骰子分布图用于说明训练数据中的骰子点数基本均匀，不存在明显采样偏置。

## 5. 网络结构与算法说明

### 5.1 基础 MCTS

本节说明基础 MCTS 的作用：

- 在没有神经网络时产生初始自我对弈数据。
- 使用 UCT 公式在探索和利用之间平衡。
- 使用 rollout 或简单启发式评估非终局局面。

UCT 公式：

```text
UCT(s,a)=Q(s,a)+c*sqrt(ln N(s)/N(s,a))
```

对应代码：

```text
einstein_chess/agents/mcts.py
```

### 5.2 策略价值网络

本节说明神经网络结构。

输入：

```text
15 x 5 x 5 状态张量
```

输出：

```text
p: 18 维策略输出
v: [-1, 1] 价值输出
```

网络使用 CNN backbone，然后分成两个 head：

- policy head：输出 18 个动作的 logits。
- value head：输出局面价值。

损失函数：

```text
L = policy_loss + value_loss_weight * value_loss
policy_loss = -Σ π log p
value_loss = MSE(v, z)
```

对应代码：

```text
einstein_chess/training/model.py
```

### 5.3 root neural-MCTS

本节说明第一版神经网络 MCTS：

```text
根节点 PUCT + 走一步后 value 评估
```

特点：

- 搜索速度快。
- 能利用策略网络给动作先验。
- 但只在根节点做 PUCT，没有完整多层搜索。

对应代码：

```text
einstein_chess/agents/neural_mcts.py
```

### 5.4 full neural-MCTS

本节重点说明最终算法：

```text
多层 PUCT 搜索
骰子 chance node
叶子 value 评估
价值回传
```

PUCT 公式：

```text
PUCT(s,a)=Q(s,a)+c_puct*P(s,a)*sqrt(N(s))/(1+N(s,a))
```

骰子 chance node：

- `sample`：每次 simulation 随机采样一个骰子点数。
- `enumerate`：枚举 1-6 六种骰子结果求平均。

最终采用：

```text
chance_mode = sample
```

原因：

- 速度更快。
- 更适合 15 分钟包干或在线平台对战。
- 在实验中已经能带来稳定提升。

对应代码：

```text
einstein_chess/agents/full_neural_mcts.py
```

建议插图：

```text
artifacts/figures/report/training_pipeline.png
```

## 6. 训练流程

### 6.1 自我对弈数据格式

每个 `.npz` 数据集包含：

| 字段 | 含义 |
|---|---|
| `states` | 状态编码，形状为 `(N, 15, 5, 5)` |
| `policies` | MCTS 访问次数分布，形状为 `(N, 18)` |
| `values` | 从当前行动方视角的胜负收益 |
| `players` | 当前行动方，红为 1，蓝为 -1 |
| `dice_rolls` | 当前骰子点数 |
| `action_ids` | 实际选择动作编号 |
| `game_ids` | 对局编号 |
| `turn_indices` | 回合编号 |
| `winners` | 胜者编号 |

对应代码：

```text
einstein_chess/training/self_play.py
scripts/generate_self_play.py
scripts/inspect_dataset.py
```

### 6.2 数据集演进

建议插入表格：

| 数据集 | 对局数 | 样本数 | 来源 | 用途 |
|---|---:|---:|---|---|
| `self_play_500g_100s.npz` | 500 | 8702 | 基础 MCTS | v1 |
| `self_play_mixed_v2_1000g.npz` | 1000 | 17230 | MCTS + root neural-MCTS | v2 |
| `self_play_mixed_v3_random.npz` | 2000 | 36025 | 加入随机开局 | v3 |
| `self_play_mixed_v4_full.npz` | 3000 | 53583 | 加入 full neural-MCTS 数据 | v4_full |

建议插图：

```text
artifacts/figures/report/self_play_dataset_growth.png
```

### 6.3 训练流程描述

可以写：

> 训练采用离线自我对弈数据监督学习方式。MCTS 或 neural-MCTS 在每个局面搜索得到访问次数分布，将其归一化作为策略标签；对局结束后，根据胜负结果为每个状态赋予价值标签。策略价值网络同时学习动作分布和局面胜负价值。

训练命令以最终 v4_full 为例：

```bash
python scripts/train_policy_value.py artifacts/data/self_play_mixed_v4_full.npz --epochs 60 --batch-size 256 --hidden-channels 64 --residual-blocks 0 --value-loss-weight 1.0 --lr-scheduler plateau --early-stopping-patience 10 --device cuda --checkpoint artifacts/checkpoints/policy_value_v4_full_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_v4_full_best_top1.pt --log artifacts/logs/train_v4_full.csv
```

## 7. 每一代模型训练过程

### 7.1 v1：基础 MCTS 数据训练

应写内容：

- 数据来自基础 MCTS 自我对弈。
- 主要作用是训练第一版策略价值网络。
- 后续用于 root neural-MCTS。

数据：

```text
artifacts/data/self_play_500g_100s.npz
```

模型：

```text
artifacts/checkpoints/policy_value_best_loss.pt
```

建议插图：

```text
artifacts/figures/report/v1_loss.png
artifacts/figures/report/v1_policy_top1.png
artifacts/figures/report/v1_value_mae.png
```

### 7.2 v2：混合 root neural-MCTS 数据训练

应写内容：

- v2 使用基础 MCTS 数据和 root neural-MCTS 数据混合训练。
- 目的是让网络学习比基础 MCTS 更强的搜索策略。

数据：

```text
artifacts/data/self_play_mixed_v2_1000g.npz
```

模型：

```text
artifacts/checkpoints/policy_value_v2_best_loss.pt
```

建议插图：

```text
artifacts/figures/report/v2_loss.png
artifacts/figures/report/v2_policy_top1.png
artifacts/figures/report/v2_value_mae.png
```

### 7.3 v3：加入随机开局数据

应写内容：

- 爱恩斯坦棋允许自由摆放开局棋子。
- 为避免模型只适应默认开局，引入随机开局自我对弈。
- v3 相比 v2 在随机开局下明显增强。

数据：

```text
artifacts/data/self_play_mixed_v3_random.npz
```

模型：

```text
artifacts/checkpoints/policy_value_v3_best_loss.pt
```

评估结果：

```text
v3 vs v2：v3 综合胜率 65%
```

建议插图：

```text
artifacts/figures/report/v3_loss.png
artifacts/figures/report/v3_policy_top1.png
artifacts/figures/report/v3_value_mae.png
```

### 7.4 ResNet v4 实验

应写内容：

- 项目尝试过 Residual CNN 结构。
- 虽然训练指标尚可，但实战对战明显弱于 v3。
- 因此最终没有采用 ResNet 版本。

评估结果：

```text
ResNet v4 vs v3：ResNet v4 胜率 32.5%
```

建议插图：

```text
artifacts/figures/report/resnet_v4_loss.png
artifacts/figures/report/resnet_v4_policy_top1.png
artifacts/figures/report/resnet_v4_value_mae.png
```

### 7.5 full neural-MCTS 搜索器增强

应写内容：

- 固定 v3 网络，只替换搜索器。
- 对比 root neural-MCTS 与 full neural-MCTS。
- full neural-MCTS 在相同网络下综合胜率 58.5%，说明多层搜索有效。

评估结果：

```text
full neural-MCTS vs root neural-MCTS：117 / 200 = 58.5%
```

建议插图：

```text
artifacts/figures/report/evaluation_progression_win_rates.png
```

### 7.6 v4_full：最终模型

应写内容：

- 使用 v3 网络 + full neural-MCTS 生成高质量随机开局数据。
- 合并得到最终训练集 `self_play_mixed_v4_full.npz`。
- 训练得到最终模型 `policy_value_v4_full_best_loss.pt`。

训练结果：

```text
最佳 val_loss: epoch 6, val_loss = 1.7847
最高 val_top1: epoch 12, val_top1 = 0.688
early stopping: epoch 16
```

评估结果：

```text
v4_full vs v3：110 / 200 = 55%
```

建议插图：

```text
artifacts/figures/report/v4_full_loss.png
artifacts/figures/report/v4_full_policy_top1.png
artifacts/figures/report/v4_full_value_mae.png
```

## 8. 超参数设定

### 8.1 最终训练超参数

建议放表格：

| 参数 | 数值 |
|---|---|
| epochs | 60 |
| batch_size | 256 |
| hidden_channels | 64 |
| residual_blocks | 0 |
| learning_rate | 0.001 |
| weight_decay | 1e-4 |
| value_loss_weight | 1.0 |
| lr_scheduler | plateau |
| early_stopping_patience | 10 |
| device | cuda |

### 8.2 最终搜索超参数

| 参数 | 数值 |
|---|---|
| agent | full-neural-mcts |
| checkpoint | policy_value_v4_full_best_loss.pt |
| simulations | 80 |
| max_depth | 12 |
| c_puct | 1.5 |
| chance_mode | sample |

### 8.3 CPU 稳定运行参数

如果在 CPU 上运行，可以写：

```text
simulations = 40
max_depth = 8
```

用于降低单步思考时间，保证比赛稳定运行。

## 9. 训练曲线与收益曲线

### 9.1 Loss 曲线

必须插入：

```text
artifacts/figures/report/v4_full_loss.png
```

可选插入：

```text
artifacts/figures/report/comparison_validation_loss.png
```

说明：

> v4_full 在验证集上取得较低的 validation loss，说明 full neural-MCTS 数据提升了策略价值网络的拟合效果。

### 9.2 策略准确率曲线

建议插入：

```text
artifacts/figures/report/v4_full_policy_top1.png
artifacts/figures/report/comparison_validation_top1.png
```

### 9.3 价值误差曲线

建议插入：

```text
artifacts/figures/report/v4_full_value_mae.png
```

### 9.4 收益曲线

注意本节必须严谨表达。当前项目不是每个 epoch 在线强化学习，而是：

```text
自我对弈生成数据
→ 策略价值网络监督训练
→ 阶段性评估
```

因此收益曲线应写成：

> 本项目以最终胜负作为状态收益标签，胜为 +1，负为 -1。训练过程中的收益变化通过自我对弈数据的平均收益和模型代际评估胜率表示。

建议插入：

```text
artifacts/figures/report/self_play_reward_curve.png
artifacts/figures/report/evaluation_progression_win_rates.png
```

## 10. 评估结果

### 10.1 代际提升结果

建议表格：

| 实验 | 综合胜率 | 结论 |
|---|---:|---|
| v3 vs v2 | 65% | 随机开局数据提升泛化能力 |
| full neural-MCTS vs root neural-MCTS | 58.5% | 多层 PUCT 搜索有效 |
| ResNet v4 vs v3 | 32.5% | ResNet 结构实验失败 |
| v4_full vs v3 | 55% | 最终模型优于上一代 |

建议插图：

```text
artifacts/figures/report/evaluation_progression_win_rates.png
artifacts/figures/report/comparison_best_training_metrics.png
```

### 10.2 最终模型对不同基线

建议插入：

```text
artifacts/figures/report/final_agent_baseline_win_rates.png
```

说明最终模型对不同类型 AI 的表现。

### 10.3 搜索模块消融实验

如果已经补跑 `policy vs full-neural-mcts`，插入：

```text
artifacts/figures/report/ablation_policy_vs_full_neural_mcts.png
```

写法：

> 为验证搜索模块贡献，实验保持同一 v4_full 策略价值网络不变，只比较纯策略网络和 full neural-MCTS。若 full neural-MCTS 胜率明显高于 policy，则说明多层搜索模块有效提升了决策质量。

### 10.4 数据覆盖分析

插入：

```text
artifacts/figures/report/final_dataset_action_distribution.png
artifacts/figures/report/final_dataset_dice_distribution.png
```

说明：

- 18 个动作均被覆盖。
- 骰子 1-6 分布较均衡。
- 训练数据没有明显动作或骰子偏置。

## 11. 运行结果

### 11.1 GUI 人机对弈

写明命令：

```bash
python main.py --red human --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

### 11.2 批量评估

写明命令：

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue random --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample
```

### 11.3 在线平台接入

写明命令：

```bash
python scripts/competition_client.py --host 服务器IP --port 8765 --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --agent full-neural-mcts --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

对应协议文件：

```text
AI_CLIENT_PROTOCOL.md
online_match.py
scripts/competition_client.py
```

### 11.4 测试结果

写明：

```bash
python -m unittest discover -s tests
```

结果：

```text
Ran 37 tests
OK
```

## 12. 总结与展望

### 12.1 项目完成情况

总结已完成内容：

- 完整规则引擎。
- GUI 人机对弈。
- 15 分钟计时比赛运行器。
- 开局布局选择。
- 基础 MCTS。
- 策略价值网络。
- root neural-MCTS。
- full neural-MCTS。
- 自我对弈数据生成。
- 训练、评估、图表生成脚本。
- 在线中间平台接入。

### 12.2 最终结论

建议写：

> 实验结果表明，随机开局数据增强、多层 neural-MCTS 搜索以及高质量自我对弈数据均能提升模型表现。最终系统采用 `policy_value_v4_full_best_loss.pt` 与 `full-neural-mcts` 作为参赛智能体，在保持稳定运行的同时具备较强对弈能力。

### 12.3 不足与改进方向

可写：

- full neural-MCTS 计算量较高，CPU 下需要降低搜索次数和深度。
- 当前收益曲线是阶段性评估收益，不是逐 epoch 在线强化学习 reward。
- 后续可研究开局布局搜索。
- 可加入红蓝对称数据增强。
- 可实现基于剩余时间的自适应搜索深度。
- 可继续优化 chance node，从采样改为更高效的期望估计。

## 附录建议

### 附录 A：文件清单

引用：

```text
docs/PROJECT_INVENTORY.md
```

### 附录 B：图表清单

引用：

```text
docs/REPORT_FIGURES_GUIDE.md
artifacts/figures/report/FIGURE_INDEX.md
```

### 附录 C：主要命令

列出：

- 生成数据命令。
- 训练模型命令。
- 评估模型命令。
- 生成图表命令。
- 在线参赛命令。

### 附录 D：核心代码路径

列出：

```text
einstein_chess/engine.py
einstein_chess/match.py
einstein_chess/agents/mcts.py
einstein_chess/agents/neural_mcts.py
einstein_chess/agents/full_neural_mcts.py
einstein_chess/training/model.py
einstein_chess/training/self_play.py
scripts/competition_client.py
```

