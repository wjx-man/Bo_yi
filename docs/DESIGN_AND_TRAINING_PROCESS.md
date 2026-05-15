# 爱恩斯坦棋智能体设计与训练过程说明

本文档记录项目从规则引擎到最终智能体的设计过程，以及每一代模型的数据来源、训练方式和评估结果。

最终结论：

- 最终模型：`artifacts/checkpoints/policy_value_v4_full_best_loss.pt`
- 最终搜索器：`full-neural-mcts`
- 最终推荐参数：`simulations=80`，`depth=12`，`chance_mode=sample`

## 一、任务目标

本项目目标是为爱恩斯坦棋自主开发具备完整对弈能力的智能程序。程序需要：

- 完整实现爱恩斯坦棋规则。
- 支持 5x5 棋盘、红蓝双方 1-6 号棋子、骰子约束走子、吃子、胜负判定。
- 支持开局自主或随机摆放棋子。
- 支持 15 分钟包干计时和完整比赛流程。
- 支持人机、机机、本地 GUI 和在线比赛客户端。
- 通过自我对弈、MCTS、策略价值网络提升棋力。
- 记录训练曲线、评估结果和运行结果，便于生成设计说明文档。

## 二、规则建模

### 1. 状态定义

一个局面由以下信息组成：

```text
board: 5x5 棋盘
current_player: 当前行动方
dice_roll: 当前骰子点数
winner: 胜者，若未结束则为空
legal_moves: 当前合法走法
turn_index: 回合编号
```

红方目标角为右下角 `(4, 4)`，蓝方目标角为左上角 `(0, 0)`。

### 2. 状态编码

神经网络输入使用 `15 x 5 x 5` 张量：

```text
0-5   : 红方 1-6 号棋子位置通道
6-11  : 蓝方 1-6 号棋子位置通道
12    : 当前行动方通道，红方为 1，蓝方为 -1
13    : 骰子点数通道，取 dice / 6
14    : 合法候选棋子位置辅助通道
```

实现位置：`einstein_chess/training/state_encoder.py`

### 3. 动作编码

动作空间为 18 维：

```text
6 个棋子编号 x 3 个方向 = 18 个动作
```

每个动作编号由棋子编号和方向组成：

```text
action_id = (piece_number - 1) * 3 + direction_index
```

红方方向：

```text
向下、向右、向右下
```

蓝方方向：

```text
向上、向左、向左上
```

实现位置：`einstein_chess/training/action_codec.py`

## 三、智能体演进路线

项目采用如下路线：

```text
规则引擎
→ 随机 AI
→ 基础 MCTS
→ 基础 MCTS 自我对弈数据
→ 策略价值网络
→ root neural-MCTS
→ 随机开局自我对弈
→ full neural-MCTS
→ full neural-MCTS 高质量自我对弈
→ 最终 v4_full 模型
```

## 四、基础 MCTS 阶段

### 1. 设计

基础 MCTS 使用 UCT 思想：

```text
UCT(s,a) = Q(s,a) + c * sqrt(ln N(s) / N(s,a))
```

流程：

```text
选择动作
→ 克隆局面
→ 执行动作
→ 随机/启发式 rollout
→ 回传胜负收益
```

实现位置：`einstein_chess/agents/mcts.py`

### 2. 数据生成

早期数据由基础 MCTS 自我对弈生成：

| 数据集 | 对局数 | 样本数 | 说明 |
|---|---:|---:|---|
| `self_play_10g_50s.npz` | 10 | 168 | 小规模测试数据。 |
| `self_play_100g_50s.npz` | 100 | 1752 | 早期试验数据。 |
| `self_play_500g_100s.npz` | 500 | 8702 | v1 主要训练数据。 |

每个样本格式为：

```text
(state, policy, value)
```

其中：

- `state` 是 15 通道状态编码。
- `policy` 是 MCTS 访问次数分布。
- `value` 是最终胜负收益。

## 五、策略价值网络

### 1. 网络输入输出

网络形式：

```text
(p, v) = f_theta(s)
```

其中：

- `p`：18 维策略 logits，对应动作概率。
- `v`：局面价值，范围为 `[-1, 1]`。

### 2. 损失函数

训练损失：

```text
L = policy_loss + value_loss_weight * value_loss
```

其中：

```text
policy_loss = - sum(pi * log p)
value_loss = MSE(v, z)
```

实现位置：`einstein_chess/training/model.py`

### 3. 训练脚本能力

`scripts/train_policy_value.py` 支持：

- latest checkpoint
- best validation loss checkpoint
- best validation top1 checkpoint
- CSV 训练日志
- `ReduceLROnPlateau` / cosine / none 学习率调度
- early stopping
- CNN 或 ResNet 结构

## 六、v1 模型

### 1. 数据来源

```text
artifacts/data/self_play_500g_100s.npz
```

该数据由基础 MCTS 自我对弈生成：

```text
500 games
8702 samples
100 simulations per move
```

### 2. 训练产物

```text
policy_value_best_loss.pt
policy_value_best_top1.pt
policy_value_latest.pt
```

### 3. 作用

v1 是第一版策略价值网络，用于替代纯随机 rollout，为 root neural-MCTS 提供初始策略先验和价值判断。

## 七、root neural-MCTS 阶段

### 1. 设计

第一版 neural-MCTS 使用：

```text
根节点 PUCT + 走一步后 value 评估
```

它不是完整多层树搜索，但计算较快，适合早期数据生成。

实现位置：`einstein_chess/agents/neural_mcts.py`

### 2. 搜索公式

```text
PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

其中：

- `P(s,a)` 来自策略网络。
- `Q(s,a)` 来自搜索统计。
- 叶子价值来自价值网络。

## 八、v2 模型

### 1. 数据来源

v2 使用混合数据：

```text
self_play_500g_100s.npz
self_play_neural_mcts_500g_80s.npz
```

合并后：

```text
artifacts/data/self_play_mixed_v2_1000g.npz
1000 games
17230 samples
```

### 2. 训练产物

```text
policy_value_v2_best_loss.pt
policy_value_v2_best_top1.pt
policy_value_v2_latest.pt
```

### 3. 作用

v2 相比 v1 引入了神经网络指导 MCTS 生成的数据，使网络学习到更高质量的搜索策略。

## 九、随机开局与 v3 模型

### 1. 引入原因

爱恩斯坦棋规则允许双方开局在出发区内自由摆放棋子。如果只用默认摆法训练，模型可能记住固定开局，泛化能力不足。因此加入随机开局自我对弈。

### 2. 新增数据

```text
artifacts/data/self_play_neural_mcts_v2_random_1000g_80s.npz
1000 games
18795 samples
```

合并得到：

```text
artifacts/data/self_play_mixed_v3_random.npz
2000 games
36025 samples
```

### 3. v3 训练结果

训练日志：

```text
artifacts/logs/train_v3_random.csv
```

主要结论：

- v3 在随机布局下明显强于 v2。
- v3 对随机 AI 和基础 MCTS 均有较好表现。

关键评估：

| 对局 | 结果 |
|---|---|
| v3 执红 vs v2 执蓝，随机开局 | v3 70/100 |
| v2 执红 vs v3 执蓝，随机开局 | v3 60/100 |
| 合计 | v3 130/200，胜率 65% |

因此 v3 成为后续 full neural-MCTS 搜索器验证的基础网络。

## 十、ResNet v4 实验

### 1. 改进尝试

曾尝试将模型改为 Residual CNN，并加入：

- residual blocks
- learning rate scheduler
- early stopping
- `value_loss_weight=0.5`

训练日志：

```text
artifacts/logs/train_resnet_v4.csv
```

### 2. 评估结果

| 对局 | 结果 |
|---|---|
| ResNet v4 执红 vs v3 执蓝 | ResNet v4 36/100 |
| v3 执红 vs ResNet v4 执蓝 | ResNet v4 29/100 |
| 合计 | ResNet v4 65/200，胜率 32.5% |

结论：ResNet v4 在当前数据规模和训练配置下明显弱于 v3，因此没有作为最终路线。

## 十一、full neural-MCTS 阶段

### 1. 引入原因

root neural-MCTS 只搜索根节点后的一步，不能充分利用价值网络。为了提升搜索质量，引入完整多层 neural-MCTS。

### 2. 设计

最终 full neural-MCTS 包含：

```text
多层 PUCT 搜索
骰子 chance node
叶子 value 评估
价值回传
```

chance node 有两种模式：

- `sample`：每次 simulation 随机采样一个骰子结果，速度较快，最终采用。
- `enumerate`：枚举 6 个骰子结果求平均，更严格但更慢。

实现位置：

```text
einstein_chess/agents/full_neural_mcts.py
```

### 3. 搜索器对比实验

在保持同一个 v3 网络不变的情况下，只比较搜索器：

| 对局 | 结果 |
|---|---|
| full neural-MCTS 执红 vs root neural-MCTS 执蓝 | full 60/100 |
| root neural-MCTS 执红 vs full neural-MCTS 执蓝 | full 57/100 |
| 合计 | full 117/200，胜率 58.5% |

结论：完整多层搜索器比根节点近似搜索更强。

## 十二、v4_full 最终模型

### 1. 数据来源

使用 v3 网络 + full neural-MCTS 生成新一轮随机开局数据：

```text
artifacts/data/self_play_full_neural_mcts_v3_random_1000g_80s.npz
1000 games
17558 samples
```

与 v3 训练集合并得到最终训练集：

```text
artifacts/data/self_play_mixed_v4_full.npz
3000 games
53583 samples
```

### 2. 训练命令

```bash
python scripts/train_policy_value.py artifacts/data/self_play_mixed_v4_full.npz --epochs 60 --batch-size 256 --hidden-channels 64 --residual-blocks 0 --value-loss-weight 1.0 --lr-scheduler plateau --early-stopping-patience 10 --device cuda --checkpoint artifacts/checkpoints/policy_value_v4_full_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_v4_full_best_top1.pt --log artifacts/logs/train_v4_full.csv
```

### 3. 训练结果

训练日志：

```text
artifacts/logs/train_v4_full.csv
```

关键结果：

```text
最佳 val_loss: epoch 6, val_loss = 1.7847
最高 val_top1: epoch 12, val_top1 = 0.688
early stopping: epoch 16
```

训练指标显示 v4_full 略优于 v3。

### 4. 对战评估

使用 full neural-MCTS 搜索器，固定搜索参数：

```text
simulations = 80
depth = 12
chance_mode = sample
layout = random
```

v4_full 与 v3 对比：

| 对局 | 结果 |
|---|---|
| v4_full 执红 vs v3 执蓝 | v4_full 57/100 |
| v3 执红 vs v4_full 执蓝 | v4_full 53/100 |
| 合计 | v4_full 110/200，胜率 55% |

结论：v4_full 相比 v3 有稳定但不夸张的提升，因此选为最终模型。

## 十三、最终运行配置

### 1. GUI 人机对弈

```bash
python main.py --red human --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample
```

CPU 环境可降低搜索量：

```bash
python main.py --red human --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

### 2. 批量评估

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue random --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample
```

### 3. 在线模型客户端

```bash
python scripts/online_model_client.py --agent full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample
```

## 十四、曲线与报告材料

训练曲线可由训练 CSV 生成：

```bash
python scripts/plot_training_log.py artifacts/logs/train_v4_full.csv --output-dir artifacts/figures --prefix train_v4_full
```

可放入报告的图：

- `train_v4_full_loss.png`
- `train_v4_full_top1.png`
- `train_v4_full_value_mae.png`

收益/胜率曲线可由多代评估结果整理：

```text
v2 -> v3: v3 随机开局综合胜率 65%
root neural-MCTS -> full neural-MCTS: full 综合胜率 58.5%
v3 -> v4_full: v4_full 综合胜率 55%
```

## 十五、最终结论

本项目最终采用：

```text
策略价值网络: policy_value_v4_full_best_loss.pt
搜索算法: full neural-MCTS
搜索参数: simulations=80, depth=12, chance_mode=sample
```

相较早期版本，项目完成了：

- 从规则引擎到完整比赛运行器。
- 从基础 MCTS 到神经网络指导搜索。
- 从固定开局到随机开局数据增强。
- 从 root neural-MCTS 到 full neural-MCTS。
- 从 v1/v2/v3 到最终 v4_full 的自我对弈强化迭代。

最终系统具备完整对弈能力、计时能力、数据生成能力、模型训练能力、评估能力和在线/本地运行能力。
