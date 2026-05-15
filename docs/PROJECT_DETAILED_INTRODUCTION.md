# 爱恩斯坦棋智能体项目详细介绍

本文档从整体角度介绍本项目的设计思路、代码结构、训练流程、最终模型和运行方式，可作为报告前言、项目说明或答辩材料的基础。

## 1. 项目简介

本项目实现了一个面向爱恩斯坦棋的完整智能对弈系统。项目不仅实现了游戏规则和可视化界面，还实现了基础 MCTS、策略价值网络、神经网络引导 MCTS、自我对弈数据生成、模型训练、模型评估、报告图表生成以及在线中间对战平台接入。

项目最终采用：

```text
策略价值网络：policy_value_v4_full_best_loss.pt
搜索算法：full-neural-mcts
搜索参数：simulations=80, depth=12, chance_mode=sample
```

在 CPU 环境下，为保证稳定运行，可以使用：

```text
simulations=40, depth=8
```

## 2. 游戏规则实现

爱恩斯坦棋棋盘为 5x5 方格。红方从左上角区域出发，蓝方从右下角区域出发。双方各有 6 枚编号为 1-6 的棋子。

游戏过程如下：

1. 双方在各自出发区内布置 6 枚棋子。
2. 每回合掷骰子。
3. 当前玩家必须移动与骰子点数相同的棋子。
4. 如果该编号棋子已经被移出棋盘，则选择编号最接近骰子点数的可用棋子。
5. 红方可向下、向右、右下移动一格。
6. 蓝方可向上、向左、左上移动一格。
7. 目标位置有棋子时，直接吃掉该棋子；可以吃掉己方棋子。
8. 率先到达对方角点或吃光对方棋子者获胜。

规则实现集中在：

```text
einstein_chess/engine.py
```

该文件定义了：

- `EinsteinGame`：棋局对象。
- `PlayerColor`：红蓝方。
- `Move`：走法。
- `GameSnapshot`：用于智能体决策的局面快照。

规则引擎是整个项目的基础。所有 AI、GUI、训练数据生成和在线比赛都依赖同一套规则引擎，保证训练和实战规则一致。

## 3. 系统功能

项目具备以下功能：

### 3.1 本地 GUI

通过 `main.py` 启动 Tkinter 图形界面，支持：

- 人类 vs 人类。
- 人类 vs AI。
- AI vs AI。
- 手动开局布局。
- 默认/随机开局布局。

示例命令：

```bash
python main.py --red human --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

### 3.2 完整比赛运行器

`einstein_chess/match.py` 实现了 `MatchRunner`，支持：

- 15 分钟包干计时。
- 非法动作判负。
- 布局错误判负。
- 超时判负。
- 最大回合数限制。
- 每步行为记录。

### 3.3 批量评估

`scripts/evaluate_agents.py` 可以自动运行多局比赛并统计胜率：

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue random --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample
```

### 3.4 在线中间平台接入

`scripts/competition_client.py` 用于连接 `online_match.py` 中间对战服务器，支持与第三方 AI 对战。

参赛命令：

```bash
python scripts/competition_client.py --host 服务器IP --port 8765 --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --agent full-neural-mcts --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

## 4. 智能体类型

项目中实现了多种智能体：

| 智能体 | 参数名 | 说明 |
|---|---|---|
| 随机 AI | `random` | 从合法动作中随机选择。 |
| 基础 MCTS | `mcts` | 使用 UCT 和 rollout，不依赖神经网络。 |
| 纯策略价值网络 | `policy` | 只使用神经网络策略头，不搜索。 |
| root neural-MCTS | `neural-mcts` | 根节点 PUCT + 一步后 value 评估。 |
| full neural-MCTS | `full-neural-mcts` | 多层 PUCT + chance node + value 回传。 |

其中最终参赛智能体为：

```text
full-neural-mcts
```

它使用最终模型：

```text
artifacts/checkpoints/policy_value_v4_full_best_loss.pt
```

## 5. 状态编码与动作编码

### 5.1 状态编码

神经网络输入为 `15 x 5 x 5` 张量。

通道设计如下：

```text
0-5   : 红方 1-6 号棋子位置
6-11  : 蓝方 1-6 号棋子位置
12    : 当前行动方
13    : 骰子点数 dice / 6
14    : 当前合法候选棋子位置
```

该编码既保留了棋盘空间结构，又显式加入了骰子信息和当前行动方信息。

代码位置：

```text
einstein_chess/training/state_encoder.py
```

### 5.2 动作编码

动作空间为 18 维：

```text
6 个棋子 x 3 个方向 = 18
```

动作编号公式：

```text
action_id = (piece_number - 1) * 3 + direction_index
```

代码位置：

```text
einstein_chess/training/action_codec.py
```

## 6. 策略价值网络

策略价值网络输入局面状态，输出：

```text
policy logits: 18 维动作策略
value: 当前局面胜负价值
```

训练目标包括：

- 策略头学习 MCTS 搜索访问次数分布。
- 价值头学习最终胜负结果。

损失函数为：

```text
L = policy_loss + value_loss_weight * value_loss
```

项目尝试过 CNN 和 ResNet。最终采用 CNN 结构，因为 ResNet 实验虽然训练指标尚可，但对战结果明显弱于 v3。

代码位置：

```text
einstein_chess/training/model.py
```

## 7. MCTS 与神经网络结合

### 7.1 基础 MCTS

基础 MCTS 用于早期数据生成。它不依赖神经网络，主要通过 UCT 和 rollout 进行决策。

代码位置：

```text
einstein_chess/agents/mcts.py
```

### 7.2 root neural-MCTS

root neural-MCTS 使用神经网络策略头作为根节点动作先验，并在走一步后用价值头评估局面。

它的优点是速度快，适合早期迭代；缺点是搜索深度有限。

代码位置：

```text
einstein_chess/agents/neural_mcts.py
```

### 7.3 full neural-MCTS

full neural-MCTS 是最终搜索器。它在多层搜索树中使用 PUCT 选择动作，在叶子节点调用价值网络，并对爱恩斯坦棋的骰子随机性加入 chance node。

核心特点：

```text
多层 PUCT
骰子 chance node
叶子 value 评估
价值回传
```

代码位置：

```text
einstein_chess/agents/full_neural_mcts.py
```

实验表明，在使用同一 v3 网络时：

```text
full neural-MCTS vs root neural-MCTS = 58.5% 胜率
```

说明完整多层搜索能够明显提升决策质量。

## 8. 自我对弈与训练数据

训练数据来自自我对弈。每一步记录：

```text
state
policy
value
player
dice_roll
action_id
game_id
turn_index
winner
```

其中：

- `policy` 来自 MCTS 搜索访问次数分布。
- `value` 来自最终胜负，赢为 `+1`，输为 `-1`。

最终训练集：

```text
artifacts/data/self_play_mixed_v4_full.npz
```

其规模为：

```text
3000 games
53583 samples
```

## 9. 模型迭代过程

项目经历了多个阶段。

### 9.1 v1

v1 使用基础 MCTS 数据训练：

```text
self_play_500g_100s.npz
```

作用是获得第一版策略价值网络。

### 9.2 v2

v2 使用基础 MCTS 和 root neural-MCTS 数据混合训练：

```text
self_play_mixed_v2_1000g.npz
```

### 9.3 v3

v3 加入随机开局数据，提升对不同初始布局的泛化能力：

```text
self_play_mixed_v3_random.npz
```

实验结果：

```text
v3 vs v2 = 65% 胜率
```

### 9.4 ResNet v4 实验

项目尝试过更深的 ResNet 结构，但实际对战效果较差：

```text
ResNet v4 vs v3 = 32.5% 胜率
```

因此没有采用。

### 9.5 v4_full

v4_full 使用 full neural-MCTS 生成的高质量数据训练：

```text
self_play_mixed_v4_full.npz
```

最终模型：

```text
policy_value_v4_full_best_loss.pt
```

对比 v3：

```text
v4_full vs v3 = 55% 胜率
```

## 10. 图表与实验结果

项目已生成报告图表，位于：

```text
artifacts/figures/report/
```

核心图表包括：

```text
training_pipeline.png
state_action_encoding.png
self_play_dataset_growth.png
v4_full_loss.png
v4_full_policy_top1.png
v4_full_value_mae.png
comparison_validation_loss.png
comparison_validation_top1.png
self_play_reward_curve.png
evaluation_progression_win_rates.png
comparison_best_training_metrics.png
final_dataset_action_distribution.png
final_dataset_dice_distribution.png
final_agent_baseline_win_rates.png
```

图表说明见：

```text
docs/REPORT_FIGURES_GUIDE.md
```

## 11. 最终运行方式

### 11.1 GUI 人机对弈

```bash
python main.py --red human --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

### 11.2 在线平台参赛

```bash
python scripts/competition_client.py --host 服务器IP --port 8765 --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --agent full-neural-mcts --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

### 11.3 批量评估

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue random --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample
```

## 12. 项目特点

本项目的特点包括：

1. **规则完整**  
   完整实现爱恩斯坦棋规则，包括骰子、最接近棋子选择、自吃、胜负判定。

2. **工程结构清晰**  
   规则、智能体、训练、评估、GUI、在线客户端分模块实现。

3. **训练流程完整**  
   支持自我对弈数据生成、NPZ 数据保存、模型训练、checkpoint 保存、训练曲线绘制。

4. **算法逐步增强**  
   从基础 MCTS 到 root neural-MCTS，再到 full neural-MCTS。

5. **实验链路清楚**  
   每次改进都有对应数据和对战评估。

6. **最终版本可实战运行**  
   支持本地 GUI 和第三方中间平台对战。

## 13. 不足与后续改进

当前项目仍有一些可改进点：

- full neural-MCTS 计算量较高，CPU 环境需要降低参数。
- 当前训练不是逐 epoch 在线强化学习，因此收益曲线采用阶段性评估胜率表示。
- 开局布局目前主要使用默认或随机方式，未来可以加入开局布局搜索。
- 可以加入红蓝对称数据增强。
- 可以根据剩余时间动态调整搜索次数和深度。
- chance node 可以进一步优化为更高效的期望估计。

## 14. 总结

本项目最终完成了一个具备完整对弈能力的爱恩斯坦棋智能体系统。系统以规则引擎为基础，通过 MCTS 自我对弈生成数据，训练策略价值网络，再利用神经网络指导 MCTS 搜索，并最终发展为包含多层 PUCT 和骰子 chance node 的 full neural-MCTS。

实验结果表明：

```text
v3 相比 v2：65% 胜率
full neural-MCTS 相比 root neural-MCTS：58.5% 胜率
v4_full 相比 v3：55% 胜率
```

最终采用：

```text
policy_value_v4_full_best_loss.pt + full-neural-mcts
```

作为参赛和展示版本。

