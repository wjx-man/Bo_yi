# 报告图表使用指南

图表已生成在：

```text
artifacts/figures/report/
```

图表索引为：

```text
artifacts/figures/report/FIGURE_INDEX.md
```

配套统计表为：

```text
artifacts/figures/report/training_summary.csv
artifacts/figures/report/dataset_summary.csv
artifacts/figures/report/evaluation_summary.csv
```

## 一、必须放入报告的图表

### 1. 训练 loss 曲线

报告要求包含训练过程的 loss 曲线。建议至少放最终模型，也可以放各代模型。

| 图表 | 用途 |
|---|---|
| `v1_loss.png` | v1 基础 MCTS 数据训练 loss。 |
| `v2_loss.png` | v2 混合数据训练 loss。 |
| `v3_loss.png` | v3 随机开局数据训练 loss。 |
| `resnet_v4_loss.png` | ResNet 结构实验 loss，用于说明该路线未被采用。 |
| `v4_full_loss.png` | 最终模型 v4_full 的 loss 曲线，必须放。 |
| `comparison_validation_loss.png` | 各代模型验证 loss 对比。 |

推荐报告写法：

> 图中可以看出，v4_full 在验证集上取得最低的 validation loss，说明加入 full neural-MCTS 生成的高质量自我对弈数据后，策略价值网络拟合效果有所提升。

### 2. 收益曲线

严格强化学习中的 reward curve 在本项目中通过自我对弈胜负收益和代际胜率体现。建议放：

| 图表 | 用途 |
|---|---|
| `self_play_reward_curve.png` | 自我对弈数据阶段的平均收益和红方胜率变化。 |
| `evaluation_progression_win_rates.png` | 主要阶段的对战胜率提升，是最重要的收益/效果图。 |

推荐报告写法：

> 本项目以最终胜负作为状态收益标签，胜为 +1，负为 -1。收益曲线通过各阶段自我对弈样本平均收益与模型代际评估胜率表示。

### 3. 状态空间与动作定义图

| 图表 | 用途 |
|---|---|
| `state_action_encoding.png` | 展示 15 通道状态编码和 18 维动作编码。 |

放在“状态空间建模与动作定义”章节。

### 4. 训练流程图

| 图表 | 用途 |
|---|---|
| `training_pipeline.png` | 展示规则引擎、MCTS、自我对弈、策略价值网络、full neural-MCTS、最终 v4_full 的整体流程。 |

放在“算法说明”或“训练流程”章节。

### 5. 数据规模图

| 图表 | 用途 |
|---|---|
| `self_play_dataset_growth.png` | 展示从基础 MCTS 到 full neural-MCTS 数据集的对局数和样本数增长。 |

放在“训练数据与训练流程”章节。

### 6. 评估结果图

| 图表 | 用途 |
|---|---|
| `evaluation_progression_win_rates.png` | 展示 v3 对 v2、full-MCTS 对 root-MCTS、v4_full 对 v3 的综合胜率。 |
| `comparison_best_training_metrics.png` | 展示各模型 best val loss 和 best val top1。 |

放在“评估结果”章节。

## 二、每个模型对应的图表

### v1

```text
v1_loss.png
v1_policy_top1.png
v1_value_mae.png
```

说明：基础 MCTS 数据训练的第一版策略价值网络。

### v2

```text
v2_loss.png
v2_policy_top1.png
v2_value_mae.png
```

说明：基础 MCTS + root neural-MCTS 混合数据训练。

### v3

```text
v3_loss.png
v3_policy_top1.png
v3_value_mae.png
```

说明：加入随机开局 root neural-MCTS 数据后训练。

### ResNet v4 实验

```text
resnet_v4_loss.png
resnet_v4_policy_top1.png
resnet_v4_value_mae.png
```

说明：Residual CNN 结构实验。虽然训练指标尚可，但对战弱于 v3，因此不作为最终模型。该组图可以放在“消融实验/失败实验分析”中。

### v4_full 最终模型

```text
v4_full_loss.png
v4_full_policy_top1.png
v4_full_value_mae.png
```

说明：最终模型，使用 full neural-MCTS 生成的高质量数据训练。报告中必须重点展示。

## 三、建议报告图表顺序

建议在 PDF 中按这个顺序放图：

1. `training_pipeline.png`
2. `state_action_encoding.png`
3. `self_play_dataset_growth.png`
4. `v4_full_loss.png`
5. `v4_full_policy_top1.png`
6. `v4_full_value_mae.png`
7. `comparison_validation_loss.png`
8. `comparison_validation_top1.png`
9. `self_play_reward_curve.png`
10. `evaluation_progression_win_rates.png`
11. `comparison_best_training_metrics.png`

如果篇幅足够，再把 v1/v2/v3/ResNet 的单独曲线作为附录。

## 四、还可以补充的重要图表

当前已经生成了报告核心图表，并且已补充以下数据覆盖类图表：

```text
final_dataset_action_distribution.png
final_dataset_dice_distribution.png
final_agent_baseline_win_rates.png
```

其中 `final_agent_baseline_win_rates.png` 会根据已有评估日志自动绘制；如果继续补跑 random、MCTS50、root neural-MCTS 等基线评估，这张图会自动包含更多柱子。

如果还想让报告更完整，可以额外补：

1. **纯神经网络 vs full neural-MCTS 消融图**

   对比 `policy(v4_full)` 和 `full-neural-mcts(v4_full)`，用于说明 MCTS 搜索模块的贡献。

2. **最终模型对不同基线的胜率柱状图**

   例如：

   ```text
   v4_full vs random
   v4_full vs mcts50
   v4_full vs v3
   ```

3. **每局步数分布图**

   使用 `*_games.csv` 里的 `steps` 字段，可以展示智能体对局稳定性和平均结束步数。

4. **胜利原因分布图**

   对比 `goal` 和 `capture_all` 两种胜利方式占比，用于说明智能体主要通过到达目标角还是吃光对方获胜。

5. **动作分布图**

   从 NPZ 数据的 `action_ids` 统计 18 个动作的出现频率，用于说明训练数据覆盖情况。

6. **骰子分布图**

   从 NPZ 数据的 `dice_rolls` 统计骰子 1-6 的分布，用于证明数据生成没有骰子偏置。

## 五、本次新增四类图表

### 1. 纯神经网络 vs full neural-MCTS 消融图

目标图表：

```text
ablation_policy_vs_full_neural_mcts.png
```

这张图需要先跑 `policy(v4_full)` 和 `full-neural-mcts(v4_full)` 的红蓝互换评估。命令已经生成到：

```text
artifacts/figures/report/ADDITIONAL_EVAL_COMMANDS.md
```

评估完成后，重新运行：

```bash
python scripts/generate_report_figures.py
```

脚本会自动生成该图。

### 2. 最终模型对不同基线胜率图

当前已生成：

```text
final_agent_baseline_win_rates.png
```

目前至少包含已有的 `v4_full vs v3` 对比。如果补跑以下评估，它会自动加入更多基线：

```text
v4_full vs random
v4_full vs mcts50
v4_full full-neural-mcts vs v4_full root neural-MCTS
```

命令同样在：

```text
artifacts/figures/report/ADDITIONAL_EVAL_COMMANDS.md
```

### 3. 最终训练集动作分布图

已生成：

```text
final_dataset_action_distribution.png
```

这张图统计最终训练集 `self_play_mixed_v4_full.npz` 中 18 个动作编号的出现次数，可用于说明训练数据覆盖了完整动作空间。

### 4. 最终训练集骰子分布图

已生成：

```text
final_dataset_dice_distribution.png
```

这张图统计最终训练集中的骰子点数分布，可用于说明自我对弈数据中的随机骰子较均匀，没有明显采样偏置。

## 六、一键重新生成图表

如果训练日志或评估结果更新，可以重新运行：

```bash
python scripts/generate_report_figures.py
```

输出目录仍然是：

```text
artifacts/figures/report/
```
