# 报告图片与表格索引说明

本文档用于整理当前项目中已经生成的所有报告图片和表格，说明它们各自展示的内容、数据来源，以及建议放入最终设计说明文档 PDF 的位置。

图片与汇总表主要位于：

```text
artifacts/figures/report/
```

训练日志和评估日志主要位于：

```text
artifacts/logs/
```

建议最终报告中优先使用 `artifacts/figures/report/` 下的图片和三个汇总 CSV 表格。`artifacts/logs/` 下的原始训练日志、评估日志和逐局结果可以作为附录或实验可复现材料，不需要全部放入正文。

## 一、报告正文建议使用的图片

### 1. 总体流程图

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `training_pipeline.png` | 展示项目从规则引擎、基础 MCTS、自我对弈数据生成、策略价值网络训练，到 Neural MCTS 迭代优化的整体流程。 | 算法总体设计 / 训练流程章节开头 | 说明本项目采用“自我对弈 + MCTS + 策略价值网络”的强化学习路线，训练主要在赛前完成，比赛阶段只进行模型推理和搜索决策。 |
| `state_action_encoding.png` | 展示 15 通道状态编码和 18 维动作编码的结构。 | 状态空间建模与动作定义章节 | 说明棋盘状态如何被转换成神经网络输入，动作如何被统一编码成固定长度策略输出。 |

这两张图建议放在报告前半部分，用来帮助读者先理解项目结构和数据接口。

## 二、各版本模型训练曲线

每个版本模型都有三类训练图：loss 曲线、策略头 top-1 曲线、价值头 MAE 曲线。它们应该放在“训练流程与实验结果”章节中。

### 1. V1 模型：基础 MCTS 数据训练

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `v1_loss.png` | V1 模型训练集与验证集 loss 曲线。 | V1 训练过程小节 | 说明 V1 使用基础 MCTS 自我对弈数据训练，是第一个策略价值网络。验证 loss 后期上升，说明数据规模较小且存在过拟合趋势。 |
| `v1_policy_top1.png` | V1 策略头 top-1 准确率曲线。 | V1 训练过程小节 | 说明网络已经能学习基础 MCTS 的动作偏好，但准确率上限有限。 |
| `v1_value_mae.png` | V1 价值头平均绝对误差曲线。 | V1 训练过程小节 | 说明价值头学习终局胜负信号，但早期数据质量较弱，价值估计稳定性一般。 |

### 2. V2 模型：混合基础 MCTS 与 Root Neural MCTS 数据训练

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `v2_loss.png` | V2 模型 loss 曲线。 | V2 训练过程小节 | 说明 V2 合并了基础 MCTS 和第一版 Neural MCTS 数据，验证 loss 相比 V1 改善。 |
| `v2_policy_top1.png` | V2 策略头 top-1 准确率曲线。 | V2 训练过程小节 | 说明策略头对搜索分布的拟合能力提升。 |
| `v2_value_mae.png` | V2 价值头 MAE 曲线。 | V2 训练过程小节 | 说明加入更强自我对弈数据后，价值预测更加稳定。 |

### 3. V3 模型：随机开局数据训练

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `v3_loss.png` | V3 模型 loss 曲线。 | V3 训练过程小节 | 说明 V3 加入随机初始布局数据，提升模型对不同开局的适应能力。 |
| `v3_policy_top1.png` | V3 策略头 top-1 准确率曲线。 | V3 训练过程小节 | 说明随机布局没有破坏策略学习，反而提高了验证集泛化表现。 |
| `v3_value_mae.png` | V3 价值头 MAE 曲线。 | V3 训练过程小节 | 说明随机开局数据让价值头面对更多局面，训练更贴近真实比赛。 |

### 4. ResNet V4 试验模型：残差网络结构尝试

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `resnet_v4_loss.png` | 残差 CNN 试验模型 loss 曲线。 | 模型结构消融实验小节 | 说明项目尝试了更复杂的 Residual CNN，但最终实战评估没有超过 V3。 |
| `resnet_v4_policy_top1.png` | 残差 CNN 策略头 top-1 准确率曲线。 | 模型结构消融实验小节 | 说明虽然验证 top-1 表现不差，但离线指标不完全等价于实战棋力。 |
| `resnet_v4_value_mae.png` | 残差 CNN 价值头 MAE 曲线。 | 模型结构消融实验小节 | 说明复杂模型可能需要更多数据支撑，否则容易出现泛化不足。 |

### 5. V4 Full 模型：最终模型

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `v4_full_loss.png` | 最终 V4 Full 模型 loss 曲线。 | 最终模型训练小节 | 说明最终模型使用 Full Neural MCTS 生成的数据训练，并加入 early stopping，最终选择 best validation loss checkpoint。 |
| `v4_full_policy_top1.png` | 最终 V4 Full 模型策略头 top-1 曲线。 | 最终模型训练小节 | 说明最终模型策略头验证准确率达到较高水平，能较好拟合搜索产生的动作分布。 |
| `v4_full_value_mae.png` | 最终 V4 Full 模型价值头 MAE 曲线。 | 最终模型训练小节 | 说明价值头可为多层 MCTS 叶子节点提供局面评估。 |

## 三、跨版本对比图

这些图建议放在“纵向对比实验”或“模型迭代结果”章节中，用来说明模型从 V1 到 V4 的演进效果。

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `comparison_validation_loss.png` | 对比 V1、V2、V3、ResNet V4、V4 Full 的验证 loss 曲线。 | 训练过程对比章节 | 说明随着数据迭代和搜索质量提升，验证 loss 整体改善；同时指出 ResNet 结构虽然更复杂，但未成为最终方案。 |
| `comparison_validation_top1.png` | 对比不同模型的验证策略 top-1 准确率。 | 训练过程对比章节 | 说明策略头学习能力随训练数据质量提升而增强，V3 和 V4 Full 是主要有效版本。 |
| `comparison_best_training_metrics.png` | 汇总每个模型的最佳验证 loss 和最佳验证 top-1。 | 超参数与训练结果汇总章节 | 可作为一个总览图，说明最终选择 V4 Full 的依据之一是验证 loss 与实战评估共同较优。 |
| `evaluation_progression_win_rates.png` | 展示关键阶段的胜率进展，例如 V3 对 V2、Full MCTS 对 Root MCTS、V4 Full 对 V3。 | 评估结果章节 | 说明项目不是只看训练 loss，而是通过 AI 对战胜率验证模型和搜索模块是否真正增强。 |

## 四、自我对弈数据与收益相关图

这些图建议放在“训练数据生成”或“自我对弈训练流程”章节中。

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `self_play_dataset_growth.png` | 展示各阶段自我对弈数据的局数和样本数量。 | 训练数据生成章节 | 说明数据从基础 MCTS 数据逐步扩展到随机开局、Full Neural MCTS 数据，最终训练集达到 3000 局、53583 个样本。 |
| `self_play_reward_curve.png` | 展示各阶段数据中的平均价值和红方胜率变化。 | 收益曲线章节 | 需要说明这不是“每个 epoch 的训练收益”，而是自我对弈数据阶段的收益统计，用来反映不同数据生成阶段的胜负信号变化。 |
| `final_dataset_action_distribution.png` | 展示最终训练集 18 个动作编码的分布。 | 数据质量分析章节 | 说明最终数据覆盖了所有动作编码，不存在动作长期缺失的问题。 |
| `final_dataset_dice_distribution.png` | 展示最终训练集骰子点数 1 到 6 的分布。 | 数据质量分析章节 | 说明骰子样本分布基本均衡，训练数据没有明显偏向某个骰子点数。 |

## 五、搜索模块与最终模型评估图

这些图建议放在“算法消融实验”和“最终运行结果”章节中。

| 图片文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `ablation_policy_vs_full_neural_mcts.png` | 对比纯策略价值网络和同一网络结合 Full Neural MCTS 后的表现。 | 消融实验章节 | 说明 MCTS 搜索模块对棋力有实际提升，证明最终方案不是单纯依赖神经网络推理。 |
| `final_agent_baseline_win_rates.png` | 展示最终智能体对随机智能体、基础 MCTS 等基线的胜率。 | 最终评估结果章节 | 说明最终 `V4 Full + Full Neural MCTS` 智能体在基线对战中具有优势。 |

如果这两张图缺少某些对照项，说明对应评估日志还没有生成。可以参考 `artifacts/figures/report/ADDITIONAL_EVAL_COMMANDS.md` 中的命令补充实验。

## 六、报告正文建议使用的汇总表格

### 1. 训练结果汇总表

| 表格文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `training_summary.csv` | 汇总每个模型的训练日志、训练 epoch 数、最佳验证 loss、最佳验证 top-1、最终训练 loss、最终验证 loss 和备注。 | 训练流程与超参数设定章节 | 建议整理成“各版本模型训练结果表”，用于说明 V1、V2、V3、ResNet V4、V4 Full 的训练结果。 |

表格中的关键内容包括：

| 模型 | 主要意义 |
|---|---|
| V1 MCTS data | 基础 MCTS 数据训练出的第一版策略价值网络。 |
| V2 mixed | 合并基础 MCTS 和 Root Neural MCTS 数据后的模型。 |
| V3 random layout | 加入随机开局数据后的模型。 |
| ResNet V4 trial | 残差网络结构试验，最终未采用。 |
| V4 full-MCTS final | 使用 Full Neural MCTS 数据训练的最终模型。 |

### 2. 数据集汇总表

| 表格文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `dataset_summary.csv` | 汇总每个自我对弈数据集的局数、样本数、平均价值、正负样本比例、红蓝胜率、策略熵和备注。 | 训练数据生成章节 | 建议整理成“自我对弈数据集统计表”，说明数据是如何从 D0 增长到 D6 的。 |

表格中的关键数据集包括：

| 数据集 | 主要意义 |
|---|---|
| D0 MCTS | 基础 MCTS 自我对弈数据。 |
| D1 root NMCTS | Root Neural MCTS 自我对弈数据。 |
| D2 mixed V2 | V2 训练用混合数据。 |
| D3 random root | 随机开局 Root Neural MCTS 数据。 |
| D4 mixed V3 | V3 训练用混合数据。 |
| D5 full NMCTS | Full Neural MCTS 生成的数据。 |
| D6 final mix | 最终 V4 Full 训练数据。 |

### 3. 评估结果汇总表

| 表格文件 | 内容说明 | 建议放置位置 | 报告中应该说明什么 |
|---|---|---|---|
| `evaluation_summary.csv` | 汇总关键对比实验的胜率，例如 V3 vs V2、Full MCTS vs Root MCTS、ResNet V4 vs V3、V4 Full vs V3。 | 评估结果章节 | 建议整理成“主要模型对战结果表”，说明最终方案经过了纵向版本对比和模块消融验证。 |

表格中的关键结论包括：

| 对比实验 | 结论 |
|---|---|
| V3 vs V2 | V3 在随机开局下整体强于 V2，说明随机布局数据有效。 |
| Full MCTS vs Root | 多层 Full Neural MCTS 强于只在根节点搜索的 Root Neural MCTS。 |
| ResNet V4 vs V3 | 残差 CNN 实战表现较弱，因此未作为最终模型。 |
| V4 Full vs V3 | 最终模型在 Full Neural MCTS 条件下略强于 V3。 |

## 七、训练日志表格

训练日志位于：

```text
artifacts/logs/
```

| 表格文件 | 内容说明 | 建议放置位置 |
|---|---|---|
| `train_500g_100s.csv` | 本地或早期 V1 训练日志。 | 可不放正文，作为附录或复现实验材料。 |
| `train_500g_100s_gpu.csv` | V1 GPU 训练日志，是 V1 图表的主要来源。 | 可在附录列出。 |
| `train_v2_mixed.csv` | V2 混合数据训练日志。 | 可在附录列出。 |
| `train_v3_random.csv` | V3 随机开局数据训练日志。 | 可在附录列出。 |
| `train_resnet_v4.csv` | ResNet V4 结构试验训练日志。 | 可在模型结构消融实验附录列出。 |
| `train_v4_full.csv` | 最终 V4 Full 模型训练日志。 | 建议在正文引用关键指标，完整表放附录。 |
| `train_smoke.csv` | 训练脚本冒烟测试日志。 | 不建议放入正式报告正文。 |

这些 CSV 的每一行通常对应一个 epoch，包含训练 loss、验证 loss、策略准确率、价值误差、学习率等信息。最终报告正文中不建议直接贴完整训练日志，而应该使用折线图和汇总表。

## 八、评估日志表格

评估日志同样位于：

```text
artifacts/logs/
```

评估日志分两类：

1. `eval_*.csv`：每次评估的汇总结果。
2. `eval_*_games.csv`：每一局比赛的详细结果。

### 1. 汇总评估表

| 表格文件 | 内容说明 | 建议放置位置 |
|---|---|---|
| `eval_random_v3_red_vs_v2_blue.csv` | 随机开局下，V3 执红对 V2。 | 用于 V3 vs V2 纵向对比。 |
| `eval_random_v2_red_vs_v3_blue.csv` | 随机开局下，V2 执红对 V3。 | 与上一项合并，消除先后手影响。 |
| `eval_random_full_v3_red_vs_root_v3_blue.csv` | V3 网络下，Full Neural MCTS 执红对 Root Neural MCTS。 | 用于搜索模块消融。 |
| `eval_random_root_v3_red_vs_full_v3_blue.csv` | V3 网络下，Root Neural MCTS 执红对 Full Neural MCTS。 | 与上一项合并，说明 Full 搜索整体更强。 |
| `eval_random_resnet_v4_red_vs_v3_blue.csv` | ResNet V4 执红对 V3。 | 用于说明 ResNet 试验未被采用。 |
| `eval_random_v3_red_vs_resnet_v4_blue.csv` | V3 执红对 ResNet V4。 | 与上一项合并。 |
| `eval_random_full_v4_loss_red_vs_full_v3_blue.csv` | V4 Full 执红对 V3。 | 用于最终模型对比。 |
| `eval_random_full_v3_red_vs_full_v4_loss_blue.csv` | V3 执红对 V4 Full。 | 与上一项合并。 |
| `eval_v3_best_loss_red_vs_random.csv` | V3 对随机智能体，V3 执红。 | 可作为基线对比材料。 |
| `eval_v3_best_loss_blue_vs_random.csv` | V3 对随机智能体，V3 执蓝。 | 可作为基线对比材料。 |
| `eval_v3_best_loss_red_vs_mcts50.csv` | V3 对基础 MCTS，V3 执红。 | 可作为基线对比材料。 |
| `eval_v3_best_loss_blue_vs_mcts50.csv` | V3 对基础 MCTS，V3 执蓝。 | 可作为基线对比材料。 |
| `eval_v3_red_vs_v2_blue.csv` | 固定或默认布局下，V3 执红对 V2。 | 早期对比记录，可放附录。 |
| `eval_v2_red_vs_v3_blue.csv` | 固定或默认布局下，V2 执红对 V3。 | 早期对比记录，可放附录。 |

### 2. 逐局评估表

所有以 `_games.csv` 结尾的文件记录逐局结果，例如胜者、步数、胜利原因等。它们主要用于复核统计结果和生成更细粒度图表。

| 表格类型 | 内容说明 | 建议放置位置 |
|---|---|---|
| `eval_*_games.csv` | 每局比赛的详细记录。 | 不建议放入正文，可作为附录或实验数据文件提交。 |

正文中只需要引用由这些逐局记录汇总出的胜率、平均步数和胜利原因统计即可。

## 九、辅助说明文档与命令表

| 文件 | 内容说明 | 建议用途 |
|---|---|---|
| `FIGURE_INDEX.md` | 自动生成的英文版图表索引。 | 可作为本中文文档的来源，不建议直接放入报告。 |
| `ADDITIONAL_EVAL_COMMANDS.md` | 建议补充运行的评估命令。 | 如果报告中还缺少最终模型对随机、MCTS、纯策略网络的完整对比，可以按此文件补实验。 |

## 十、最终报告推荐插图顺序

建议最终报告正文按以下顺序插入图表：

1. `training_pipeline.png`：项目总体训练流程。
2. `state_action_encoding.png`：状态编码与动作编码。
3. `self_play_dataset_growth.png`：自我对弈数据规模增长。
4. `final_dataset_dice_distribution.png`：骰子分布。
5. `final_dataset_action_distribution.png`：动作分布。
6. `v1_loss.png`、`v2_loss.png`、`v3_loss.png`、`v4_full_loss.png`：各阶段 loss 曲线。
7. `comparison_validation_loss.png`：跨版本验证 loss 对比。
8. `comparison_validation_top1.png`：跨版本策略准确率对比。
9. `self_play_reward_curve.png`：自我对弈阶段收益曲线。
10. `ablation_policy_vs_full_neural_mcts.png`：纯网络与搜索增强对比。
11. `evaluation_progression_win_rates.png`：关键阶段胜率进展。
12. `final_agent_baseline_win_rates.png`：最终模型对基线结果。

如果报告篇幅有限，可以把每个版本的 `policy_top1` 和 `value_mae` 曲线放到附录，只在正文保留 loss 曲线和跨版本对比图。

## 十一、最终报告推荐表格顺序

建议正文保留以下三张表：

1. **自我对弈数据集统计表**：由 `dataset_summary.csv` 整理而来。
2. **各版本模型训练结果表**：由 `training_summary.csv` 整理而来。
3. **主要对战评估结果表**：由 `evaluation_summary.csv` 整理而来。

建议附录保留以下内容：

1. 原始训练日志文件列表。
2. 原始评估日志文件列表。
3. 每局评估结果 `_games.csv` 文件说明。
4. 补充评估命令 `ADDITIONAL_EVAL_COMMANDS.md`。

## 十二、写报告时需要注意的表述

1. `self_play_reward_curve.png` 应称为“自我对弈阶段收益统计曲线”，不要写成“每个 epoch 的强化学习 reward 曲线”。
2. 训练 loss 只能说明模型拟合数据的程度，最终棋力还需要结合 AI 对战胜率判断。
3. ResNet V4 是一次结构改进实验，但实验结果没有超过 V3，因此最终采用的是 V4 Full 数据训练出的 CNN 模型，而不是 ResNet 模型。
4. 最终参赛方案不是纯神经网络，而是 `policy_value_v4_full_best_loss.pt` 与 `full-neural-mcts` 搜索结合。
5. 如果报告需要强调“完整对弈能力”，应结合计时模块、人机对弈、AI 对战脚本和线上平台接入脚本一起说明。

