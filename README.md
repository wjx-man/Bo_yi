# 爱恩斯坦棋智能对弈项目

本项目是一个基于 Python 实现的爱恩斯坦棋智能对弈系统，包含完整规则引擎、图形界面、人机对弈、AI 对战、15 分钟包干计时、自我对弈数据生成、PyTorch 策略价值网络训练、MCTS 搜索、Neural MCTS 搜索以及第三方线上对战平台接入能力。

项目最终推荐参赛方案为：

```text
policy_value_v4_full_best_loss.pt + full-neural-mcts
```

也就是使用最终训练得到的策略价值网络，并结合完整多层 Neural MCTS 进行搜索决策。

## 一、项目功能

当前项目已经实现：

- 爱恩斯坦棋 5×5 棋盘规则；
- 红蓝双方 1 到 6 号棋子；
- 骰子选子规则；
- 红方向右、向下、向右下移动；
- 蓝方向左、向上、向左上移动；
- 吃子与自吃规则；
- 到达对方出发区角点获胜；
- 吃光对方棋子获胜；
- 手动选择初始布局；
- 随机初始布局；
- Tkinter 图形界面；
- 人类玩家、随机 AI、基础 MCTS、策略价值网络、Root Neural MCTS、Full Neural MCTS；
- 15 分钟包干计时比赛运行器；
- 自我对弈数据导出为 `.npz`；
- PyTorch 策略价值网络训练；
- 训练曲线和评估图表生成；
- 第三方线上对战平台客户端。

## 二、环境要求

基础运行需要：

```text
Python 3.9+
numpy
torch
matplotlib
```

如果只运行规则、GUI 或基础 MCTS，可以使用 CPU。  
如果进行模型训练或大规模 Neural MCTS 评估，建议使用 GPU。

## 三、快速运行

启动图形界面：

```bash
python main.py
```

运行人类对战最终 AI：

```bash
python main.py --red human --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

如果使用 GPU，可以把 `--device cpu` 改成：

```bash
--device cuda
```

## 四、线上比赛客户端

使用最终模型接入线上对战服务器：

```bash
python scripts/competition_client.py --host 172.18.11.154 --port 8913 --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --agent full-neural-mcts --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

如果机器性能足够，可以提高搜索量：

```bash
python scripts/competition_client.py --host 172.18.11.154 --port 8913 --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --agent full-neural-mcts --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample
```

不同版本模型的线上对战命令如下。

V1：

```bash
python scripts/competition_client.py --host 172.18.11.154 --port 8913 --checkpoint artifacts/checkpoints/policy_value_best_loss.pt --device cpu --agent neural-mcts --neural-mcts-simulations 80
```

V2：

```bash
python scripts/competition_client.py --host 172.18.11.154 --port 8913 --checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --device cpu --agent neural-mcts --neural-mcts-simulations 80
```

V3：

```bash
python scripts/competition_client.py --host 172.18.11.154 --port 8913 --checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --device cpu --agent full-neural-mcts --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

V4：

```bash
python scripts/competition_client.py --host 172.18.11.154 --port 8913 --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --agent full-neural-mcts --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

## 五、智能体类型

项目支持以下智能体：

| 智能体名称 | 含义 |
|---|---|
| `human` | 人类玩家，只用于 GUI。 |
| `random` | 随机合法动作 AI。 |
| `mcts` | 基础 MCTS，不使用神经网络。 |
| `policy` | 纯策略价值网络，不进行搜索。 |
| `neural-mcts` | Root Neural MCTS，只在根节点使用神经网络先验和一步价值评估。 |
| `full-neural-mcts` | 完整多层 Neural MCTS，包含多层 PUCT、骰子随机节点和叶子价值回传。 |

最终推荐使用：

```text
full-neural-mcts
```

## 六、自我对弈数据生成

生成基础 MCTS 自我对弈数据：

```bash
python scripts/generate_self_play.py --games 10 --simulations 50 --output artifacts/data/self_play_10g_50s.npz
```

生成 Neural MCTS 自我对弈数据：

```bash
python scripts/generate_self_play.py --agent neural-mcts --checkpoint artifacts/checkpoints/policy_value_best_loss.pt --device cuda --games 500 --simulations 80 --output artifacts/data/self_play_neural_mcts_500g_80s.npz
```

生成随机开局 Neural MCTS 数据：

```bash
python scripts/generate_self_play.py --agent neural-mcts --layout random --checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --device cuda --games 1000 --simulations 80 --output artifacts/data/self_play_neural_mcts_v2_random_1000g_80s.npz
```

生成 Full Neural MCTS 数据：

```bash
python scripts/generate_self_play.py --agent full-neural-mcts --layout random --checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --device cuda --games 1000 --simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/data/self_play_full_neural_mcts_v3_random_1000g_80s.npz
```

检查数据集：

```bash
python scripts/inspect_dataset.py artifacts/data/self_play_10g_50s.npz
```

合并数据集：

```bash
python scripts/merge_datasets.py artifacts/data/self_play_500g_100s.npz artifacts/data/self_play_neural_mcts_500g_80s.npz --output artifacts/data/self_play_mixed_v2_1000g.npz
```

## 七、模型训练

训练一个基础策略价值网络：

```bash
python scripts/train_policy_value.py artifacts/data/self_play_10g_50s.npz --epochs 20 --batch-size 32
```

训练 V1 模型：

```bash
python scripts/train_policy_value.py artifacts/data/self_play_500g_100s.npz --epochs 50 --batch-size 256 --hidden-channels 64 --device cuda --checkpoint artifacts/checkpoints/policy_value_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_best_top1.pt --log artifacts/logs/train_500g_100s_gpu.csv
```

训练 V2 模型：

```bash
python scripts/train_policy_value.py artifacts/data/self_play_mixed_v2_1000g.npz --epochs 60 --batch-size 256 --hidden-channels 64 --device cuda --checkpoint artifacts/checkpoints/policy_value_v2_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_v2_best_top1.pt --log artifacts/logs/train_v2_mixed.csv
```

训练 V3 模型：

```bash
python scripts/train_policy_value.py artifacts/data/self_play_mixed_v3_random.npz --epochs 50 --batch-size 256 --hidden-channels 64 --device cuda --checkpoint artifacts/checkpoints/policy_value_v3_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_v3_best_top1.pt --log artifacts/logs/train_v3_random.csv
```

训练最终 V4 Full 模型：

```bash
python scripts/train_policy_value.py artifacts/data/self_play_mixed_v4_full.npz --epochs 60 --batch-size 256 --hidden-channels 64 --residual-blocks 0 --value-loss-weight 1.0 --lr-scheduler plateau --early-stopping-patience 10 --device cuda --checkpoint artifacts/checkpoints/policy_value_v4_full_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_v4_full_best_top1.pt --log artifacts/logs/train_v4_full.csv
```

## 八、模型评估

评估纯策略价值网络：

```bash
python scripts/evaluate_agents.py --games 50 --red policy --blue random --checkpoint artifacts/checkpoints/policy_value_best_loss.pt --device cuda
```

评估 Root Neural MCTS：

```bash
python scripts/evaluate_agents.py --games 50 --red neural-mcts --blue mcts --checkpoint artifacts/checkpoints/policy_value_best_loss.pt --device cuda --neural-mcts-simulations 80 --mcts-simulations 50
```

评估随机开局下 V3 与 V2：

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red neural-mcts --blue neural-mcts --red-checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --blue-checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --device cuda --neural-mcts-simulations 80 --output artifacts/logs/eval_random_v3_red_vs_v2_blue.csv
```

评估 Full Neural MCTS 与 Root Neural MCTS：

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue neural-mcts --red-checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --blue-checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_full_v3_red_vs_root_v3_blue.csv
```

评估最终模型对基础 MCTS：

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --mcts-simulations 50 --output artifacts/logs/eval_random_full_v4_red_vs_mcts50_blue.csv
```

## 九、图表生成

生成单个训练日志图：

```bash
python scripts/plot_training_log.py artifacts/logs/train_v2_mixed.csv --output-dir artifacts/figures --prefix train_v2_mixed
```

生成报告所需图表：

```bash
python scripts/generate_report_figures.py
```

报告图表默认输出到：

```text
artifacts/figures/report/
```

相关说明文档：

- `docs/CODE_PRESENTATION_WALKTHROUGH.md`：面向代码答辩的展示顺序、逐段注释和常见问题；
- `docs/REPORT_OUTLINE.md`：最终报告大纲；
- `docs/REPORT_FIGURES_AND_TABLES_INDEX.md`：报告图片和表格使用说明；
- `docs/PROJECT_DETAILED_INTRODUCTION.md`：项目详细介绍；
- `docs/DESIGN_AND_TRAINING_PROCESS.md`：设计与训练过程说明；
- `docs/PROJECT_INVENTORY.md`：项目文件和数据说明。

## 十、重要数据与模型

主要训练数据：

| 数据文件 | 说明 |
|---|---|
| `artifacts/data/self_play_500g_100s.npz` | 基础 MCTS 生成的 V1 数据。 |
| `artifacts/data/self_play_mixed_v2_1000g.npz` | V2 混合数据。 |
| `artifacts/data/self_play_mixed_v3_random.npz` | 加入随机开局后的 V3 数据。 |
| `artifacts/data/self_play_mixed_v4_full.npz` | 最终 V4 Full 训练数据。 |

主要模型：

| 模型文件 | 说明 |
|---|---|
| `artifacts/checkpoints/policy_value_best_loss.pt` | V1 模型。 |
| `artifacts/checkpoints/policy_value_v2_best_loss.pt` | V2 模型。 |
| `artifacts/checkpoints/policy_value_v3_best_loss.pt` | V3 模型。 |
| `artifacts/checkpoints/policy_value_v4_full_best_loss.pt` | 最终推荐模型。 |

## 十一、项目结构

```text
bo_yi/
├── main.py
├── online_match.py
├── AI_CLIENT_PROTOCOL.md
├── README.md
├── einstein_chess/
│   ├── engine.py
│   ├── match.py
│   ├── players.py
│   ├── ui.py
│   ├── online_match_client.py
│   ├── agents/
│   │   ├── mcts.py
│   │   ├── policy_value.py
│   │   ├── neural_mcts.py
│   │   └── full_neural_mcts.py
│   └── training/
│       ├── action_codec.py
│       ├── state_encoder.py
│       ├── model.py
│       ├── dataset.py
│       └── self_play.py
├── scripts/
│   ├── competition_client.py
│   ├── generate_self_play.py
│   ├── inspect_dataset.py
│   ├── merge_datasets.py
│   ├── train_policy_value.py
│   ├── evaluate_agents.py
│   ├── plot_training_log.py
│   └── generate_report_figures.py
├── artifacts/
│   ├── data/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
├── docs/
└── tests/
```

## 十二、核心文件说明

| 文件 | 说明 |
|---|---|
| `einstein_chess/engine.py` | 爱恩斯坦棋规则引擎。 |
| `einstein_chess/match.py` | 完整比赛流程与计时模块。 |
| `einstein_chess/ui.py` | Tkinter 图形界面。 |
| `einstein_chess/agents/mcts.py` | 基础 MCTS 智能体。 |
| `einstein_chess/agents/policy_value.py` | 纯策略价值网络智能体。 |
| `einstein_chess/agents/neural_mcts.py` | Root Neural MCTS 智能体。 |
| `einstein_chess/agents/full_neural_mcts.py` | 完整多层 Neural MCTS 智能体。 |
| `einstein_chess/training/state_encoder.py` | 15 通道状态编码。 |
| `einstein_chess/training/action_codec.py` | 18 维动作编码。 |
| `einstein_chess/training/model.py` | PyTorch 策略价值网络。 |
| `einstein_chess/training/self_play.py` | 自我对弈数据生成。 |
| `scripts/competition_client.py` | 第三方线上对战平台客户端。 |
| `scripts/train_policy_value.py` | 模型训练脚本。 |
| `scripts/evaluate_agents.py` | AI 对战评估脚本。 |
| `scripts/generate_report_figures.py` | 报告图表生成脚本。 |

## 十三、测试

运行全部测试：

```bash
python -m unittest discover tests
```

测试覆盖规则引擎、比赛流程、训练数据、模型训练、MCTS、Neural MCTS、数据合并和报告脚本等模块。

## 十四、最终说明

本项目的技术路线可以概括为：

```text
规则引擎
  → 基础 MCTS
  → 自我对弈数据
  → 策略价值网络
  → Root Neural MCTS
  → 随机开局数据增强
  → Full Neural MCTS
  → 最终模型与线上比赛客户端
```

最终系统既可以本地进行人机对弈，也可以通过 `scripts/competition_client.py` 接入线上对战平台，与其他智能体完成完整比赛流程。
