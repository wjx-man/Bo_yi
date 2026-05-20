# Einstein Chess AI

爱恩斯坦棋 AI 对战模拟器与深度强化学习训练系统，支持人人、人机、机机对战，纯后端自博弈训练，棋局记录、经验回放、评估、曲线输出和报告生成。

## 规则说明

棋盘为 5x5。红方从左上三角区出发，向右、向下、右下移动，目标角点为 `(4,4)`；蓝方从右下三角区出发，向左、向上、左上移动，目标角点为 `(0,0)`。每回合掷骰，若对应编号棋子存活则必须移动该棋子；否则分别寻找小于骰点的最大存活编号和大于骰点的最小存活编号作为候选。目标格有任意棋子都可进入，并移出目标格棋子，包括己方棋子。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行 GUI

```bash
python -m src.play_gui
```

## 后端训练

```bash
python -m src.train --config configs/default.yaml
```

训练会输出 `checkpoints/`、`logs/`、`plots/`、`data/game_records/` 和 `data/replay_buffer/replay.pkl`。

## 评估模型

```bash
python -m src.evaluate --checkpoint checkpoints/model.pt
```

## 回放棋局

```bash
python -m src.replay --record data/game_records/game_000001.json
```

## 运行测试

```bash
pytest
```

## 生成报告

```bash
python scripts/export_report.py
```

若系统安装了 `pandoc`，脚本会额外生成 `docs/report.pdf`；否则保留可提交的 `docs/report.md`。

