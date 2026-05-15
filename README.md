# 爱恩斯坦棋

This repository contains a playable Python prototype of 爱恩斯坦棋.

Current scope:

- Tkinter GUI
- Human vs human gameplay
- Manual opening layout selection
- Separated rule engine and UI
- Player interface reserved for future AI work
- Timed match runner with 15-minute sudden-death clocks
- Basic MCTS agent for self-play data generation
- 15-channel state encoding, 18-action encoding, and NPZ self-play dataset export
- PyTorch policy-value network training from self-play data
- Policy-value agent inference and match evaluation scripts
- Neural MCTS agent with network priors and value evaluation
- Training curve plotting and CSV evaluation reports

Run:

```bash
python main.py
```

Run human vs neural-MCTS in the GUI:

```bash
python main.py --red human --blue neural-mcts --checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --device cuda --neural-mcts-simulations 80
```

Connect the final model to an online match server:

```bash
python scripts/competition_client.py --host 127.0.0.1 --port 8765 --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --agent full-neural-mcts --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample
```

Run human vs policy network in the GUI:

```bash
python main.py --red human --blue policy --checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --device cuda
```

Generate MCTS self-play data:

```bash
python scripts/generate_self_play.py --games 10 --simulations 50 --output artifacts/data/self_play_10g_50s.npz
```

Generate neural-MCTS self-play data:

```bash
python scripts/generate_self_play.py --agent neural-mcts --checkpoint artifacts/checkpoints/policy_value_best_loss.pt --device cuda --games 500 --simulations 80 --output artifacts/data/self_play_neural_mcts_500g_80s.npz
```

Generate random-layout neural-MCTS self-play data:

```bash
python scripts/generate_self_play.py --agent neural-mcts --layout random --checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --device cuda --games 1000 --simulations 80 --output artifacts/data/self_play_neural_mcts_v2_random_1000g_80s.npz
```

Inspect a generated dataset:

```bash
python scripts/inspect_dataset.py artifacts/data/self_play_10g_50s.npz
```

Merge self-play datasets:

```bash
python scripts/merge_datasets.py artifacts/data/self_play_500g_100s.npz artifacts/data/self_play_neural_mcts_500g_80s.npz --output artifacts/data/self_play_mixed_v2_1000g.npz
```

Train a policy-value network:

```bash
python scripts/train_policy_value.py artifacts/data/self_play_10g_50s.npz --epochs 20 --batch-size 32
```

Train on GPU and save best checkpoints:

```bash
python scripts/train_policy_value.py artifacts/data/self_play_500g_100s.npz --epochs 50 --batch-size 256 --hidden-channels 64 --device cuda --checkpoint artifacts/checkpoints/policy_value_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_best_top1.pt --log artifacts/logs/train_500g_100s_gpu.csv
```

Train a v2 network from mixed data:

```bash
python scripts/train_policy_value.py artifacts/data/self_play_mixed_v2_1000g.npz --epochs 60 --batch-size 256 --hidden-channels 64 --device cuda --checkpoint artifacts/checkpoints/policy_value_v2_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_v2_best_top1.pt --log artifacts/logs/train_v2_mixed.csv
```

Train a residual policy-value network:

```bash
python scripts/train_policy_value.py artifacts/data/self_play_mixed_v3_random.npz --epochs 80 --batch-size 256 --hidden-channels 64 --residual-blocks 4 --value-loss-weight 0.5 --lr-scheduler plateau --early-stopping-patience 10 --device cuda --checkpoint artifacts/checkpoints/policy_value_resnet_v4_latest.pt --best-val-loss-checkpoint artifacts/checkpoints/policy_value_resnet_v4_best_loss.pt --best-val-top1-checkpoint artifacts/checkpoints/policy_value_resnet_v4_best_top1.pt --log artifacts/logs/train_resnet_v4.csv
```

Evaluate a trained policy-value agent:

```bash
python scripts/evaluate_agents.py --games 50 --red policy --blue random --checkpoint artifacts/checkpoints/policy_value_best_loss.pt --device cuda
```

Evaluate a neural MCTS agent:

```bash
python scripts/evaluate_agents.py --games 50 --red neural-mcts --blue mcts --checkpoint artifacts/checkpoints/policy_value_best_loss.pt --device cuda --neural-mcts-simulations 80 --mcts-simulations 50
```

Evaluate neural MCTS with random opening layouts:

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red neural-mcts --blue neural-mcts --red-checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --blue-checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --device cuda --neural-mcts-simulations 80 --output artifacts/logs/eval_random_v3_red_vs_v2_blue.csv
```

Evaluate full multi-layer neural MCTS against root neural MCTS:

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue neural-mcts --red-checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --blue-checkpoint artifacts/checkpoints/policy_value_v3_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_full_v3_red_vs_root_v3_blue.csv
```

Save evaluation results as CSV:

```bash
python scripts/evaluate_agents.py --games 100 --red neural-mcts --blue random --checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt --device cuda --neural-mcts-simulations 80 --output artifacts/logs/eval_v2_neural_mcts_vs_random.csv
```

Plot training curves:

```bash
python scripts/plot_training_log.py artifacts/logs/train_v2_mixed.csv --output-dir artifacts/figures --prefix train_v2_mixed
```

Main files:

- [main.py](main.py)
- [scripts/generate_self_play.py](scripts/generate_self_play.py)
- [scripts/inspect_dataset.py](scripts/inspect_dataset.py)
- [scripts/train_policy_value.py](scripts/train_policy_value.py)
- [scripts/evaluate_agents.py](scripts/evaluate_agents.py)
- [scripts/merge_datasets.py](scripts/merge_datasets.py)
- [scripts/plot_training_log.py](scripts/plot_training_log.py)
- [einstein_chess/agents/mcts.py](einstein_chess/agents/mcts.py)
- [einstein_chess/agents/neural_mcts.py](einstein_chess/agents/neural_mcts.py)
- [einstein_chess/agents/policy_value.py](einstein_chess/agents/policy_value.py)
- [einstein_chess/engine.py](einstein_chess/engine.py)
- [einstein_chess/match.py](einstein_chess/match.py)
- [einstein_chess/players.py](einstein_chess/players.py)
- [einstein_chess/training/action_codec.py](einstein_chess/training/action_codec.py)
- [einstein_chess/training/dataset.py](einstein_chess/training/dataset.py)
- [einstein_chess/training/model.py](einstein_chess/training/model.py)
- [einstein_chess/training/self_play.py](einstein_chess/training/self_play.py)
- [einstein_chess/training/state_encoder.py](einstein_chess/training/state_encoder.py)
- [einstein_chess/ui.py](einstein_chess/ui.py)
- [tests/test_engine.py](tests/test_engine.py)
- [tests/test_match.py](tests/test_match.py)
- [tests/test_merge_datasets.py](tests/test_merge_datasets.py)
- [tests/test_mcts.py](tests/test_mcts.py)
- [tests/test_neural_mcts_agent.py](tests/test_neural_mcts_agent.py)
- [tests/test_policy_value_agent.py](tests/test_policy_value_agent.py)
- [tests/test_policy_value_training.py](tests/test_policy_value_training.py)
- [tests/test_reporting_scripts.py](tests/test_reporting_scripts.py)
- [tests/test_training_data.py](tests/test_training_data.py)
