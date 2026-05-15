# Additional Evaluation Commands

Run these commands on the GPU machine, then run `python scripts/generate_report_figures.py` again.

## 1. Pure Network vs Full Neural-MCTS Ablation

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue policy --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_full_v4_red_vs_policy_v4_blue.csv
```

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red policy --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_policy_v4_red_vs_full_v4_blue.csv
```

## 2. Final Agent vs Random

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue random --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_full_v4_red_vs_random_blue.csv
```

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red random --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_random_red_vs_full_v4_blue.csv
```

## 3. Final Agent vs MCTS50

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --mcts-simulations 50 --output artifacts/logs/eval_random_full_v4_red_vs_mcts50_blue.csv
```

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red mcts --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --mcts-simulations 50 --output artifacts/logs/eval_random_mcts50_red_vs_full_v4_blue.csv
```

## 4. Final Full Neural-MCTS vs Root Neural-MCTS

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --neural-mcts-simulations 80 --output artifacts/logs/eval_random_full_v4_red_vs_root_v4_blue.csv
```

```bash
python scripts/evaluate_agents.py --layout random --games 100 --red neural-mcts --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --neural-mcts-simulations 80 --output artifacts/logs/eval_random_root_v4_red_vs_full_v4_blue.csv
```
