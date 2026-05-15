# Report Figure Index

This directory contains generated figures for the final design report.

| Figure | Report section | Description |
|---|---|---|
| `v1_loss.png` | Training process / loss curve | V1 MCTS data loss curves. Initial policy-value network trained from basic MCTS self-play. |
| `v1_policy_top1.png` | Training process / policy accuracy | V1 MCTS data policy-head top-1 accuracy. |
| `v1_value_mae.png` | Training process / value accuracy | V1 MCTS data value-head MAE curve. |
| `v2_loss.png` | Training process / loss curve | V2 mixed loss curves. Mixed basic MCTS and root neural-MCTS self-play. |
| `v2_policy_top1.png` | Training process / policy accuracy | V2 mixed policy-head top-1 accuracy. |
| `v2_value_mae.png` | Training process / value accuracy | V2 mixed value-head MAE curve. |
| `v3_loss.png` | Training process / loss curve | V3 random layout loss curves. Added random-opening root neural-MCTS data. |
| `v3_policy_top1.png` | Training process / policy accuracy | V3 random layout policy-head top-1 accuracy. |
| `v3_value_mae.png` | Training process / value accuracy | V3 random layout value-head MAE curve. |
| `resnet_v4_loss.png` | Training process / loss curve | ResNet V4 trial loss curves. Residual CNN experiment; rejected after match evaluation. |
| `resnet_v4_policy_top1.png` | Training process / policy accuracy | ResNet V4 trial policy-head top-1 accuracy. |
| `resnet_v4_value_mae.png` | Training process / value accuracy | ResNet V4 trial value-head MAE curve. |
| `v4_full_loss.png` | Training process / loss curve | V4 full-MCTS final loss curves. Final CNN model trained with full neural-MCTS self-play data. |
| `v4_full_policy_top1.png` | Training process / policy accuracy | V4 full-MCTS final policy-head top-1 accuracy. |
| `v4_full_value_mae.png` | Training process / value accuracy | V4 full-MCTS final value-head MAE curve. |
| `comparison_validation_loss.png` | Training process / cross-generation comparison | Compares val_total_loss for all model generations. |
| `comparison_validation_top1.png` | Training process / cross-generation comparison | Compares val_policy_top1 for all model generations. |
| `comparison_best_training_metrics.png` | Training process / hyperparameter results | Best validation loss and policy top-1 for each trained model. |
| `self_play_dataset_growth.png` | Training data / self-play scale | Number of games and training samples for each self-play dataset. |
| `self_play_reward_curve.png` | Training process / reward curve | Reward-oriented curve from self-play data: mean value and red-side winner rate by data generation stage. |
| `evaluation_progression_win_rates.png` | Evaluation / longitudinal comparison | Win-rate comparison for the main model and search improvements. |
| `training_pipeline.png` | Algorithm overview / training pipeline | Overall self-play reinforcement learning pipeline used by the project. |
| `state_action_encoding.png` | State/action modeling | Visual summary of the 15-channel state encoding and 18-action encoding. |
| `final_dataset_action_distribution.png` | Training data / action distribution | Distribution of the 18 encoded actions in the final V4_full training dataset. |
| `final_dataset_dice_distribution.png` | Training data / dice distribution | Distribution of dice rolls 1-6 in the final V4_full training dataset. |
| `ablation_policy_vs_full_neural_mcts.png` | Ablation / search module | Pure policy-value network compared with the same network guided by full neural-MCTS. |
| `final_agent_baseline_win_rates.png` | Evaluation / final baselines | Final V4_full full-neural-MCTS agent win rates against available baselines. |

Recommended final model: `policy_value_v4_full_best_loss.pt`.