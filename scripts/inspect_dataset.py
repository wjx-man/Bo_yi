from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from einstein_chess.training import ACTION_SIZE, STATE_CHANNELS


REQUIRED_FIELDS = (
    "states",
    "policies",
    "values",
    "players",
    "dice_rolls",
    "action_ids",
    "game_ids",
    "turn_indices",
    "winners",
)


def main() -> None:
    args = _parse_args()
    ok = inspect_dataset(args.path, strict=args.strict)
    raise SystemExit(0 if ok else 1)


def inspect_dataset(path: Path, strict: bool = False) -> bool:
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return False

    issues: list[str] = []
    with np.load(path) as data:
        missing_fields = [field for field in REQUIRED_FIELDS if field not in data.files]
        if missing_fields:
            print(f"ERROR: missing fields: {missing_fields}")
            return False

        states = data["states"]
        policies = data["policies"]
        values = data["values"]
        players = data["players"]
        dice_rolls = data["dice_rolls"]
        action_ids = data["action_ids"]
        game_ids = data["game_ids"]
        turn_indices = data["turn_indices"]
        winners = data["winners"]

        sample_count = int(states.shape[0]) if states.ndim > 0 else 0
        _check_shapes(
            issues,
            sample_count,
            states,
            policies,
            values,
            players,
            dice_rolls,
            action_ids,
            game_ids,
            turn_indices,
            winners,
        )

        policy_sums = policies.sum(axis=1) if policies.ndim == 2 else np.asarray([])
        max_probs = policies.max(axis=1) if policies.ndim == 2 and sample_count else np.asarray([])
        entropies = _policy_entropy(policies) if policies.ndim == 2 else np.asarray([])

        zero_policy_count = int(np.sum(np.isclose(policy_sums, 0.0))) if policy_sums.size else 0
        bad_policy_sum_count = (
            int(np.sum(np.abs(policy_sums - 1.0) > 1e-4)) if policy_sums.size else 0
        )
        invalid_action_count = int(
            np.sum((action_ids < 0) | (action_ids >= ACTION_SIZE))
        )
        invalid_dice_count = int(np.sum((dice_rolls < 1) | (dice_rolls > 6)))
        invalid_player_count = int(~np.isin(players, [-1, 1]).all())
        invalid_value_count = int(~np.isin(values, [-1.0, 0.0, 1.0]).all())

        if zero_policy_count:
            issues.append(f"{zero_policy_count} samples have empty policy rows.")
        if bad_policy_sum_count:
            issues.append(f"{bad_policy_sum_count} policy rows do not sum to 1.")
        if invalid_action_count:
            issues.append(f"{invalid_action_count} action_ids are outside [0, 17].")
        if invalid_dice_count:
            issues.append(f"{invalid_dice_count} dice_rolls are outside [1, 6].")
        if invalid_player_count:
            issues.append("players must contain only -1 and 1.")
        if invalid_value_count:
            issues.append("values must contain only -1, 0, and 1.")
        if strict and sample_count < 100:
            issues.append("strict mode expects at least 100 samples.")

        print(f"path: {path}")
        print(f"samples: {sample_count}")
        print(f"states: {states.shape} {states.dtype}")
        print(f"policies: {policies.shape} {policies.dtype}")
        print(f"values: {values.shape} {values.dtype}")
        print(f"players: {_format_counts(players)}")
        print(f"values_distribution: {_format_counts(values)}")
        print(f"winners_distribution: {_format_counts(winners)}")
        print(f"dice_distribution: {_format_counts(dice_rolls)}")
        print(f"action_distribution: {_format_counts(action_ids)}")
        print(f"games: {int(len(np.unique(game_ids))) if game_ids.size else 0}")
        print(f"turn_index_min_max: {_min_max(turn_indices)}")
        print(f"policy_sum_min_max: {_min_max(policy_sums)}")
        print(f"policy_max_mean: {_mean(max_probs):.6f}")
        print(f"policy_entropy_mean: {_mean(entropies):.6f}")
        print(f"zero_policy_rows: {zero_policy_count}")
        print(f"bad_policy_sum_rows: {bad_policy_sum_count}")
        print(f"invalid_action_ids: {invalid_action_count}")
        print(f"invalid_dice_rolls: {invalid_dice_count}")

    if issues:
        print("status: FAIL")
        for issue in issues:
            print(f"- {issue}")
        return False

    print("status: OK")
    return True


def _check_shapes(
    issues: list[str],
    sample_count: int,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    players: np.ndarray,
    dice_rolls: np.ndarray,
    action_ids: np.ndarray,
    game_ids: np.ndarray,
    turn_indices: np.ndarray,
    winners: np.ndarray,
) -> None:
    expected_state_shape = (sample_count, STATE_CHANNELS, 5, 5)
    if states.shape != expected_state_shape:
        issues.append(f"states shape should be {expected_state_shape}, got {states.shape}.")
    if policies.shape != (sample_count, ACTION_SIZE):
        issues.append(f"policies shape should be {(sample_count, ACTION_SIZE)}, got {policies.shape}.")
    for name, array in (
        ("values", values),
        ("players", players),
        ("dice_rolls", dice_rolls),
        ("action_ids", action_ids),
        ("game_ids", game_ids),
        ("turn_indices", turn_indices),
        ("winners", winners),
    ):
        if array.shape != (sample_count,):
            issues.append(f"{name} shape should be {(sample_count,)}, got {array.shape}.")


def _policy_entropy(policies: np.ndarray) -> np.ndarray:
    clipped = np.clip(policies, 1e-12, 1.0)
    return -np.sum(policies * np.log(clipped), axis=1)


def _format_counts(values: np.ndarray) -> dict[float, int]:
    if values.size == 0:
        return {}
    unique_values, counts = np.unique(values, return_counts=True)
    return {
        float(value): int(count)
        for value, count in zip(unique_values, counts)
    }


def _min_max(values: np.ndarray) -> tuple[float, float] | None:
    if values.size == 0:
        return None
    return (float(np.min(values)), float(np.max(values)))


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an 爱恩斯坦棋 self-play NPZ dataset."
    )
    parser.add_argument("path", type=Path, help="Path to the NPZ dataset.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable stricter checks for larger training runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
