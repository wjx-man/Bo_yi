from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
    args.output.parent.mkdir(parents=True, exist_ok=True)

    merged = merge_npz_datasets(args.inputs)
    np.savez_compressed(args.output, **merged)

    print(f"saved: {args.output}")
    print(f"inputs: {[str(path) for path in args.inputs]}")
    print(f"samples: {merged['states'].shape[0]}")
    print(f"states: {merged['states'].shape}")
    print(f"policies: {merged['policies'].shape}")
    print(f"values: {merged['values'].shape}")
    print(f"games: {len(np.unique(merged['game_ids']))}")
    print(f"value_distribution: {_format_counts(merged['values'])}")
    print(f"winner_distribution: {_format_counts(merged['winners'])}")


def merge_npz_datasets(paths: list[Path]) -> dict[str, np.ndarray]:
    if len(paths) < 2:
        raise ValueError("At least two datasets are required.")

    loaded_parts: list[dict[str, np.ndarray]] = []
    game_id_offset = 0
    for path in paths:
        with np.load(path) as data:
            _validate_fields(path, data.files)
            part = {field: data[field] for field in REQUIRED_FIELDS}

        part = dict(part)
        part["game_ids"] = part["game_ids"].astype(np.int32) + game_id_offset
        loaded_parts.append(part)
        if part["game_ids"].size:
            game_id_offset = int(part["game_ids"].max()) + 1

    first_shapes = {
        field: loaded_parts[0][field].shape[1:]
        for field in REQUIRED_FIELDS
    }
    for part in loaded_parts[1:]:
        for field in REQUIRED_FIELDS:
            if part[field].shape[1:] != first_shapes[field]:
                raise ValueError(
                    f"{field} trailing shape mismatch: "
                    f"{part[field].shape[1:]} != {first_shapes[field]}"
                )

    return {
        field: np.concatenate([part[field] for part in loaded_parts], axis=0)
        for field in REQUIRED_FIELDS
    }


def _validate_fields(path: Path, fields: list[str]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")


def _format_counts(values: np.ndarray) -> dict[float, int]:
    unique_values, counts = np.unique(values, return_counts=True)
    return {
        float(value): int(count)
        for value, count in zip(unique_values, counts)
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple 爱恩斯坦棋 self-play NPZ datasets."
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Input NPZ datasets.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
