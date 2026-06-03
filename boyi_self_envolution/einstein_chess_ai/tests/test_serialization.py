"""验证棋局记录能够无损保存为 JSON 并重新读取。"""

from src.utils.serialization import load_json, save_json


def test_game_record_save_and_load(tmp_path):
    # tmp_path 由 pytest 提供，测试结束后会自动清理临时文件。
    record = {
        "game_id": 1,
        "first_player": "red",
        "initial_board": [[0] * 5 for _ in range(5)],
        "moves": [],
        "winner": "red",
        "win_reason": "reach_corner",
        "total_steps": 0,
    }
    path = tmp_path / "game_000001.json"
    save_json(record, path)
    loaded = load_json(path)
    assert loaded["winner"] == "red"
    assert loaded["win_reason"] == "reach_corner"

