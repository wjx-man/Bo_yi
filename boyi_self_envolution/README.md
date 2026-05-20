# Einstein Chess AI

Cleaned project folder for pushing to a remote repository.

## Contents

- `einstein_chess_ai/`: AI source code, config, tests, and requirements.
- `game_sim/bo_yi/`: Einstein chess simulator used by the online match server.
- `online_match.py`: JSONL/TCP online match server.
- `online_match_server_launcher.py`: launcher that wires `game_sim/bo_yi` into `sys.path`.
- `online_model_client.py`: model-backed online client.
- `weights/`: retained model weights only:
  - `model.pt`
  - `model_best.pt`
  - `model_pc.pt`

Generated outputs, caches, training histories, checkpoints, logs, reports, and papers are intentionally omitted.

## Setup

```bash
cd einstein_chess_ai
pip install -r requirements.txt
```

## Run

GUI:

```bash
cd einstein_chess_ai
python -m src.play_gui
```

Online server:

```bash
python online_match_server_launcher.py --host 0.0.0.0 --port 8765
```

Online model client:

```bash
python online_model_client.py --host 127.0.0.1 --port 8765
```

By default, `online_model_client.py` uses `weights/model.pt`. Use `--checkpoint weights/model_best.pt` or `--checkpoint weights/model_pc.pt` to select another retained model.

## Tests

```bash
cd einstein_chess_ai
pytest
```
