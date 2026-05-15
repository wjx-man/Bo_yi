# 爱恩斯坦棋 AI 接入协议

这份文档给外部 AI 作者使用。你们只需要写一个 TCP 客户端，连接裁判平台，按行收发 JSON。裁判平台负责掷骰、给出合法着法、检查走子是否合法、更新棋局、判断胜负。

## 连接方式

平台启动后会监听一个 TCP 地址，例如：

```bash
python online_match.py --host 0.0.0.0 --port 8765 --games 100 --layout random
```

如果要把对战保存为后续训练数据，可以加：

```bash
python online_match.py --host 0.0.0.0 --port 8765 --games 100 --layout random --record-jsonl artifacts/data/online_100_records.jsonl --dataset-output artifacts/data/online_100_dataset.npz
```

保存文件说明：

- `--record-jsonl`: 原始棋谱文件，一行一局，含双方布局、终局结果和完整 `behaviours`
- `--dataset-output`: 训练用 NPZ 文件，字段兼容本项目现有训练数据格式

NPZ 字段：

- `states`: 每步走子前的 15 通道棋盘状态
- `policies`: one-hot 策略标签，表示比赛中实际选择的动作
- `values`: 从该步行棋方视角看的最终胜负，赢为 `1.0`，输为 `-1.0`，未分胜负为 `0.0`
- `players`: 行棋方编码，红方 `1`，蓝方 `-1`
- `dice_rolls`: 当步骰子
- `action_ids`: 18 动作空间里的动作编号
- `game_ids`: 第几局
- `turn_indices`: 回合号
- `winners`: 胜者编码，红方 `1`，蓝方 `-1`，未分胜负 `0`

两个 AI 客户端连接同一个地址：

- 第一个连入的是红方 `red`
- 第二个连入的是蓝方 `blue`
- 每条消息都是一行 UTF-8 JSON，以 `\n` 结束
- 不要在 JSON 字符串里写裸换行

服务端第一条消息是 `welcome`：

```json
{"type":"welcome","role":"red","protocol_version":1,"total_games":100,"layout_mode":"random","peer":"..."}
```

重要字段：

- `role`: 你的颜色，`red` 或 `blue`
- `protocol_version`: 当前为 `1`
- `total_games`: 本次连续对战局数
- `layout_mode`: 平台开局模式，可能是 `default`、`random`、`agent`

## 每局流程

每一局都按下面流程走：

1. 客户端发送一次 `layout`
2. 服务端广播初始 `state`
3. 轮到某方时，服务端给该方的 `state` 中包含 `legal_moves`
4. 行棋方发送 `move`，如果没有合法着法则发送 `pass`
5. 服务端验招、更新棋局、继续广播 `state`
6. 终局时服务端发送 `game_over`
7. 如果还有下一局，客户端继续发送下一局 `layout`

即使平台使用 `--layout default` 或 `--layout random`，客户端也应在每局开始时发送一条 `layout` 用来同步流程；此时平台会忽略你提交的具体布局。只有 `--layout agent` 时，平台会使用双方提交的布局。

## 布局消息

客户端到服务端：

```json
{"type":"layout","order":[1,2,3,4,5,6]}
```

`order` 必须是 `1..6` 的一个排列，表示把哪些棋子放到己方 6 个起始格上。

红方起始格顺序：

```text
[(0,0), (0,1), (0,2), (1,0), (1,1), (2,0)]
```

蓝方起始格顺序：

```text
[(4,4), (4,3), (4,2), (3,4), (3,3), (2,4)]
```

例如红方发送：

```json
{"type":"layout","order":[3,1,2,6,5,4]}
```

含义是：

- 红方棋子 3 放在 `(0,0)`
- 红方棋子 1 放在 `(0,1)`
- 红方棋子 2 放在 `(0,2)`
- 红方棋子 6 放在 `(1,0)`
- 红方棋子 5 放在 `(1,1)`
- 红方棋子 4 放在 `(2,0)`

默认布局就是 `[1,2,3,4,5,6]`。

## 棋盘和坐标

坐标是 `[row, col]`：

- `row` 范围是 `0..4`，向下增大
- `col` 范围是 `0..4`，向右增大
- `[0,0]` 是红方起点角、蓝方目标角
- `[4,4]` 是蓝方起点角、红方目标角

`board` 是 5x5 二维数组：

- `0`: 空格
- `1..6`: 红方对应编号棋子
- `-1..-6`: 蓝方对应编号棋子

示例：

```json
[
  [1,2,3,0,0],
  [4,5,0,0,0],
  [6,0,0,0,-6],
  [0,0,0,-5,-4],
  [0,0,-3,-2,-1]
]
```

## 状态消息

服务端到客户端：

```json
{
  "type":"state",
  "game_index":1,
  "total_games":100,
  "layout_mode":"random",
  "turn_index":1,
  "current_player":"red",
  "dice_roll":4,
  "board":[[1,2,3,0,0],[4,5,0,0,0],[6,0,0,0,-6],[0,0,0,-5,-4],[0,0,-3,-2,-1]],
  "winner":null,
  "your_role":"red",
  "legal_move_count":3,
  "behaviour_count":0,
  "last_behaviour":null,
  "legal_moves":[
    {"color":"red","piece_number":4,"from":[1,0],"to":[2,0]},
    {"color":"red","piece_number":4,"from":[1,0],"to":[1,1]},
    {"color":"red","piece_number":4,"from":[1,0],"to":[2,1]}
  ]
}
```

字段说明：

- `game_index`: 当前第几局，从 `1` 开始
- `total_games`: 总局数
- `turn_index`: 当前回合序号，从 `1` 开始
- `current_player`: 当前行棋方，`red` 或 `blue`
- `dice_roll`: 当前回合骰子点数，范围 `1..6`
- `board`: 当前棋盘
- `winner`: 终局前为 `null`，终局后为 `red` 或 `blue`
- `your_role`: 接收方自己的颜色
- `legal_move_count`: 当前行棋方合法着法数量
- `legal_moves`: 只有轮到你且棋局未结束时才会出现
- `behaviour_count`: 本局已经记录的行为数量
- `last_behaviour`: 上一个行为；开局第一个状态为 `null`

不要自己掷骰。AI 每步应使用服务端发来的 `dice_roll` 和 `legal_moves`。

对方回合时，你通常只会收到 `legal_move_count`，不会看到具体 `legal_moves`。

## 行为记录 behaviour

平台会把双方每一步实际行为广播给双方。你可以用它展示对方刚刚做了什么，也可以保存完整棋谱。

`state.last_behaviour` 示例：

```json
{
  "turn_index":1,
  "color":"red",
  "dice_roll":4,
  "legal_move_count":3,
  "action":"move",
  "move":{"color":"red","piece_number":4,"from":[1,0],"to":[2,1]}
}
```

字段说明：

- `turn_index`: 这个行为发生在哪个回合
- `color`: 做出行为的一方，`red` 或 `blue`
- `dice_roll`: 当回合骰子点数
- `legal_move_count`: 当时裁判给出的合法着法数量
- `action`: `move` 或 `pass`
- `move`: 当 `action == "move"` 时为着法对象；当 `action == "pass"` 时为 `null`

pass 示例：

```json
{
  "turn_index":12,
  "color":"blue",
  "dice_roll":6,
  "legal_move_count":0,
  "action":"pass",
  "move":null
}
```

注意：`state` 只带上一条 `last_behaviour`，避免每步重复发送完整棋谱。客户端如果想实时保存完整行为列表，可以每收到一条新的 `last_behaviour` 就追加到本地数组。终局 `game_over` 会带完整 `behaviours`。

## 走子消息

轮到你时，如果 `legal_moves` 非空，从中选一手，发送：

```json
{"type":"move","piece_number":4,"to":[2,1]}
```

只需要发送：

- `piece_number`: 要走的棋子编号
- `to`: 目标坐标 `[row, col]`

不用发送 `from`。裁判平台会根据棋盘上该编号棋子的唯一位置确定起点。

你发送的走子必须精确匹配当前 `legal_moves` 中的一条。比如上面例子中，`piece_number=4` 且 `to=[2,1]` 是合法的。

如果 `legal_moves` 是空数组，发送：

```json
{"type":"pass"}
```

如果明明有合法着法却发送 `pass`，或发送不在 `legal_moves` 里的着法，平台会认为违规。

## 终局消息

服务端到客户端：

```json
{
  "type":"game_over",
  "game_index":1,
  "total_games":100,
  "winner":"red",
  "reason":"goal",
  "final":{
    "board":[[0,0,0,0,1],[0,0,0,0,0],[0,0,0,0,-6],[0,0,0,-5,-4],[0,0,-3,-2,-1]],
    "turn_index":22,
    "steps":21
  },
  "behaviours":[
    {
      "turn_index":1,
      "color":"red",
      "dice_roll":4,
      "legal_move_count":3,
      "action":"move",
      "move":{"color":"red","piece_number":4,"from":[1,0],"to":[2,1]}
    }
  ]
}
```

常见 `reason`：

- `goal`: 一方走到对角目标格获胜
- `capture_all`: 一方吃光对方棋子获胜
- `illegal_move`: 有一方提交非法着法
- `protocol_error`: 消息格式或类型不符合协议
- `move_timeout`: 单步超时
- `max_turns_exceeded`: 超过平台设置的最大步数，未分胜负

收到 `game_over` 后，如果你希望继续下一局，不要断开连接，直接发送下一局 `layout`。

## 客户端最小伪代码

```python
connect(host, port)
welcome = read_json_line()
role = welcome["role"]

for game in range(welcome.get("total_games", 1)):
    send_json_line({"type": "layout", "order": choose_layout_order(role)})

    while True:
        msg = read_json_line()

        if msg["type"] == "game_over":
            print(msg["winner"], msg["reason"])
            break

        if msg["type"] == "error":
            print("server error:", msg["message"])
            continue

        if msg["type"] != "state":
            continue

        if msg["winner"] is not None:
            continue

        if msg["your_role"] != msg["current_player"]:
            continue

        legal_moves = msg.get("legal_moves")
        if legal_moves is None:
            continue

        if not legal_moves:
            send_json_line({"type": "pass"})
            continue

        chosen = your_ai_choose_move(
            board=msg["board"],
            dice_roll=msg["dice_roll"],
            legal_moves=legal_moves,
            role=role,
        )
        send_json_line({
            "type": "move",
            "piece_number": chosen["piece_number"],
            "to": chosen["to"],
        })
```

## 联调建议

可以先用项目自带随机客户端验证连接流程：

```bash
python online_match.py --host 127.0.0.1 --port 8765 --games 2 --layout agent
python scripts/online_client.py --host 127.0.0.1 --port 8765 --games 2
python scripts/online_client.py --host 127.0.0.1 --port 8765 --games 2
```

也可以用逐步 GUI 观察每一步：

```bash
python scripts/online_step_gui.py --host 127.0.0.1 --port 8765
```

逐步 GUI 会等待两个 AI 客户端连入。棋盘打开后，点击 `下一步` 才会请求当前行棋方提交一手。

## 常见坑

- 每条 JSON 后面必须有换行 `\n`
- `layout.order` 是“起始格顺序上的棋子编号”，不是棋子编号对应的坐标列表
- 只在自己回合行动，判断条件是 `your_role == current_player`
- 不要自己计算所有合法着法后强行提交，直接从平台给的 `legal_moves` 里选
- `move` 消息不要带 `from`，只带 `piece_number` 和 `to`
- 多局对战时，收到 `game_over` 后继续发下一局 `layout`
- 平台负责掷骰；AI 不要在本地重新随机骰子
- 双方看到的 `last_behaviour` / `behaviours` 是公开信息，不包含任何 AI 私有搜索过程
