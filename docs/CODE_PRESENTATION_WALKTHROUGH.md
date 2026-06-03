# 爱恩斯坦棋项目代码答辩讲解指南

本文档用于在向老师展示代码时，按照一条清晰的主线讲解项目。目标不是逐行朗读代码，而是让老师理解：

```text
规则如何建模
→ AI 如何获得合法走法
→ 基础 MCTS 如何冷启动
→ 自我对弈如何生成训练数据
→ 策略价值网络如何学习
→ V1、V2、V3、V4 Full 如何演进
→ 最终智能体如何参加比赛
```

最终参赛方案：

```text
policy_value_v4_full_best_loss.pt + full-neural-mcts
```

---

## 一、先讲整体，不要立刻打开代码

建议先用 1 分钟说完下面这段：

> 这是一个完整的爱恩斯坦棋智能对弈项目。项目首先实现统一的规则引擎，保证 GUI、搜索、自我对弈和线上比赛使用同一套规则。之后使用基础 MCTS 进行冷启动，生成第一批自我对弈数据并训练 V1 策略价值网络。V1 再用于指导 Root Neural MCTS，生成 V2 数据；V2 加入随机开局数据训练 V3；最后使用 V3 指导 Full Neural MCTS 进行多层搜索，生成最终 V4 Full 数据。比赛阶段使用 V4 Full 网络结合 Full Neural MCTS 决策。

可以画出下面的流程：

```text
engine.py
  ↓
mcts.py
  ↓
self_play.py
  ↓
state_encoder.py + action_codec.py
  ↓
model.py
  ↓
neural_mcts.py
  ↓
full_neural_mcts.py
  ↓
competition_client.py
```

---

## 二、建议给老师看的文件顺序

不要按照文件夹顺序展示。按照“规则 → 决策 → 学习 → 实战”的顺序展示。

| 顺序 | 文件 | 主要讲解目标 |
|---:|---|---|
| 1 | `einstein_chess/engine.py` | 证明规则建模完整，所有模块共用同一套游戏状态。 |
| 2 | `einstein_chess/agents/mcts.py` | 讲项目如何在没有数据、没有模型时完成冷启动。 |
| 3 | `einstein_chess/training/self_play.py` | 讲 MCTS 如何自己和自己下棋，并生成训练样本。 |
| 4 | `einstein_chess/training/state_encoder.py` | 讲 `15 × 5 × 5` 状态输入。 |
| 5 | `einstein_chess/training/action_codec.py` | 讲 `18` 维动作空间和策略答案。 |
| 6 | `einstein_chess/training/model.py` | 讲策略价值网络的输入、策略头和价值头。 |
| 7 | `einstein_chess/agents/neural_mcts.py` | 讲 V1 如何参与搜索并生成 V2 数据。 |
| 8 | `einstein_chess/agents/full_neural_mcts.py` | 讲最终多层搜索、骰子 chance node 和价值回传。 |
| 9 | `einstein_chess/match.py` | 讲比赛计时、非法动作判负和完整比赛流程。 |
| 10 | `scripts/competition_client.py` | 讲最终模型如何接入线上比赛平台。 |

时间较短时，优先展示前 8 个文件。`main.py`、GUI、图表脚本和测试文件可以作为补充。

---

# 第一部分：规则引擎

## 三、`einstein_chess/engine.py`

### 这个文件做什么

这个文件是整个项目的基础。它负责：

```text
红蓝双方定义
棋子和走法定义
棋盘初始化
骰子规则
候选棋子规则
合法走法生成
走棋和吃子
胜负判断
局面快照
```

可以这样对老师说：

> 搜索算法不应该自己判断规则，神经网络也不应该自己判断规则。所有 AI 都只能向规则引擎询问合法走法，并通过规则引擎执行走法，这样可以保证训练和比赛规则一致。

---

### `PlayerColor`

```python
class PlayerColor(str, Enum):
    RED = "red"
    BLUE = "blue"
```

它表示玩家颜色。

重点属性：

```text
opponent        获取对手颜色
goal            获取目标角
deltas          获取允许移动的三个方向
start_positions 获取六个开局位置
```

红方目标角：

```text
(4, 4)
```

蓝方目标角：

```text
(0, 0)
```

红方移动方向：

```text
向下、向右、向右下
```

蓝方移动方向：

```text
向上、向左、向左上
```

---

### `Piece`

```python
@dataclass(frozen=True)
class Piece:
    color: PlayerColor
    number: int
```

一枚棋子由颜色和编号组成。

例如：

```python
Piece(PlayerColor.RED, 3)
```

表示红方 3 号棋子。

---

### `Move`

```python
@dataclass(frozen=True)
class Move:
    color: PlayerColor
    piece_number: int
    from_position: Position
    to_position: Position
```

一条走法记录：

```text
谁走
走哪枚棋子
从哪里走
走到哪里
```

例如：

```python
Move(PlayerColor.RED, 3, (2, 1), (3, 2))
```

表示红方 3 号棋子从 `(2, 1)` 向右下走到 `(3, 2)`。

---

### `GameSnapshot`

```python
@dataclass(frozen=True)
class GameSnapshot:
    board: tuple[tuple[int, ...], ...]
    current_player: PlayerColor
    dice_roll: int
    winner: PlayerColor | None
    legal_moves: tuple[Move, ...]
    turn_index: int
```

这是提供给 AI 的只读局面。

可以这样解释：

> AI 不直接修改真正的棋局对象，只接收一个局面快照。快照包含棋盘、当前玩家、骰子、合法走法和回合数。这样可以减少 AI 意外修改比赛状态的风险。

棋盘编码规则：

```text
0      空格
1~6    红方 1~6 号棋子
-1~-6  蓝方 1~6 号棋子
```

---

### `EinsteinGame.__init__`

这个构造函数创建一盘棋。

主要状态：

```text
board          5 × 5 棋盘
current_player 当前行动方
dice_roll      当前骰子
winner         当前胜者
turn_index     回合编号
move_history   历史走法
```

最后调用 `reset()` 完成棋盘初始化。

---

### `default_layout_for` 和 `random_layout_for`

```python
default_layout_for(...)
random_layout_for(...)
```

这两个函数分别生成默认开局和随机开局。

随机开局是 V3 的关键：

> V1 和 V2 主要学习固定开局，V3 通过随机打乱 1 到 6 号棋子的摆放位置，提升模型面对不同开局时的泛化能力。

---

### `clone`

```python
def clone(self) -> "EinsteinGame":
```

这个函数复制当前棋局。

它对搜索算法非常重要：

> MCTS 会进行大量假想比赛。每次假想比赛必须在复制棋盘上运行，不能影响真正的比赛棋盘。

---

### `get_candidate_numbers`

```python
def get_candidate_numbers(self, color, dice_roll) -> list[int]:
```

这个函数实现骰子选子规则。

例子：

```text
骰子掷出 3，3 号棋子仍在棋盘上
→ 只能选择 3 号棋子
```

```text
骰子掷出 3，3 号棋子已经被吃掉
当前还剩 1、2、5、6 号棋子
→ 可以选择最接近 3 的 2 号和 5 号棋子
```

这是爱恩斯坦棋区别于普通棋类的重要规则。

---

### `get_legal_moves`

```python
def get_legal_moves(self, color=None, dice_roll=None) -> list[Move]:
```

这个函数生成所有合法走法。

执行过程：

```text
根据骰子找到允许移动的棋子编号
→ 找到棋子位置
→ 根据红蓝方允许方向生成目标位置
→ 去掉棋盘外的位置
→ 返回 Move 列表
```

需要强调：

> 目标位置即使有己方棋子也仍然合法，因为爱恩斯坦棋允许自吃。

---

### `apply_move`

```python
def apply_move(self, move: Move) -> PlayerColor | None:
```

这个函数真正执行走法。

执行过程：

```text
检查游戏是否已经结束
→ 检查走法是否合法
→ 清空原位置
→ 在目标位置放入移动棋子
→ 如果目标位置原来有棋子，它会被覆盖，相当于被吃掉
→ 判断胜负
→ 如果未结束，切换玩家并重新掷骰子
```

老师可能会问“吃子在哪里实现”。

回答：

> 目标位置不需要单独删除棋子，因为给 `board[to_row][to_col]` 重新赋值时，原棋子会被覆盖，这同时支持吃对方棋子和自吃。

---

### `_resolve_winner`

胜利条件有两个：

```text
到达对方目标角
吃光对方所有棋子
```

---

### 这一部分的总结话术

> `engine.py` 将爱恩斯坦棋规则集中在一个模块中。之后无论是 GUI、人类玩家、MCTS、自我对弈还是线上客户端，都不重复实现规则，而是调用 `get_legal_moves()` 和 `apply_move()`。这保证训练得到的 AI 在比赛中不会因为规则不一致而失效。

---

# 第二部分：基础 MCTS 冷启动

## 四、`einstein_chess/agents/mcts.py`

### 这个文件做什么

这个文件实现项目最早期的基础 MCTS 智能体。

当时项目没有：

```text
人工棋谱
训练数据
神经网络模型
```

所以需要一个只依赖规则引擎、可以自己下棋并生成数据的第一代智能体。

需要准确说明：

> 这个版本只在当前根局面统计各个合法走法，没有保存完整多层搜索树。它更接近“根节点 UCT 选择 + 启发式随机 rollout”。

---

### `MCTSSearchStats`

```python
@dataclass
class MCTSSearchStats:
    visits: int = 0
    value_sum: float = 0.0
```

每个合法走法保存两个数字：

```text
visits     被测试了多少次
value_sum  所有测试结果的总分
```

平均价值：

```text
mean_value = value_sum / visits
```

---

### `MCTSAgent.__init__`

重要参数：

```text
simulations       每次真正走棋前进行多少次假想比赛
exploration       UCT 探索系数
max_rollout_steps 每次假想比赛最多继续多少步
rng               随机数生成器
```

输出记录：

```text
last_visit_counts 每个走法被访问多少次
last_policy       访问次数归一化后的策略分布
```

`last_policy` 后面会成为神经网络的策略答案。

---

### `choose_move`

这是基础 MCTS 的核心入口。

可以按下面的顺序逐段讲：

```text
1. 接收当前局面和合法走法。
2. 如果存在一步获胜走法，直接选择。
3. 为每个合法走法创建统计信息。
4. 重复进行 simulations 次假想比赛。
5. 每次选择一个根节点走法。
6. 复制棋局并执行该走法。
7. 继续 rollout，得到胜负或局面评分。
8. 更新该走法的访问次数和总分。
9. 最终选择访问次数最多的走法。
```

为什么最终优先选择访问次数最多的走法：

> 某个走法只测试一次并获胜，平均分可能很高，但结论不可靠。访问次数多表示搜索器反复验证后仍然愿意投入计算资源。

---

### `_select_root_move`

这个函数决定下一次假想比赛测试哪个走法。

第一阶段：

```text
存在从未访问过的走法
→ 从未访问走法中随机选择
```

意义：

> 每个合法走法至少先看一次，不能完全忽略未知走法。

第二阶段：

```text
所有走法都访问过
→ 使用 UCT 平衡“目前表现好”和“还没有充分尝试”
```

公式：

```text
UCT = 平均价值 + 探索奖励
```

不需要在答辩时展开复杂推导，只需要解释：

```text
平均价值高的走法值得继续尝试
访问次数少的走法也会得到额外探索机会
```

---

### `_rollout`

这个函数进行后续假想比赛。

```text
如果游戏结束，返回 +1 或 -1
如果没有合法走法，跳过当前回合
否则选择一个 rollout 走法并执行
达到最大步数仍未结束时，使用启发式局面评价
```

---

### `_select_rollout_move`

rollout 不是完全随机，而是加入简单常识：

```text
能一步到达目标角，立即走获胜棋
能吃对方棋子，优先吃子
否则随机走
```

这使模拟结果比纯随机更有参考价值。

---

### `_evaluate_non_terminal`

如果假想比赛太长，程序不会无限模拟，而是根据：

```text
双方最接近目标角的距离
双方剩余棋子数量
```

估计局面好坏。

公式含义：

```text
我方更接近目标角，分数更高
我方剩余棋子更多，分数更高
```

---

### `_build_policy`

这个函数将访问次数转换为概率。

例子：

```text
动作 A：20 次
动作 B：10 次
动作 C：70 次
```

转换后：

```text
A：0.20
B：0.10
C：0.70
```

这个策略分布会成为神经网络训练的 `policy` 答案。

---

### 这一部分的总结话术

> 基础 MCTS 的主要价值不是成为最终参赛 AI，而是解决冷启动问题。它只依赖规则引擎，通过大量假想比赛生成比随机走棋更有质量的决策，并把访问次数分布保存为后续神经网络的策略标签。

---

# 第三部分：自我对弈数据

## 五、`einstein_chess/training/self_play.py`

### 这个文件做什么

这个文件让两个智能体自己和自己下棋，并将每一步保存为训练样本。

每个样本最重要的三个内容：

```text
state   当前局面
policy  搜索器对各个动作的推荐程度
value   当前行动方最终是赢还是输
```

---

### `SelfPlaySample`

保存单步样本：

```text
state       15 × 5 × 5 状态
policy      18 维策略分布
player      当前行动方
dice_roll   当前骰子
action_id   实际执行动作
turn_index  当前回合
game_id     所属对局
```

其中训练直接使用的是：

```text
state、policy、value
```

其他字段主要用于分析和报告。

---

### `SelfPlayDataset`

保存整个数据集，并通过 `save_npz()` 写入 `.npz` 文件。

---

### `generate_self_play_dataset`

这个函数负责生成多局数据。

```text
循环生成每一局自我对弈
→ 收集所有单步样本
→ 根据最终胜者填写每个样本的 value
→ 构建完整数据集
→ 可选保存为 NPZ
```

---

### `generate_self_play_game`

这是单局自我对弈的核心。

可以逐段讲：

```text
1. 创建棋局，可以选择默认开局或随机开局。
2. 为红蓝双方创建相同类型的智能体。
3. 每回合从规则引擎获取合法走法。
4. 智能体调用 choose_move() 决策。
5. 保存当前 state 和智能体 last_policy。
6. 执行走法。
7. 重复直到游戏结束或达到最大回合数。
```

重要代码含义：

```python
state=encode_state(snapshot)
```

将局面转换为神经网络输入。

```python
policy=policy_dict_to_vector(agent.last_policy)
```

将搜索器的访问次数分布转换为 18 维策略答案。

```python
action_id=move_to_action_id(move)
```

记录最终实际执行的动作编号。

---

### `_values_for_samples`

整局比赛结束后才知道 `value`。

```text
如果样本当时的行动方最终获胜，value = +1
如果样本当时的行动方最终失败，value = -1
如果没有分出胜负，value = 0
```

需要强调：

> `value` 始终从样本当时行动方的角度记录，不是固定从红方角度记录。

---

### 版本数据来源

| 模型 | 新数据由谁生成 | 布局 |
|---|---|---|
| V1 | 基础 MCTS | 默认布局 |
| V2 | V1 + Root Neural MCTS | 默认布局 |
| V3 | V2 + Root Neural MCTS | 随机布局 |
| V4 Full | V3 + Full Neural MCTS | 随机布局 |

---

# 第四部分：神经网络的题目和答案

## 六、`einstein_chess/training/state_encoder.py`

### 这个文件做什么

将人类可以理解的棋盘局面转换为神经网络可以读取的 `15 × 5 × 5` 数字张量。

可以把它解释为：

```text
15 张重叠的 5 × 5 透明棋盘
```

---

### 15 个通道

| 通道 | 含义 |
|---:|---|
| 0-5 | 红方 1-6 号棋子位置 |
| 6-11 | 蓝方 1-6 号棋子位置 |
| 12 | 当前行动方，红方填 `1`，蓝方填 `-1` |
| 13 | 当前骰子点数，保存为 `dice / 6` |
| 14 | 当前合法候选棋子的来源位置 |

前 12 个棋子通道只使用 `0` 和 `1`。

例子：

```text
红方 3 号棋子在 (2, 1)
→ 通道 2 的 (2, 1) 位置为 1
```

为什么每枚棋子需要独立通道：

> 棋子编号影响骰子选子规则。如果只记录“这里有红棋”，网络无法知道它是 1 号还是 6 号。

---

### `encode_state`

执行过程：

```text
创建全零的 15 × 5 × 5 数组
→ 遍历棋盘，将红蓝棋子写入对应通道
→ 填写当前玩家通道
→ 填写骰子通道
→ 标记所有合法走法的起始棋子位置
```

---

## 七、`einstein_chess/training/action_codec.py`

### 这个文件做什么

将不同棋子的不同移动方向统一编码为固定长度的 18 维动作空间。

```text
6 枚棋子 × 每枚棋子 3 个方向 = 18 个动作
```

---

### 动作编号

公式：

```text
action_id = (piece_number - 1) × 3 + direction_index
```

红方方向顺序：

```text
0：向右
1：向下
2：向右下
```

蓝方方向顺序：

```text
0：向左
1：向上
2：向左上
```

例如红方 3 号棋子向右下：

```text
(3 - 1) × 3 + 2 = 8
```

所以动作编号为 `8`。

---

### `move_to_action_id`

将 `Move` 转换为动作编号。

---

### `action_id_to_move`

将动作编号转换回当前局面中的 `Move`。

需要检查：

```text
动作编号是否合法
棋子是否还在棋盘上
动作是否属于当前合法走法
```

---

### `legal_action_mask`

神经网络始终输出 18 个动作分数，但当前局面通常只有少数动作合法。

这个函数生成：

```text
合法动作位置为 1
非法动作位置为 0
```

后续只在合法动作中计算概率。

---

### `policy_dict_to_vector`

将搜索器的走法概率字典转换为 18 维数组。

例子：

```text
3 号向右：0.2
3 号向下：0.1
3 号向右下：0.7
```

转换后：

```text
[0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

---

# 第五部分：策略价值网络

## 八、`einstein_chess/training/model.py`

### 这个文件做什么

这个文件定义神经网络：

```text
输入：15 × 5 × 5 局面
输出 1：18 个动作分数
输出 2：1 个局面价值
```

---

### `PolicyValueNet`

最终使用的普通 CNN 主干包含三层卷积：

```text
15 × 5 × 5
→ 64 × 5 × 5
→ 64 × 5 × 5
→ 64 × 5 × 5
```

卷积层负责从棋盘附近位置中提取特征，例如：

```text
接近目标角的棋子
附近吃子机会
当前候选棋子的周围关系
对方可能的威胁
```

这些特征不是人工写死的，而是由训练数据学习得到。

---

### 策略头 `policy_head`

```text
64 × 5 × 5
→ 展平为 1600 个数字
→ 128 个隐藏特征
→ 18 个动作分数
```

回答的问题：

```text
当前局面下，哪些动作更值得走？
```

---

### 价值头 `value_head`

```text
64 × 5 × 5
→ 展平为 1600 个数字
→ 128 个隐藏特征
→ 1 个价值分数
→ Tanh 限制到 [-1, 1]
```

回答的问题：

```text
当前行动方从这个局面开始，最后更可能赢还是输？
```

价值分数不是严格胜率：

```text
+1 附近：当前行动方非常有利
0 附近：局面不明确
-1 附近：当前行动方非常不利
```

---

### `policy_value_loss`

训练时有两个错误需要同时减少：

```text
policy_loss  网络动作预测与搜索器策略答案之间的误差
value_loss   网络价值预测与最终胜负答案之间的误差
```

总损失：

```text
total_loss = policy_loss + value_loss_weight × value_loss
```

答辩时不需要讲反向传播细节，只需要说明：

> 网络同时学习“怎么走”和“局面好不好”。策略答案来自搜索访问次数分布，价值答案来自整局自我对弈的最终胜负。

---

### `ResidualBlock`

项目曾尝试更深的 ResNet V4。

结论：

```text
网络结构更复杂，但实战对 V3 胜率只有 32.5%
因此没有采用
```

可以强调：

> 项目最终选择模型不是只看结构复杂度或训练指标，而是看实际对战结果。

---

# 第六部分：V2 的 Root Neural MCTS

## 九、`einstein_chess/agents/neural_mcts.py`

### 这个文件做什么

这个文件实现 Root Neural MCTS。

它使用 V1 或后续网络帮助当前根节点搜索，但不会建立完整多层搜索树。

---

### 与基础 MCTS 的区别

基础 MCTS：

```text
执行一个走法
→ 继续随机或启发式 rollout
→ 得到胜负结果
```

Root Neural MCTS：

```text
网络策略头提供根节点走法先验
→ 执行一个走法
→ 网络价值头直接评价下一局面
```

---

### `NeuralMCTSSearchStats`

与基础 MCTS 相比，多了：

```text
prior
```

`prior` 是网络策略头对该走法的初始推荐程度。

---

### `choose_move`

执行过程：

```text
检查一步获胜
→ 使用网络策略头获得每个合法走法的 prior
→ 创建每个走法的搜索统计
→ 重复 simulations 次根节点搜索
→ 每次使用 PUCT 选择走法
→ 执行走法
→ 如果未结束，使用网络价值头评价下一局面
→ 更新走法统计
→ 返回访问次数最多的走法
```

---

### `_root_priors`

这个函数：

```text
将当前局面编码为 15 × 5 × 5
→ 使用网络策略头输出 18 个动作分数
→ 使用合法动作 mask 去掉非法动作
→ 只保留当前合法走法的概率
```

---

### `_evaluate_leaf`

这个函数使用价值头评价执行一步后的局面。

注意价值视角转换：

```text
网络 value 从下一局面当前行动方角度输出
如果下一局面轮到对手，需要取负号转换为根节点玩家视角
```

---

### V2 数据如何产生

```text
V1 网络 + Root Neural MCTS
→ 生成 500 局新数据
→ 与基础 MCTS 旧数据合并
→ 训练 V2
```

需要强调：

> 搜索阶段 V1 是冻结的，只进行预测，不会立即反向训练。收集完整数据集后，才训练新的 V2 网络。

---

# 第七部分：最终 Full Neural MCTS

## 十、`einstein_chess/agents/full_neural_mcts.py`

### 这个文件做什么

这是最终搜索器，支持：

```text
多层 PUCT 搜索
每层使用网络策略先验
叶子节点价值评估
价值回传
骰子 chance node
预测缓存
```

---

### `StateKey`

```python
StateKey = tuple[tuple[tuple[int, ...], ...], str, int]
```

一个局面键由以下内容组成：

```text
棋盘
当前玩家
当前骰子
```

同一个棋盘如果当前玩家或骰子不同，就是不同局面。

---

### `FullNeuralMCTSNode`

每个搜索树节点保存：

```text
当前玩家
该局面每个合法走法的统计
```

每个走法统计包括：

```text
prior       网络初始推荐
visits      搜索访问次数
value_sum   回传价值总和
mean_value  平均价值
```

---

### `choose_move`

执行过程：

```text
清空本次决策的搜索树和预测缓存
→ 检查一步获胜
→ 从根局面创建或展开根节点
→ 重复 simulations 次调用 _search()
→ 将根节点访问次数转换为策略分布
→ 返回访问次数最多的走法
```

---

### `_search`

这是最终搜索器最重要的递归函数。

按分支讲解：

```text
如果游戏已经结束
→ 返回终局价值
```

```text
如果当前没有合法走法
→ 跳过当前回合
→ 递归搜索下一层
→ 因为换了玩家，返回值取负号
```

```text
如果当前局面是第一次遇到
→ 使用网络策略头展开节点
→ 使用网络价值头评价局面
→ 本次模拟停止并回传价值
```

```text
如果达到最大搜索深度
→ 使用网络价值头评价局面
```

```text
否则
→ 使用 PUCT 选择走法
→ 搜索执行走法后的子局面
→ 更新该走法访问次数和价值总和
```

---

### 为什么第一次遇到新局面就停止

如果每次都无限向下搜索，计算量会迅速爆炸。

Full Neural MCTS 的做法是：

```text
第一次遇到新局面：用网络评价并建立节点
后续模拟再次到达该局面：从这个节点继续向下搜索
```

所以搜索树会随着模拟次数逐渐变深。

---

### `_evaluate_child_after_move`

这个函数处理：

```text
执行走法
→ 检查是否获胜
→ 处理下一回合骰子随机性
→ 递归搜索下一层
```

---

### 骰子 chance node

爱恩斯坦棋每走一步后都要重新掷骰子，因此搜索树中存在随机事件。

项目支持两种模式：

```text
sample     每次模拟随机采样一个骰子，速度快
enumerate  枚举 1 到 6 并取平均，更严格但更慢
```

最终使用：

```text
chance_mode = sample
```

---

### `_select_move`

使用 PUCT 选择下一步要搜索的走法。

它综合考虑：

```text
走法目前的平均价值
网络策略头给出的先验概率
该走法是否还没有被充分搜索
```

网络负责提供经验，搜索负责验证和修正经验。

---

### `_get_or_expand_node`

第一次遇到局面时：

```text
调用网络策略头获得合法走法先验
→ 为每个合法走法创建统计对象
→ 保存到搜索树
```

---

### `_predict`

这个函数同时调用网络策略头和价值头。

执行过程：

```text
将局面编码为 15 × 5 × 5
→ 生成合法动作 mask
→ 网络输出 policy logits 和 value
→ 只在合法动作中做 softmax
→ 缓存预测结果
```

预测缓存的意义：

> 搜索过程中同一个局面可能被多次访问。缓存可以避免重复进行相同的神经网络推理。

---

### V4 Full 数据如何产生

```text
V3 网络 + Full Neural MCTS + 随机开局
→ 生成 1000 局新数据
→ 与 V3 训练数据合并
→ 训练最终 V4 Full 网络
```

最终训练集：

```text
3000 局
53583 个样本
```

---

# 第八部分：比赛运行

## 十一、`einstein_chess/match.py`

### 这个文件做什么

这个文件负责完整比赛流程，而不是 AI 算法本身。

它支持：

```text
15 分钟包干计时
布局时间计费
非法布局判负
非法走法判负
超时判负
智能体异常判负
最大回合限制
逐步比赛日志
```

---

### `MatchRunner.play`

执行过程：

```text
准备双方布局
→ 检查剩余时间
→ 获取当前合法走法
→ 调用当前智能体 choose_move()
→ 扣除思考时间
→ 检查超时、异常和非法走法
→ 执行走法
→ 记录比赛步骤
→ 判断胜负
```

可以强调：

> AI 算法只负责返回走法，比赛运行器负责裁判工作。这样可以清晰分离“会不会下棋”和“比赛是否合法”。

---

## 十二、`scripts/competition_client.py`

### 这个文件做什么

这个文件是最终参赛入口，用于连接线上比赛服务器。

默认模型：

```text
artifacts/checkpoints/policy_value_v4_full_best_loss.pt
```

默认智能体：

```text
full-neural-mcts
```

---

### `main_async`

执行过程：

```text
根据命令行参数创建智能体
→ 连接服务器
→ 读取比赛局数和角色
→ 提交开局布局
→ 循环进行每一局比赛
→ 统计胜负
```

---

### `_play_one_game`

执行过程：

```text
接收服务器消息
→ 如果比赛结束，返回胜者
→ 如果不是自己的回合，继续等待
→ 如果没有合法走法，发送 pass
→ 将服务器局面转换为 GameSnapshot
→ 调用 agent.choose_move()
→ 检查返回走法是否合法
→ 将走法发送给服务器
```

可以这样总结：

> 最终线上客户端没有重新实现 AI，只是把服务器局面转换为项目内部的 `GameSnapshot`，然后调用与本地 GUI、批量评估相同的智能体。

---

# 第九部分：版本演进答辩话术

## 十三、V1 到 V4 Full

| 版本 | 教材来源 | 解决的问题 |
|---|---|---|
| V1 | 基础 MCTS 自我对弈 | 从零开始获得第一版策略价值网络 |
| V2 | 基础 MCTS 数据 + V1 引导的 Root Neural MCTS 数据 | 使用网络帮助搜索，生成更高质量教材 |
| V3 | V2 引导的 Root Neural MCTS 随机开局数据 | 提升不同初始布局下的泛化能力 |
| V4 Full | V3 引导的 Full Neural MCTS 随机开局数据 | 学习多层搜索产生的更高质量策略 |

实验结果：

```text
V3 vs V2：V3 胜率 65%
Full Neural MCTS vs Root Neural MCTS：Full 胜率 58.5%
V4 Full vs V3：V4 Full 胜率 55%
ResNet V4 vs V3：ResNet V4 胜率 32.5%，因此放弃
```

---

# 第十部分：老师可能会问的问题

## 十四、常见问题与回答

### 1. 为什么不直接使用随机 AI 生成数据？

> 随机 AI 的走法质量太低，神经网络会学习到大量无意义行为。基础 MCTS 通过假想比赛提供更有质量的策略分布，适合作为冷启动老师。

### 2. 为什么策略答案不是最终执行动作，而是访问次数分布？

> 访问次数分布包含更多信息。它不仅告诉网络哪个动作最好，也告诉网络其他动作相对有多值得考虑。

### 3. 为什么价值答案来自最终胜负，而不是 MCTS 评价？

> 最终胜负是更直接的监督信号。MCTS 和网络价值只是中间估计，可能存在偏差。

### 4. 神经网络是否一定不会超过老师 MCTS？

> 纯模仿确实受老师质量限制，但网络可以从大量相似局面中总结规律，价值头还学习最终胜负。更重要的是，项目将网络重新放回搜索器中，形成“网络提供经验、搜索修正网络”的迭代过程。

### 5. 为什么 V3 要随机开局？

> 比赛允许不同的棋子摆放。如果只训练固定布局，模型可能记住开局位置而不是学习通用棋理。

### 6. Root Neural MCTS 和 Full Neural MCTS 的区别是什么？

> Root Neural MCTS 只比较当前一步，并用网络评价走一步后的局面。Full Neural MCTS 会在多层局面中反复使用策略先验、叶子价值和价值回传，能够发现短期好看但后续危险的走法。

### 7. 为什么需要骰子 chance node？

> 爱恩斯坦棋的未来不仅由玩家走法决定，还由骰子决定。Full Neural MCTS 必须将骰子随机性纳入搜索。

### 8. 为什么最终不用更复杂的 ResNet？

> ResNet V4 的实际对战胜率明显低于 V3。项目以实战结果为准，没有因为网络更复杂就采用它。

### 9. 为什么最终仍然使用搜索，而不是只使用 V4 Full 网络？

> 网络提供快速经验判断，但可能犯错。Full Neural MCTS 可以向未来搜索多层，验证并修正网络建议。

### 10. 如何保证训练和比赛规则一致？

> 所有模块都使用同一个 `EinsteinGame` 规则引擎。自我对弈、搜索、GUI、评估和线上客户端都通过相同的合法走法和走棋接口运行。

---

# 第十一部分：建议的现场演示

## 十五、演示命令

### 人人对战

```powershell
python main.py --red human --blue human
```

### 人类对最终 AI

```powershell
python main.py --red human --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

### 展示最终线上客户端

```powershell
python scripts/competition_client.py --host 服务器IP --port 端口 --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cpu --agent full-neural-mcts --full-neural-mcts-simulations 40 --full-neural-mcts-depth 8 --chance-mode sample
```

---

# 第十二部分：5 分钟精简讲解稿

## 十六、时间不足时这样讲

> 首先看 `engine.py`。这里统一定义了棋盘、棋子、骰子候选规则、合法走法、走棋和胜负判断。所有 AI 和比赛程序都只调用这套规则，因此训练和实战规则一致。
>
> 然后看 `mcts.py`。项目初期没有数据和模型，所以使用基础 MCTS 冷启动。它对当前每个合法走法进行多次假想比赛，记录访问次数和收益，最后选择访问次数最多的走法。访问次数分布还可以作为神经网络的策略答案。
>
> `self_play.py` 让两个智能体自己和自己下棋。每一步保存当前局面 `state`、搜索器策略分布 `policy`，整局结束后根据胜负填写 `value`。
>
> `state_encoder.py` 将局面编码为 `15 × 5 × 5`，包括双方 12 枚编号棋子、当前玩家、骰子和合法候选棋子。`action_codec.py` 将 6 枚棋子的 3 个方向编码为 18 个动作。
>
> `model.py` 定义策略价值网络。策略头输出 18 个动作分数，价值头输出当前行动方的局面价值。V1 学习基础 MCTS 数据。
>
> `neural_mcts.py` 使用 V1 的策略头提供根节点先验，用价值头评价走一步后的局面，并生成 V2 数据。V3 再加入随机开局数据，提升不同布局下的泛化能力。
>
> 最后看 `full_neural_mcts.py`。它会在多层搜索树中反复使用网络策略先验，在新叶子节点使用价值头评价，并处理骰子 chance node。项目使用 V3 加 Full Neural MCTS 生成最终 V4 Full 数据。最终参赛方案是 V4 Full 网络结合 Full Neural MCTS，而不是纯网络。

---

# 第十三部分：讲解时不要说错的地方

## 十七、重要注意事项

1. 不要说基础 `mcts.py` 是完整多层 MCTS。它只维护根节点走法统计。
2. 不要说 V2 生成数据时会立即反向训练 V1。V1 在搜索阶段被冻结，只负责推理。
3. 不要说策略头直接输出概率。模型输出的是 logits，之后才在合法动作中转换为概率。
4. 不要说价值头输出严格胜率。它输出的是 `[-1, 1]` 范围内的局面价值。
5. 不要说 V4 是 ResNet。最终 V4 Full 使用的是普通 CNN，ResNet V4 是失败实验。
6. 不要说 Full Neural MCTS 每一层都立即停止。已展开节点会继续向下搜索，第一次遇到的新节点才使用网络评价并回传。
7. 不要说训练 `value` 来自网络评价。训练 `value` 来自整局自我对弈的最终胜负。
8. 不要说最终方案是纯神经网络。最终方案是 `V4 Full + Full Neural MCTS`。

