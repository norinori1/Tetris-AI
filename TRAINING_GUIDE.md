# 学習パラメータのカスタマイズ例 / Training Configuration Examples

## 基本的な使い方 / Basic Usage

### 短時間で試す（テスト用）/ Quick Test
```bash
# 100エピソードで学習（約5-10分）
python train.py  # デフォルトは1000エピソード

# コード内で変更する場合
# train.py の NUM_EPISODES = 100 に変更
```

### 通常の学習 / Normal Training
```bash
# デフォルト設定で学習
python train.py  # 1000エピソード、約1-2時間
```

### 長時間学習（より良い性能）/ Extended Training
```python
# train.py を編集:
NUM_EPISODES = 5000  # 5000エピソード、約5-10時間
# または
NUM_EPISODES = 10000  # 10000エピソード、約10-20時間
```

## ハイパーパラメータのチューニング / Hyperparameter Tuning

### dqn_agent.py で調整可能なパラメータ

```python
class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        # 学習速度を変更 / Change learning speed
        self.learning_rate = 0.0001  # デフォルト
        # 速く学習: 0.001
        # ゆっくり学習: 0.00001
        
        # 割引率を変更 / Change discount factor
        self.gamma = 0.99  # デフォルト（将来の報酬を重視）
        # 短期的: 0.9
        # 長期的: 0.999
        
        # 探索率の設定 / Exploration settings
        self.epsilon = 1.0          # 初期探索率
        self.epsilon_min = 0.01     # 最小探索率
        self.epsilon_decay = 0.995  # 減衰率
        # より探索的: epsilon_decay = 0.999
        # より活用的: epsilon_decay = 0.99
        
        # バッチサイズを変更 / Change batch size
        self.batch_size = 64  # デフォルト
        # 大きく: 128 (より安定、遅い)
        # 小さく: 32 (速い、不安定)
        
        # ターゲットネットワーク更新頻度 / Target network update
        self.target_update_freq = 1000  # デフォルト
        # より頻繁: 500
        # より安定: 2000
```

### tetris_env.py で報酬を調整

```python
# Constants for reward design
T_PIECE_SHAPE_ID = 2

# これらの値を変更して報酬バランスを調整
HOLE_PENALTY = 0.5      # 穴のペナルティ（大きい→穴を避ける）
HEIGHT_PENALTY = 0.01   # 高さのペナルティ
BUMPINESS_PENALTY = 0.01  # 凹凸のペナルティ
SURVIVAL_REWARD = 0.01   # 生存報酬
GAME_OVER_PENALTY = 10   # ゲームオーバーペナルティ

# 調整例:
# より攻撃的: HOLE_PENALTY = 1.0, HEIGHT_PENALTY = 0.05
# より保守的: HOLE_PENALTY = 0.2, HEIGHT_PENALTY = 0.001
```

## 実験の追跡 / Tracking Experiments

### 複数の設定で実験する例

```bash
# 実験1: デフォルト設定
python train.py

# モデルを保存したら名前を変更
mkdir -p experiments
mv models/tetris_dqn_final.pth experiments/exp1_default.pth
mv models/training_curves.png experiments/exp1_curves.png

# 実験2: 高い学習率
# dqn_agent.py で learning_rate = 0.001 に変更
python train.py
mv models/tetris_dqn_final.pth experiments/exp2_high_lr.pth

# 実験3: 大きなバッチサイズ
# dqn_agent.py で batch_size = 128 に変更
python train.py
mv models/tetris_dqn_final.pth experiments/exp3_large_batch.pth
```

### 実験結果の比較

```bash
# それぞれのモデルを評価
python play.py --model experiments/exp1_default.pth --evaluate --eval-episodes 100
python play.py --model experiments/exp2_high_lr.pth --evaluate --eval-episodes 100
python play.py --model experiments/exp3_large_batch.pth --evaluate --eval-episodes 100
```

## トラブルシューティング / Troubleshooting

### 学習が進まない / Training Not Improving

```python
# dqn_agent.py で以下を試す:
self.learning_rate = 0.001      # 学習率を上げる
self.epsilon_decay = 0.999      # 探索をより長く続ける
self.batch_size = 32            # バッチサイズを小さく
```

### 学習が不安定 / Training Unstable

```python
# dqn_agent.py で以下を試す:
self.learning_rate = 0.00001    # 学習率を下げる
self.batch_size = 128           # バッチサイズを大きく
self.target_update_freq = 2000  # ターゲット更新を遅く
```

### メモリ不足 / Out of Memory

```python
# dqn_agent.py の ReplayBuffer:
def __init__(self, capacity=10000):  # デフォルト
# capacity を小さくする:
def __init__(self, capacity=5000):

# train.py:
MAX_STEPS = 1000  # デフォルト
# 短くする:
MAX_STEPS = 500
```

## 推奨設定 / Recommended Settings

### 初心者向け（クイックテスト）/ Beginner (Quick Test)
```python
# train.py
NUM_EPISODES = 100
MAX_STEPS = 500

# dqn_agent.py
learning_rate = 0.001
batch_size = 32
```

### 中級者向け（バランス型）/ Intermediate (Balanced)
```python
# train.py
NUM_EPISODES = 2000
MAX_STEPS = 1000

# dqn_agent.py
learning_rate = 0.0001  # デフォルト
batch_size = 64         # デフォルト
```

### 上級者向け（高性能）/ Advanced (High Performance)
```python
# train.py
NUM_EPISODES = 10000
MAX_STEPS = 2000

# dqn_agent.py
learning_rate = 0.00005
batch_size = 128
target_update_freq = 2000
capacity = 50000  # ReplayBuffer
```
