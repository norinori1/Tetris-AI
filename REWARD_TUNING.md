"""
報酬設計のトラブルシューティング手順

学習がまだ不十分な場合の追加対策
"""

# ❌ 学習が不十分な場合の対策

## 1. さらに報酬を増やす
# tetris_env.py で以下を試す:

SURVIVAL_REWARD = 0.2  # 0.1 → 0.2に増加
# または
HOLE_PENALTY = 0.1   # 0.2 → 0.1にさらに削減
HEIGHT_PENALTY = 0.0005
BUMPINESS_PENALTY = 0.0005

## 2. 学習率を上げる
# dqn_agent.py で以下を変更:

self.learning_rate = 0.0005  # 0.0001 → 0.0005に増加

## 3. バッチサイズを小さくする
# dqn_agent.py で以下を変更:

self.batch_size = 32  # 64 → 32に削減（より頻繁に学習）

## 4. ε-greedy探索を調整
# dqn_agent.py で以下を変更:

self.epsilon_decay = 0.9995  # 0.995 → 0.9995に（探索をより長く）

## 推奨される段階的な改善:

# ステップ1: 現在の修正で様子を見る（1000エピソード）
# → Avg Lines が 0.5 以上になったか確認

# ステップ2: 改善がない場合
SURVIVAL_REWARD = 0.2

# ステップ3: さらに改善がない場合
self.learning_rate = 0.0005

# ステップ4: 最終手段
self.batch_size = 32
