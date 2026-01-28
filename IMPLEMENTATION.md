# Tetris AI 実装ドキュメント

## 目次
1. [概要](#概要)
2. [フレームワーク選択](#フレームワーク選択)
3. [DQN（Deep Q-Network）の説明](#dqndeep-q-networkの説明)
4. [報酬設計](#報酬設計)
5. [ニューラルネットワーク構造](#ニューラルネットワーク構造)
6. [実装コードの説明](#実装コードの説明)
7. [使い方](#使い方)
8. [学習のヒント](#学習のヒント)

---

## 概要

このプロジェクトは、深層強化学習を用いてテトリスをプレイするAIを実装したものです。DQN（Deep Q-Network）アルゴリズムを使用し、PyTorchで実装されています。

**セキュリティ**: PyTorch 2.6.0以上を使用し、`torch.load`で`weights_only=True`を指定することで、セキュリティ脆弱性を回避しています。

### プロジェクト構成
```
Tetris-AI/
├── Tetris-AI/
│   ├── Tetris_AI.py      # 元のテトリスゲーム（人間用）
│   ├── tetris_env.py     # Gym環境ラッパー
│   ├── dqn_agent.py      # DQNエージェント実装
│   ├── train.py          # 学習スクリプト
│   └── play.py           # 学習済みモデルで遊ぶスクリプト
├── requirements.txt       # 必要なライブラリ
└── IMPLEMENTATION.md      # このドキュメント
```

---

## フレームワーク選択

### なぜDQNを選んだか？

テトリスAIの実装には**DQN（Deep Q-Network）**を選択しました。PPOとの比較：

| 特徴 | DQN | PPO |
|------|-----|-----|
| **適用分野** | 離散的行動空間 | 連続・離散両方 |
| **学習の安定性** | 中程度（経験再生で改善） | 高い |
| **実装の複雑さ** | 比較的シンプル | やや複雑 |
| **サンプル効率** | 良い（経験再生あり） | 中程度 |
| **テトリスへの適合性** | ◎ 最適 | ○ 可能だがオーバースペック |

#### DQNが適している理由：

1. **離散的な行動空間**: テトリスの行動（左、右、回転、落下）は完全に離散的
2. **実装のシンプルさ**: 学習・理解に適している
3. **実績**: Atariゲームなどの類似タスクで成功実績が多い
4. **経験再生**: 過去の経験を再利用できるため、サンプル効率が良い

---

## DQN（Deep Q-Network）の説明

### 強化学習の基本概念

**強化学習**は、エージェント（AI）が環境と相互作用しながら、報酬を最大化する行動を学習する手法です。

```
エージェント → 行動(Action) → 環境
    ↑                            ↓
    └── 状態(State) + 報酬(Reward) ←┘
```

- **状態（State）**: 環境の現在の様子（テトリスのグリッド、現在のピース）
- **行動（Action）**: エージェントが取れる行動（左移動、右移動、回転、落下）
- **報酬（Reward）**: 行動の良し悪しを示すスコア
- **方策（Policy）**: 状態から行動を選択する戦略

### Q学習とは

**Q学習**は、各状態-行動ペアの価値（Q値）を学習する手法です。

```
Q(状態, 行動) = 期待される累積報酬
```

最適なQ値を学習できれば、各状態で最も高いQ値を持つ行動を選べば良いことになります。

### DQNの革新

従来のQ学習では、全ての状態-行動ペアをテーブルで管理していましたが、テトリスのように状態空間が膨大な場合は不可能です。

**DQN**は、**ニューラルネットワーク**を使ってQ値を近似することで、この問題を解決します。

```
ニューラルネットワーク(状態) → [Q(状態, 行動1), Q(状態, 行動2), ...]
```

### DQNの主要技術

#### 1. 優先度付き経験再生（Prioritized Experience Replay）

過去の経験（状態、行動、報酬、次の状態）をバッファに保存し、**TD誤差に基づいて優先度付きでサンプリング**して学習します。

**従来の経験再生との違い**:
- 通常の経験再生: ランダムにサンプリング
- 優先度付き経験再生: 重要な経験（TD誤差が大きい、正の報酬など）を優先的にサンプリング

```python
# 優先度付きバッファの設定
buffer = PrioritizedReplayBuffer(
    capacity=50000,      # 大きなバッファサイズ
    alpha=0.6,           # 優先度の強さ
    beta_start=0.4       # 重要度サンプリングの補正
)

# 正の報酬（ライン消去）には高い初期優先度を設定
if reward > 0:
    priority = max_priority * 2  # ライン消去経験を優先
```

**メリット**:
- データの相関を減らす（連続したデータで学習すると偏りが生じる）
- **稀な成功体験（ライン消去）を頻繁に学習できる**
- 学習効率が大幅に向上

#### 2. Double DQN

Q値の過大評価を防ぐため、行動選択と評価を分離します：

```python
# 行動選択: ポリシーネットワーク
next_actions = policy_network(next_states).argmax()

# 行動評価: ターゲットネットワーク
next_q_values = target_network(next_states).gather(next_actions)

# 目標Q値
target_q = reward + gamma * next_q_values
```

**メリット**:
- Q値の過大評価を防ぐ
- より安定した学習

#### 3. ターゲットネットワーク

学習を安定化させるため、2つのネットワークを使用します：

- **ポリシーネットワーク**: 現在のQ値を計算（頻繁に更新）
- **ターゲットネットワーク**: 目標Q値を計算（定期的にコピー、500ステップごと）

```python
# 現在のQ値
current_q = policy_network(state)[action]

# 目標Q値（ターゲットネットワーク使用）
target_q = reward + gamma * max(target_network(next_state))

# 損失を計算して学習
loss = (current_q - target_q)^2
```

**メリット**:
- 学習ターゲットが安定する
- 発散を防ぐ

#### 4. ε-greedy探索（改善版）

学習中は探索と活用のバランスを取ります：

- **探索**: ランダムな行動を取る（確率ε）
- **活用**: 最も良いと思われる行動を取る（確率1-ε）

```python
if random() < epsilon:
    action = random_action()  # 探索
else:
    action = argmax(Q(state))  # 活用
```

**改善点**:
- `epsilon_decay = 0.9995`（以前は0.995）: よりゆっくり減衰
- `epsilon_min = 0.05`（以前は0.01）: 最小値を高く設定して継続的な探索を促進

### DQNの学習フロー

1. 環境をリセットして初期状態を取得
2. ε-greedyで行動を選択
3. 行動を実行して報酬と次の状態を取得
4. 経験を**優先度付き**経験再生バッファに保存
5. バッファから**優先度に基づいて**ミニバッチをサンプリング
6. **Double DQN方式で**Q値を計算してネットワークを更新
7. **TD誤差に基づいて経験の優先度を更新**
8. 定期的にターゲットネットワークを更新
9. 2-8を繰り返す

---

## 報酬設計

報酬設計は強化学習で最も重要な要素の一つです。良い報酬設計がなければ、エージェントは望ましい行動を学習できません。

### 報酬の構成要素

#### 1. ライン消去報酬（主要報酬）- 大幅に増加

```python
LINE_CLEAR_REWARDS = {
    1: 10,   # Single line（以前は1）
    2: 30,   # Double（以前は3）
    3: 60,   # Triple（以前は5）
    4: 100,  # Tetris（以前は8）
}
```

**変更理由**: ライン消去報酬を大幅に増加させることで、エージェントがライン消去の重要性を学習しやすくなる。ペナルティとの相対的なバランスを改善。

#### 2. T-Spinボーナス

```python
TSPIN_REWARDS = {
    1: 15,   # T-Spin Single
    2: 25,   # T-Spin Double
    3: 40,   # T-Spin Triple
}
```

**理由**: 高度なテクニックを学習させる

#### 3. Back-to-Backボーナス

```python
if is_difficult and back_to_back:
    reward *= 1.5
```

**理由**: 連続した高度なプレイを推奨

#### 4. 穴ペナルティ - 軽減

```python
holes = count_holes(grid)
reward -= (holes - prev_holes) * 0.3  # 以前は0.5
```

**変更理由**: ペナルティを軽減することで、ライン消去報酬との相対的なバランスを改善

#### 5. 高さペナルティ - 軽減

```python
height = max_column_height(grid)
reward -= (height - prev_height) * 0.005  # 以前は0.01
```

**変更理由**: 高さペナルティを軽減して、エージェントがより積極的にプレイできるようにする

#### 6. 凹凸ペナルティ - 軽減

```python
bumpiness = sum(abs(heights[i] - heights[i+1]))
reward -= (bumpiness - prev_bumpiness) * 0.005  # 以前は0.01
```

**変更理由**: 平らな盤面を維持しつつ、ライン消去を優先するバランス

#### 7. 生存報酬

```python
reward += 0.01
```

**理由**: 長く生き残ることを推奨

#### 8. ゲームオーバーペナルティ - 軽減

```python
if game_over:
    reward -= 5  # 以前は10
```

**変更理由**: ゲームオーバーペナルティを軽減することで、エージェントがリスクを取ってライン消去を狙うことを促進

### 報酬設計の考え方

良い報酬設計のポイント：

1. **スパース報酬を避ける**: ライン消去だけでなく、中間的な良い行動にも報酬を与える
2. **報酬の大きさを調整**: **ライン消去報酬がペナルティの累積を上回るように設計**
3. **短期的報酬と長期的報酬**: 即座の報酬と将来の報酬のバランス
4. **負の報酬**: 避けるべき行動にはペナルティを与える（ただし軽めに）

---

## ニューラルネットワーク構造

### 入力層

入力は**209次元のベクトル**：

```
入力 = グリッド状態(200) + ピース種類(7) + ピース位置(2)
```

- **グリッド状態**: 20×10 = 200個の値（0 or 1）
- **ピース種類**: 7種類をone-hot encoding（I, O, T, S, Z, J, L）
- **ピース位置**: x座標とy座標（正規化済み）

### 隠れ層

```python
self.network = nn.Sequential(
    nn.Linear(209, 256),    # 第1隠れ層
    nn.ReLU(),
    nn.Linear(256, 256),    # 第2隠れ層
    nn.ReLU(),
    nn.Linear(256, 128),    # 第3隠れ層
    nn.ReLU(),
    nn.Linear(128, 5)       # 出力層
)
```

- **活性化関数**: ReLU（Rectified Linear Unit）
  - 勾配消失問題を軽減
  - 計算が高速
  - 深層学習で標準的

### 出力層

**5つのQ値**を出力：

```
出力 = [Q(左移動), Q(右移動), Q(回転), Q(ハードドロップ), Q(ソフトドロップ)]
```

各値は、その行動を取ったときの期待累積報酬を表します。

### なぜこの構造？

- **深さ**: 3層の隠れ層で適度な表現力
- **幅**: 256→256→128と徐々に減少
- **パラメータ数**: 過学習を防ぐため適度な大きさ
- **正則化**: ドロップアウトなしでシンプルに

---

## 実装コードの説明

### 1. tetris_env.py - 環境ラッパー

Gym互換の環境を提供します。

```python
class TetrisEnv(gym.Env):
    def reset(self):
        # ゲームを初期化
        self.grid = [[0]*10 for _ in range(20)]
        self.current_piece = Tetromino()
        return observation
    
    def step(self, action):
        # 行動を実行
        # 報酬を計算
        # 次の状態を返す
        return next_state, reward, done, info
```

**主要メソッド**:

- `reset()`: 新しいエピソード開始時にゲームをリセット
- `step(action)`: 行動を実行して結果を返す
- `_get_observation()`: 現在の状態を観測として返す
- `_calculate_board_stats()`: 盤面統計（高さ、穴、凹凸）を計算
- `render()`: ゲームを可視化（オプション）

### 2. dqn_agent.py - DQNエージェント

DQNアルゴリズムの実装です。

```python
class DQN(nn.Module):
    """ニューラルネットワーク"""
    def __init__(self, input_size, output_size):
        # ネットワーク構造を定義
    
    def forward(self, x):
        # 順伝播
        return self.network(x)

class ReplayBuffer:
    """経験再生バッファ"""
    def push(self, state, action, reward, next_state, done):
        # 経験を保存
    
    def sample(self, batch_size):
        # ランダムにサンプリング
        return batch

class DQNAgent:
    """DQNエージェント"""
    def select_action(self, state, training=True):
        # ε-greedyで行動選択
        if random() < epsilon:
            return random_action()
        else:
            return argmax(Q(state))
    
    def train_step(self):
        # 1ステップの学習
        # ミニバッチをサンプリング
        # Q値を計算
        # 損失を計算してネットワークを更新
```

**主要メソッド**:

- `select_action()`: ε-greedy方策で行動を選択
- `store_transition()`: 経験をバッファに保存
- `train_step()`: ネットワークを1ステップ学習
- `save()/load()`: モデルの保存/読み込み

### 3. train.py - 学習スクリプト

エージェントを学習させるメインスクリプトです。

```python
def train_dqn(num_episodes=1000):
    env = TetrisEnv()
    agent = DQNAgent(state_size, action_size)
    
    for episode in range(num_episodes):
        state = env.reset()
        
        for step in range(max_steps):
            # 行動選択
            action = agent.select_action(state)
            
            # 環境で実行
            next_state, reward, done, info = env.step(action)
            
            # 経験を保存
            agent.store_transition(state, action, reward, next_state, done)
            
            # 学習
            agent.train_step()
            
            state = next_state
            if done:
                break
        
        # 定期的にモデルを保存
        if episode % 100 == 0:
            agent.save(f'model_{episode}.pth')
```

**機能**:

- 指定したエピソード数だけ学習
- 定期的にモデルを保存
- 学習曲線をプロット
- 統計情報を表示

### 4. play.py - プレイスクリプト

学習済みモデルでテトリスをプレイします。

```python
def play_tetris(model_path, num_games=5, render=True):
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0.0  # 探索なし
    
    for game in range(num_games):
        state = env.reset()
        
        while not done:
            env.render()
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
        
        print(f"Lines cleared: {info['lines_cleared']}")
```

**機能**:

- 学習済みモデルでプレイ
- 可視化機能
- 評価モード（統計収集）

---

## 使い方

### 1. 環境構築

```bash
# リポジトリをクローン（既にある場合はスキップ）
git clone https://github.com/norinori1/Tetris-AI.git
cd Tetris-AI

# 依存ライブラリをインストール
pip install -r requirements.txt
```

### 2. 学習

```bash
# デフォルト設定で学習（1000エピソード）
python train.py
```

学習中、以下の情報が表示されます：
- エピソード数
- 平均報酬
- 平均ライン消去数
- 平均損失
- ε（探索率）

学習済みモデルは`models/`ディレクトリに保存されます。

### 3. プレイ

```bash
# 学習済みモデルでプレイ（可視化あり）
python play.py --model models/tetris_dqn_final.pth --games 5

# 可視化なしで評価
python play.py --model models/tetris_dqn_final.pth --evaluate --eval-episodes 100
```

**オプション**:
- `--model`: モデルファイルのパス
- `--games`: プレイするゲーム数
- `--no-render`: 可視化を無効化
- `--delay`: 行動間の遅延（秒）
- `--evaluate`: 評価モード
- `--eval-episodes`: 評価エピソード数

### 4. カスタマイズ

学習のハイパーパラメータを変更したい場合は、`dqn_agent.py`を編集：

```python
# DQNAgent クラス内（改善後のハイパーパラメータ）
self.gamma = 0.99           # 割引率
self.epsilon = 1.0          # 初期探索率
self.epsilon_min = 0.05     # 最小探索率（以前は0.01）
self.epsilon_decay = 0.9995 # 探索率の減衰（以前は0.995）
self.learning_rate = 0.0005 # 学習率（以前は0.0001）
self.batch_size = 128       # ミニバッチサイズ（以前は64）
self.target_update_freq = 500  # ターゲット更新頻度（以前は1000）
```

---

## 学習のヒント

### 学習がうまくいかない場合

#### 1. ライン消去を学習しない

**原因と対策**:
- **報酬バランスが悪い** → ライン消去報酬を大幅に増加（10, 30, 60, 100）
- **ペナルティが強すぎる** → 穴・高さ・凹凸ペナルティを軽減
- **稀な成功体験が学習されない** → 優先度付き経験再生を使用
- **探索が足りない** → epsilon_decayを0.9995に、epsilon_minを0.05に

#### 2. 報酬が増えない

**原因と対策**:
- εが高すぎる → epsilon_decayを調整
- 学習率が不適切 → learning_rateを変更（0.001～0.00001）
- 報酬設計が悪い → 報酬の重みを調整

#### 3. 学習が不安定

**原因と対策**:
- バッチサイズが小さい → 64→128に増やす
- ターゲットネットワークの更新頻度 → 500-1000程度に調整
- 勾配爆発 → 勾配クリッピングを調整

#### 4. 過学習

**原因と対策**:
- ネットワークが大きすぎる → 隠れ層を削減
- 経験再生バッファが小さい → capacity=50000以上を推奨

### 学習時間の目安

- **CPU**: 1000エピソード ≈ 2-4時間
- **GPU**: 1000エピソード ≈ 30-60分

**注意**: ライン消去を学習するには最低5000エピソード以上の学習を推奨

### より良い性能を得るには

1. **長く学習する**: 10000-50000エピソード
2. **ハイパーパラメータチューニング**: グリッドサーチやベイズ最適化
3. **ネットワーク構造の改良**: 隠れ層を512ユニットに増加（実装済み）
4. **報酬設計の改良**: ライン消去報酬を大幅に増加（実装済み）
5. **アルゴリズムの改良**: Double DQN、優先度付き経験再生（実装済み）

### 実装済みのDQN改良版

このプロジェクトでは以下の改良が実装されています：

- ✅ **Double DQN**: Q値の過大評価を防ぐ
- ✅ **Prioritized Experience Replay**: 重要な経験（ライン消去）を優先的に学習
- ✅ **次のピース情報**: 観測空間に次のピースの情報を追加
- ✅ **報酬設計の最適化**: ライン消去報酬の大幅増加、ペナルティの軽減

---

## 技術的詳細

### ハイパーパラメータの意味

- **γ (gamma) = 0.99**: 割引率
  - 将来の報酬をどれだけ重視するか
  - 0に近い→近視眼的、1に近い→遠視眼的

- **ε (epsilon)**: 探索率
  - ランダム行動を取る確率
  - 1.0→0.05に減衰（よりゆっくり、より高い最小値）

- **学習率 = 0.0005**: ネットワークの更新ステップサイズ
  - 大きい→学習が速いが不安定
  - 小さい→学習が遅いが安定

- **バッチサイズ = 128**: 一度に学習するサンプル数
  - 大きい→安定だが遅い
  - 小さい→速いが不安定

### 損失関数

平均二乗誤差（MSE）を使用：

```
Loss = (Q(s,a) - (r + γ * max Q(s',a')))^2
```

- Q(s,a): 現在のQ値
- r: 報酬
- γ: 割引率
- max Q(s',a'): 次状態の最大Q値

### 最適化手法

**Adam optimizer**を使用：
- 学習率を自動調整
- モーメンタムとRMSpropの利点を組み合わせ
- 深層学習で標準的

---

## まとめ

このTetris AIは、DQNアルゴリズムを使用した強化学習の実装例です。

**学んだこと**:
1. 強化学習の基本概念（状態、行動、報酬）
2. Q学習とDQNの理論
3. 経験再生とターゲットネットワークの重要性
4. 報酬設計の重要性
5. ε-greedy探索
6. PyTorchによる実装

**次のステップ**:
- より高度なDQN変種の実装
- 他のゲームへの応用
- 他の強化学習アルゴリズム（PPO、A3Cなど）の学習

Happy Learning! 🎮🤖
