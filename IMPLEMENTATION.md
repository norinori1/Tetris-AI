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

#### 1. 経験再生（Experience Replay）

過去の経験（状態、行動、報酬、次の状態）をバッファに保存し、ランダムにサンプリングして学習します。

**メリット**:
- データの相関を減らす（連続したデータで学習すると偏りが生じる）
- 同じ経験を複数回使える（サンプル効率が良い）

```python
# 経験をバッファに保存
buffer.push(state, action, reward, next_state, done)

# ランダムにミニバッチをサンプリング
batch = buffer.sample(batch_size=64)
```

#### 2. ターゲットネットワーク

学習を安定化させるため、2つのネットワークを使用します：

- **ポリシーネットワーク**: 現在のQ値を計算（頻繁に更新）
- **ターゲットネットワーク**: 目標Q値を計算（定期的にコピー）

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

#### 3. ε-greedy探索

学習中は探索と活用のバランスを取ります：

- **探索**: ランダムな行動を取る（確率ε）
- **活用**: 最も良いと思われる行動を取る（確率1-ε）

```python
if random() < epsilon:
    action = random_action()  # 探索
else:
    action = argmax(Q(state))  # 活用
```

εは時間とともに減少させ、初期は探索、後期は活用に重点を置きます。

### DQNの学習フロー

1. 環境をリセットして初期状態を取得
2. ε-greedyで行動を選択
3. 行動を実行して報酬と次の状態を取得
4. 経験を経験再生バッファに保存
5. バッファからランダムにミニバッチをサンプリング
6. Q値を計算してネットワークを更新
7. 定期的にターゲットネットワークを更新
8. 2-7を繰り返す

---

## 報酬設計

報酬設計は強化学習で最も重要な要素の一つです。良い報酬設計がなければ、エージェントは望ましい行動を学習できません。

### 報酬の構成要素

#### 1. ライン消去報酬（主要報酬）

```python
if num_lines == 1:
    reward = 1
elif num_lines == 2:
    reward = 3
elif num_lines == 3:
    reward = 5
elif num_lines == 4:  # テトリス
    reward = 8
```

**理由**: 複数ライン同時消去を推奨（効率的なプレイ）

#### 2. T-Spinボーナス

```python
if is_tspin:
    if num_lines == 1:
        reward = 8
    elif num_lines == 2:
        reward = 12
    elif num_lines == 3:
        reward = 16
```

**理由**: 高度なテクニックを学習させる

#### 3. Back-to-Backボーナス

```python
if is_difficult and back_to_back:
    reward *= 1.5
```

**理由**: 連続した高度なプレイを推奨

#### 4. 穴ペナルティ

```python
holes = count_holes(grid)
reward -= (holes - prev_holes) * 0.5
```

**理由**: 穴を作ると後で消しにくくなる

#### 5. 高さペナルティ

```python
height = max_column_height(grid)
reward -= (height - prev_height) * 0.01
```

**理由**: 盤面を低く保つことが重要

#### 6. 凹凸ペナルティ

```python
bumpiness = sum(abs(heights[i] - heights[i+1]))
reward -= (bumpiness - prev_bumpiness) * 0.01
```

**理由**: 平らな盤面が望ましい

#### 7. 生存報酬

```python
reward += 0.01
```

**理由**: 長く生き残ることを推奨

#### 8. ゲームオーバーペナルティ

```python
if game_over:
    reward -= 10
```

**理由**: ゲームオーバーを避ける

### 報酬設計の考え方

良い報酬設計のポイント：

1. **スパース報酬を避ける**: ライン消去だけでなく、中間的な良い行動にも報酬を与える
2. **報酬の大きさを調整**: メイン報酬とペナルティのバランスが重要
3. **短期的報酬と長期的報酬**: 即座の報酬と将来の報酬のバランス
4. **負の報酬**: 避けるべき行動にはペナルティを与える

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
# DQNAgent クラス内
self.gamma = 0.99           # 割引率
self.epsilon = 1.0          # 初期探索率
self.epsilon_min = 0.01     # 最小探索率
self.epsilon_decay = 0.995  # 探索率の減衰
self.learning_rate = 0.0001 # 学習率
self.batch_size = 64        # ミニバッチサイズ
```

---

## 学習のヒント

### 学習がうまくいかない場合

#### 1. 報酬が増えない

**原因と対策**:
- εが高すぎる → epsilon_decayを調整
- 学習率が不適切 → learning_rateを変更（0.001～0.00001）
- 報酬設計が悪い → 報酬の重みを調整

#### 2. 学習が不安定

**原因と対策**:
- バッチサイズが小さい → 64→128に増やす
- ターゲットネットワークの更新頻度 → 1000→2000に増やす
- 勾配爆発 → 勾配クリッピングを調整

#### 3. 過学習

**原因と対策**:
- ネットワークが大きすぎる → 隠れ層を削減
- 経験再生バッファが小さい → capacity=10000→50000

### 学習時間の目安

- **CPU**: 1000エピソード ≈ 2-4時間
- **GPU**: 1000エピソード ≈ 30-60分

### より良い性能を得るには

1. **長く学習する**: 5000-10000エピソード
2. **ハイパーパラメータチューニング**: グリッドサーチやベイズ最適化
3. **ネットワーク構造の改良**: 畳み込み層の追加など
4. **報酬設計の改良**: より詳細な評価基準
5. **アルゴリズムの改良**: Double DQN、Dueling DQN等

### DQNの改良版

さらに性能を上げたい場合は、以下の手法を検討：

- **Double DQN**: Q値の過大評価を防ぐ
- **Dueling DQN**: 価値関数とアドバンテージ関数を分離
- **Prioritized Experience Replay**: 重要な経験を優先的に学習
- **Noisy Networks**: パラメトリック探索
- **Rainbow DQN**: 上記すべてを組み合わせ

---

## 技術的詳細

### ハイパーパラメータの意味

- **γ (gamma) = 0.99**: 割引率
  - 将来の報酬をどれだけ重視するか
  - 0に近い→近視眼的、1に近い→遠視眼的

- **ε (epsilon)**: 探索率
  - ランダム行動を取る確率
  - 1.0→0.01に減衰

- **学習率 = 0.0001**: ネットワークの更新ステップサイズ
  - 大きい→学習が速いが不安定
  - 小さい→学習が遅いが安定

- **バッチサイズ = 64**: 一度に学習するサンプル数
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
