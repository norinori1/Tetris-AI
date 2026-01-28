# Tetris AI - DQN Implementation

Deep Q-Network (DQN) implementation for playing Tetris automatically using reinforcement learning.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train the Agent

```bash
python train.py
```

This will train the DQN agent for 1000 episodes and save models in the `models/` directory.

### Play with Trained Agent

```bash
python play.py --model models/tetris_dqn_final.pth --games 5
```

### Evaluate Agent

```bash
python play.py --model models/tetris_dqn_final.pth --evaluate --eval-episodes 100
```

## Files

- `tetris_env.py`: Gym-compatible Tetris environment
- `dqn_agent.py`: DQN agent implementation with experience replay
- `train.py`: Training script
- `play.py`: Play/evaluation script
- `IMPLEMENTATION.md`: Detailed documentation in Japanese (機械学習の詳細説明)

## Features

- Deep Q-Network (DQN) with experience replay
- Target network for stable learning
- Comprehensive reward shaping (line clears, holes, height, bumpiness)
- T-Spin and Back-to-Back detection
- Training visualization and statistics

## Documentation

For detailed explanation of the implementation, reward design, and machine learning concepts, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

詳細な実装説明、報酬設計、機械学習の概念については[IMPLEMENTATION.md](IMPLEMENTATION.md)をご覧ください。
