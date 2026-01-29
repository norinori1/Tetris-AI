# Tetris AI - DQN Implementation

Deep Q-Network (DQN) implementation for playing Tetris automatically using reinforcement learning.

[日本語ドキュメント (Japanese Documentation)](IMPLEMENTATION.md) | [学習ガイド (Training Guide)](TRAINING_GUIDE.md)

## Overview

This project implements a Tetris-playing AI using Deep Q-Network (DQN), a reinforcement learning algorithm. The AI learns to play Tetris through trial and error, gradually improving its strategy.

**Key Features:**
- 🎮 DQN-based reinforcement learning agent
- 🔄 Experience replay for efficient learning
- 🎯 Target network for stable training
- 📊 Comprehensive reward shaping
- 📈 Training visualization
- 🎪 Support for T-Spin and Back-to-Back detection

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train the Agent

```bash
# Quick test (100 episodes, ~5-10 minutes)
python demo_training.py

# Full training (1000 episodes, ~1-2 hours)
python train.py
```

The trained model will be saved in the `models/` directory.

### Play with Trained Agent

```bash
# Play 5 games with visualization
python play.py --model models/tetris_dqn_final.pth --games 5

# Evaluate performance over 100 games
python play.py --model models/tetris_dqn_final.pth --evaluate --eval-episodes 100
```

### Test the Implementation

```bash
# Run comprehensive tests
python test_implementation.py
```

## Files

- **tetris_env.py**: Gym-compatible Tetris environment
- **dqn_agent.py**: DQN agent implementation with experience replay
- **train.py**: Training script with visualization
- **play.py**: Play/evaluation script
- **demo_training.py**: Quick training demo
- **test_implementation.py**: Test suite
- **IMPLEMENTATION.md**: Detailed documentation in Japanese (機械学習の詳細説明)
- **TRAINING_GUIDE.md**: Training configuration guide (学習パラメータガイド)

## Documentation

For detailed explanation of the implementation, reward design, and machine learning concepts:
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Complete guide in Japanese
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training configuration examples

詳細な実装説明、報酬設計、機械学習の概念については:
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - 完全なガイド（日本語）
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 学習パラメータの設定例

## Customization

### Training Parameters

Edit `train.py`:
```python
NUM_EPISODES = 1000  # Number of training episodes
MAX_STEPS = 1000     # Max steps per episode
SAVE_FREQ = 100      # Save model every N episodes
```

### Hyperparameters

Edit `dqn_agent.py`:
```python
self.gamma = 0.99           # Discount factor
self.learning_rate = 0.001  # Learning rate
self.batch_size = 64        # Mini-batch size
self.epsilon_decay = 0.995  # Exploration decay rate
```

### Reward Design (v15)

Edit `tetris_env.py`:
```python
# v15 uses exponential scaling for line clears
LINE_CLEAR_BASE_REWARD = 1000.0  # Base reward
# 1 line: 1000, 2 lines: 2000, 3 lines: 4000, 4 lines: 8000

WELL_REWARD = 2.0           # Well-building reward
HEIGHT_DANGER_PENALTY = 5.0 # Danger zone penalty
COMBO_BONUS = 1.2           # Combo multiplier
```

See [V15_REWARD_REDESIGN.md](V15_REWARD_REDESIGN.md) for detailed explanation of the reward system improvements.

## Performance

With default settings (1000 episodes):
- Average lines cleared: varies based on training
- Training time: ~1-2 hours on CPU, ~30-60 minutes on GPU

For better performance, train for 5000-10000 episodes.

## Algorithm

This implementation uses **DQN (Deep Q-Network)** which combines:
1. **Q-Learning**: Learn action-value function
2. **Deep Neural Network**: Approximate Q-values
3. **Experience Replay**: Store and reuse past experiences
4. **Target Network**: Stabilize training

## Requirements

- Python 3.8+
- PyTorch 2.6.0+ (updated for security patches)
- Gymnasium 0.29.1+
- Pygame 2.5.2+
- NumPy 1.24.3+
- Matplotlib 3.8.0+

## License

This is an educational project for learning reinforcement learning.

## Contributing

Feel free to open issues or submit pull requests for improvements!
