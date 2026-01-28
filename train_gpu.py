"""
GPU対応トレーニングスクリプト
RTX 4060に最適化された設定
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import os
import time


def train_dqn_gpu(num_episodes=2000, max_steps=1000, save_freq=100, model_dir='models'):
    """
    Train DQN agent on Tetris environment with GPU optimization
    
    Args:
        num_episodes: Number of episodes to train (推奨: 2000-5000)
        max_steps: Maximum steps per episode
        save_freq: Save model every N episodes
        model_dir: Directory to save models
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup device (自動でGPUを検出)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("GPU対応 Tetris AI トレーニング")
    print("=" * 60)
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print("=" * 60)
    print()
    
    # Create environment and agent
    env = TetrisEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Training statistics
    episode_rewards = []
    episode_lines = []
    episode_lengths = []
    losses = []
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Training for {num_episodes} episodes...")
    print()
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lines.append(info.get('lines_cleared', 0))
        episode_lengths.append(step + 1)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress with estimated time
        if (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_episode = elapsed_time / (episode + 1)
            remaining_episodes = num_episodes - (episode + 1)
            eta_seconds = avg_time_per_episode * remaining_episodes
            eta_minutes = eta_seconds / 60
            
            avg_reward = np.mean(episode_rewards[-10:])
            avg_lines = np.mean(episode_lines[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(losses[-10:]) if losses else 0
            
            print(f"Episode {episode + 1}/{num_episodes} [ETA: {eta_minutes:.1f}分]")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Lines: {avg_lines:.2f}")
            print(f"  Avg Steps: {avg_length:.1f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print()
        
        # Save model
        if (episode + 1) % save_freq == 0:
            model_path = os.path.join(model_dir, f'tetris_dqn_episode_{episode + 1}.pth')
            agent.save(model_path)
            print(f"✓ Model saved: {model_path}")
            print()
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'tetris_dqn_final.pth')
    agent.save(final_model_path)
    
    total_time = time.time() - start_time
    print("=" * 60)
    print("Training Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final model saved to {final_model_path}")
    print("=" * 60)
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lines, losses, model_dir)
    
    env.close()
    
    return agent


def plot_training_curves(rewards, lines, losses, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Moving average function
    def moving_average(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot rewards
    axes[0].plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= 50:
        axes[0].plot(moving_average(rewards), label='Moving Avg (50)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot lines cleared
    axes[1].plot(lines, alpha=0.3, label='Raw')
    if len(lines) >= 50:
        axes[1].plot(moving_average(lines), label='Moving Avg (50)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Lines Cleared')
    axes[1].set_title('Lines Cleared per Episode')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot loss
    if losses:
        axes[2].plot(losses, alpha=0.3, label='Raw')
        if len(losses) >= 50:
            axes[2].plot(moving_average(losses), label='Moving Avg (50)')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Training curves saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Tetris AI with GPU')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes to train (default: 2000)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max steps per episode (default: 1000)')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Save model every N episodes (default: 100)')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    
    args = parser.parse_args()
    
    print("Starting GPU-optimized training...")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print()
    
    agent = train_dqn_gpu(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_freq=args.save_freq,
        model_dir=args.model_dir
    )
    
    print("\nTraining complete! To test the trained model:")
    print(f"  python play.py --model {args.model_dir}/tetris_dqn_final.pth --games 5")
