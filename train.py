"""
Training script for DQN Tetris Agent
Trains the agent using Deep Q-Learning with Prioritized Experience Replay
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import os
import time


def train_dqn(num_episodes=1000, max_steps=1000, save_freq=100, model_dir='models'):
    """
    Train DQN agent on Tetris environment
    
    Args:
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        save_freq: Save model every N episodes
        model_dir: Directory to save models
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    best_avg_lines = 0
    
    print("Starting training...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Improved settings: Prioritized Experience Replay, Double DQN, Larger Network")
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
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_lines = np.mean(episode_lines[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(losses[-10:]) if losses else 0
            
            # Calculate ETA
            elapsed = time.time() - start_time
            episodes_done = episode + 1
            time_per_episode = elapsed / episodes_done
            remaining_episodes = num_episodes - episodes_done
            eta_seconds = remaining_episodes * time_per_episode
            eta_minutes = eta_seconds / 60
            
            print(f"Episode {episode + 1}/{num_episodes} [ETA: {eta_minutes:.1f}分]")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Lines: {avg_lines:.2f}")
            print(f"  Avg Steps: {avg_length:.1f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print()
        
        # Save model at intervals and when performance improves
        current_avg_lines = np.mean(episode_lines[-100:]) if len(episode_lines) >= 100 else np.mean(episode_lines)
        
        if (episode + 1) % save_freq == 0:
            model_path = os.path.join(model_dir, f'tetris_dqn_episode_{episode + 1}.pth')
            agent.save(model_path)
            print(f"✓ Model saved: {model_path}")
            print()
        
        # Save best model when average lines improve
        if current_avg_lines > best_avg_lines and len(episode_lines) >= 50:
            best_avg_lines = current_avg_lines
            best_model_path = os.path.join(model_dir, 'tetris_dqn_best.pth')
            agent.save(best_model_path)
            print(f"★ New best model saved! Avg Lines: {best_avg_lines:.2f}")
            print()
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'tetris_dqn_final.pth')
    agent.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lines, losses, model_dir)
    
    env.close()
    return agent, episode_rewards, episode_lines


def plot_training_curves(episode_rewards, episode_lines, losses, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Moving average of rewards
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Avg Reward')
        axes[0, 1].grid(True)
    
    # Lines cleared
    axes[1, 0].plot(episode_lines)
    axes[1, 0].set_title('Lines Cleared per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Lines')
    axes[1, 0].grid(True)
    
    # Training loss
    if losses:
        axes[1, 1].plot(losses)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")
    plt.close()


if __name__ == '__main__':
    # Training configuration
    NUM_EPISODES = 1000
    MAX_STEPS = 1000
    SAVE_FREQ = 100
    
    # Train agent
    agent, rewards, lines = train_dqn(
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        save_freq=SAVE_FREQ
    )
    
    print("\nTraining completed!")
    num_final_episodes = min(100, len(rewards))
    print(f"Average reward (last {num_final_episodes} episodes): {np.mean(rewards[-num_final_episodes:]):.2f}")
    print(f"Average lines (last {num_final_episodes} episodes): {np.mean(lines[-num_final_episodes:]):.2f}")
