"""
Play script for trained DQN Tetris Agent
Loads trained model and plays Tetris
"""
import torch
import numpy as np
import os
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import argparse
import time


def play_tetris(model_path, num_games=5, render=True, delay=0.1):
    """
    Play Tetris using trained DQN agent
    
    Args:
        model_path: Path to saved model
        num_games: Number of games to play
        render: Whether to render the game
        delay: Delay between actions (seconds)
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train a model first using: python train.py")
        return
    
    # Setup device with proper fallback
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[OK] Using GPU (CUDA): {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[!] GPU not available - Using CPU instead")
    
    # Create environment
    render_mode = 'human' if render else None
    env = TetrisEnv(render_mode=render_mode)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent and load model
    agent = DQNAgent(state_size, action_size, device=device)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during evaluation
    
    print(f"Model loaded from {model_path}")
    print(f"Playing {num_games} games...\n")
    
    # Play games
    total_scores = []
    total_lines = []
    
    for game in range(num_games):
        state, _ = env.reset()
        done = False
        steps = 0
        game_reward = 0
        
        print(f"Game {game + 1}/{num_games}")
        
        while not done:
            if render:
                env.render()
                time.sleep(delay)
            
            # Select action (greedy)
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            game_reward += reward
            state = next_state
            steps += 1
        
        # Print game results
        lines_cleared = info.get('lines_cleared', 0)
        total_scores.append(game_reward)
        total_lines.append(lines_cleared)
        
        print(f"  Steps: {steps}")
        print(f"  Lines cleared: {lines_cleared}")
        print(f"  Total reward: {game_reward:.2f}")
        print()
    
    # Print summary
    print("=" * 50)
    print("Summary:")
    print(f"Average lines cleared: {np.mean(total_lines):.2f} ± {np.std(total_lines):.2f}")
    print(f"Average reward: {np.mean(total_scores):.2f} ± {np.std(total_scores):.2f}")
    print(f"Max lines cleared: {max(total_lines)}")
    print(f"Min lines cleared: {min(total_lines)}")
    
    env.close()


def evaluate_agent(model_path, num_episodes=100):
    """
    Evaluate agent performance over multiple episodes
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train a model first using: python train.py")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = TetrisEnv()
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, device=device)
    agent.load(model_path)
    agent.epsilon = 0.0
    
    scores = []
    lines = []
    
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        scores.append(episode_reward)
        lines.append(info.get('lines_cleared', 0))
        
        if (episode + 1) % 10 == 0:
            print(f"  {episode + 1}/{num_episodes} completed")
    
    env.close()
    
    metrics = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_lines': np.mean(lines),
        'std_lines': np.std(lines),
        'max_lines': max(lines),
        'min_lines': min(lines)
    }
    
    print("\nEvaluation Results:")
    print(f"  Mean score: {metrics['mean_score']:.2f} ± {metrics['std_score']:.2f}")
    print(f"  Mean lines: {metrics['mean_lines']:.2f} ± {metrics['std_lines']:.2f}")
    print(f"  Max lines: {metrics['max_lines']}")
    print(f"  Min lines: {metrics['min_lines']}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Tetris with trained DQN agent')
    parser.add_argument('--model', type=str, default='models/tetris_dqn_final.pth',
                        help='Path to trained model')
    parser.add_argument('--games', type=int, default=5,
                        help='Number of games to play')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Delay between actions in seconds')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation mode (100 episodes, no render)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_agent(args.model, args.eval_episodes)
    else:
        play_tetris(args.model, args.games, not args.no_render, args.delay)
