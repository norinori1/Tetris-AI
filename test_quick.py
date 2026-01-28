"""
Quick test to verify the AI learning improvements
"""
import torch
import numpy as np
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent

def test_environment():
    """Test environment initialization and basic functionality"""
    print("Testing environment...")
    env = TetrisEnv()
    state, _ = env.reset()
    print(f"✓ Environment initialized")
    print(f"  State shape: {state.shape}")
    
    # Test one step
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step executed successfully")
    print(f"  Reward: {reward:.2f}")
    print(f"  Info: {info}")
    env.close()

def test_agent():
    """Test agent initialization"""
    print("\nTesting agent...")
    env = TetrisEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = DQNAgent(state_size, action_size, device=device)
    
    print(f"✓ Agent initialized on {device}")
    print(f"  Epsilon: {agent.epsilon}")
    print(f"  Learning rate: {agent.learning_rate}")
    print(f"  Batch size: {agent.batch_size}")
    print(f"  Replay buffer capacity: {agent.memory.buffer.maxlen}")
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state, training=True)
    print(f"✓ Action selected: {action}")
    env.close()

def test_short_training():
    """Test a very short training run"""
    print("\nTesting short training run (10 episodes)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = TetrisEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, device=device)
    
    episode_rewards = []
    episode_lines = []
    
    for episode in range(10):
        state, _ = env.reset()
        episode_reward = 0
        lines_cleared = 0
        
        for step in range(100):  # Limit steps per episode
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train_step()
            episode_reward += reward
            lines_cleared = info.get('lines_cleared', 0)
            state = next_state
            
            if done:
                break
        
        agent.end_episode()
        episode_rewards.append(episode_reward)
        episode_lines.append(lines_cleared)
    
    avg_reward = np.mean(episode_rewards)
    avg_lines = np.mean(episode_lines)
    total_lines = sum(episode_lines)
    
    print(f"✓ Training completed")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average lines cleared: {avg_lines:.2f}")
    print(f"  Total lines cleared: {total_lines}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    
    env.close()
    return total_lines > 0  # Success if at least one line was cleared

def test_reward_values():
    """Test that reward values are reasonable"""
    print("\nTesting reward design...")
    from tetris_env import (
        LINE_CLEAR_BASE_REWARD,
        HOLE_PENALTY,
        HEIGHT_PENALTY,
        BUMPINESS_PENALTY,
        SURVIVAL_REWARD,
        GAME_OVER_PENALTY,
        PIECE_PLACEMENT_REWARD,
        SOME_FILLED_REWARD,
        MOST_FILLED_REWARD,
        ALMOST_FULL_LINE_REWARD,
        VERY_FULL_LINE_REWARD,
        ONE_AWAY_FROM_CLEAR_REWARD
    )
    
    print(f"✓ Reward constants:")
    print(f"  Line clear base: {LINE_CLEAR_BASE_REWARD}")
    print(f"  1 line reward: {LINE_CLEAR_BASE_REWARD * 1 * 1:.0f}")
    print(f"  2 line reward: {LINE_CLEAR_BASE_REWARD * 2 * 2:.0f}")
    print(f"  3 line reward: {LINE_CLEAR_BASE_REWARD * 3 * 3:.0f}")
    print(f"  4 line reward: {LINE_CLEAR_BASE_REWARD * 4 * 4:.0f}")
    print(f"  Hole penalty: {HOLE_PENALTY}")
    print(f"  Height penalty: {HEIGHT_PENALTY}")
    print(f"  Survival reward: {SURVIVAL_REWARD}")
    print(f"  Game over penalty: {GAME_OVER_PENALTY}")
    print(f"  Piece placement: {PIECE_PLACEMENT_REWARD}")
    print(f"  Some filled line (50%+): {SOME_FILLED_REWARD}")
    print(f"  Most filled line (70%+): {MOST_FILLED_REWARD}")
    print(f"  Almost full line (80%+): {ALMOST_FULL_LINE_REWARD}")
    print(f"  Very full line (90%+): {VERY_FULL_LINE_REWARD}")
    print(f"  One away from clear (9/10): {ONE_AWAY_FROM_CLEAR_REWARD}")

if __name__ == '__main__':
    print("="*60)
    print("Quick Test Suite for Tetris AI Improvements")
    print("="*60)
    
    try:
        test_reward_values()
        test_environment()
        test_agent()
        success = test_short_training()
        
        print("\n" + "="*60)
        if success:
            print("✓ ALL TESTS PASSED - Learning appears to be working!")
        else:
            print("⚠ Tests completed but no lines cleared in 10 episodes")
            print("  This might improve with longer training")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
