"""
Medium-length training test (100 episodes) to verify learning progress
"""
import torch
import numpy as np
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent

def test_medium_training():
    """Test a medium training run (100 episodes)"""
    print("="*60)
    print("Medium Training Test (100 episodes, ~500 steps each)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    env = TetrisEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, device=device)
    
    episode_rewards = []
    episode_lines = []
    
    for episode in range(100):
        state, _ = env.reset()
        episode_reward = 0
        lines_cleared = 0
        
        for step in range(500):  # Max 500 steps per episode
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
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_lines = np.mean(episode_lines[-10:])
            max_lines = max(episode_lines[-10:])
            print(f"Episode {episode + 1}/100:")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Lines (last 10): {avg_lines:.2f}")
            print(f"  Max Lines (last 10): {max_lines}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    # Final statistics
    print("\n" + "="*60)
    print("Final Statistics:")
    print("="*60)
    total_lines = sum(episode_lines)
    avg_reward = np.mean(episode_rewards)
    avg_lines = np.mean(episode_lines)
    max_lines = max(episode_lines)
    
    # Last 20 episodes performance
    avg_reward_late = np.mean(episode_rewards[-20:])
    avg_lines_late = np.mean(episode_lines[-20:])
    total_lines_late = sum(episode_lines[-20:])
    
    print(f"Overall:")
    print(f"  Total lines cleared: {total_lines}")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average lines per episode: {avg_lines:.2f}")
    print(f"  Max lines in one episode: {max_lines}")
    
    print(f"\nLast 20 episodes:")
    print(f"  Total lines cleared: {total_lines_late}")
    print(f"  Average reward: {avg_reward_late:.2f}")
    print(f"  Average lines per episode: {avg_lines_late:.2f}")
    
    print(f"\nAgent state:")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Replay buffer size: {len(agent.memory)}")
    
    env.close()
    
    # Success criteria: At least 1 line cleared in total
    success = total_lines > 0
    
    print("\n" + "="*60)
    if success:
        print(f"✓ SUCCESS: Agent cleared {total_lines} line(s) during training!")
        if total_lines_late > total_lines * 0.5:
            print("✓ GREAT: Learning is improving (most lines in later episodes)")
    else:
        print("✗ ISSUE: No lines cleared in 100 episodes")
        print("  Recommendation: Check reward design or increase training time")
    print("="*60)
    
    return success

if __name__ == '__main__':
    try:
        test_medium_training()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
