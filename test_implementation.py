"""
Test script to verify the Tetris environment and agent work correctly
"""
import sys
import torch
import numpy as np
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent

def test_environment():
    """Test basic environment functionality"""
    print("Testing Tetris Environment...")
    
    env = TetrisEnv()
    
    # Test reset
    state, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  State shape: {state.shape}")
    print(f"  Action space: {env.action_space.n}")
    
    # Test step
    for action in range(env.action_space.n):
        env.reset()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Action {action} executed successfully (reward: {reward:.2f})")
    
    # Test a short episode
    env.reset()
    total_reward = 0
    for _ in range(50):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"✓ Short episode completed")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Lines cleared: {info.get('lines_cleared', 0)}")
    
    env.close()
    print("✓ Environment test passed!\n")
    return True


def test_agent():
    """Test DQN agent functionality"""
    print("Testing DQN Agent...")
    
    # Setup device with proper fallback
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[OK] Using GPU (CUDA): {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[!] GPU not available - Using CPU instead")
    print()
    
    env = TetrisEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, device=device)
    print(f"✓ Agent created successfully")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state, training=True)
    print(f"✓ Action selection works (action: {action})")
    
    # Test storing transition
    next_state, reward, terminated, truncated, info = env.step(action)
    agent.store_transition(state, action, reward, next_state, terminated)
    print(f"✓ Transition stored in replay buffer")
    
    # Fill replay buffer
    for _ in range(100):
        state, _ = env.reset()
        for _ in range(10):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            state = next_state
            if terminated or truncated:
                break
    
    print(f"✓ Replay buffer filled (size: {len(agent.memory)})")
    
    # Test training step
    loss = agent.train_step()
    if loss is not None:
        print(f"✓ Training step successful (loss: {loss:.4f})")
    else:
        print("✓ Training step executed (buffer filling)")
    
    env.close()
    print("✓ Agent test passed!\n")
    return True


def test_integration():
    """Test integration of environment and agent"""
    print("Testing Integration...")
    
    env = TetrisEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, device='cpu')
    
    # Run a few training steps
    total_rewards = []
    
    for episode in range(3):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, steps={step + 1}, lines={info.get('lines_cleared', 0)}")
    
    print(f"✓ Integration test passed!")
    print(f"  Average reward: {np.mean(total_rewards):.2f}")
    
    env.close()
    print()
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Tetris AI - Implementation Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Run tests
        test_environment()
        test_agent()
        test_integration()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("The implementation is working correctly!")
        print("You can now:")
        print("  1. Train the agent: python train.py")
        print("  2. Play with trained model: python play.py --model models/tetris_dqn_final.pth")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
