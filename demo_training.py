"""
Quick training demo to verify the entire training pipeline works
"""
import os
import sys

# Make sure we can import the modules
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import torch
import numpy as np

def quick_train_demo():
    """Run a very short training to verify everything works"""
    print("=" * 60)
    print("Quick Training Demo - Verifying Training Pipeline")
    print("=" * 60)
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    env = TetrisEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, device=device)
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print()
    
    # Quick training for 10 episodes
    num_episodes = 10
    max_steps = 100
    
    print(f"Training for {num_episodes} episodes...")
    print()
    
    episode_rewards = []
    episode_lines = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # End of episode - decay epsilon per episode
        agent.end_episode()
        
        episode_rewards.append(episode_reward)
        episode_lines.append(info.get('lines_cleared', 0))
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"reward={episode_reward:.2f}, "
              f"lines={info.get('lines_cleared', 0)}, "
              f"steps={step + 1}, "
              f"epsilon={agent.epsilon:.3f}")
    
    env.close()
    
    # Test saving model
    test_model_dir = '/tmp/test_models'
    os.makedirs(test_model_dir, exist_ok=True)
    model_path = os.path.join(test_model_dir, 'test_model.pth')
    agent.save(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Test loading model
    new_agent = DQNAgent(state_size, action_size, device=device)
    new_agent.load(model_path)
    print(f"✓ Model loaded successfully")
    
    # Test evaluation
    env2 = TetrisEnv()
    state, _ = env2.reset()
    action = new_agent.select_action(state, training=False)
    print(f"✓ Loaded model can select actions (action: {action})")
    env2.close()
    
    print()
    print("=" * 60)
    print("Quick Training Demo Complete!")
    print("=" * 60)
    print()
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average lines: {np.mean(episode_lines):.2f}")
    print()
    print("The training pipeline is working correctly! 🎉")
    print()
    print("To train a full model, run:")
    print("  python train.py")
    print()
    print("For more episodes (better performance):")
    print("  Edit train.py and change NUM_EPISODES to 5000 or 10000")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

if __name__ == '__main__':
    try:
        quick_train_demo()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
