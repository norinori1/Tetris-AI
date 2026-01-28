"""
AIエージェントの動作をデバッグするスクリプト
ラインが積みあがるか確認
"""
import torch
import numpy as np
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import time

def debug_agent(num_episodes=3, max_steps=500, verbose=True):
    """
    Debug agent behavior
    
    Args:
        num_episodes: Number of episodes to test
        max_steps: Max steps per episode
        verbose: Print detailed logs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("AIエージェント動作デバッグ")
    print("=" * 80)
    print(f"Device: {device}\n")
    
    env = TetrisEnv(render_mode=None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, device=device)
    
    # ランダムなアクション選択（ε=1.0でテスト）
    agent.epsilon = 1.0
    
    action_names = ['左移動', '右移動', '回転', 'ハードドロップ', 'ソフトドロップ']
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_lines = 0
        
        print(f"\n{'='*80}")
        print(f"エピソード {episode + 1}/{num_episodes}")
        print(f"{'='*80}")
        print(f"{'ステップ':<8} {'アクション':<12} {'報酬':<10} {'消除行':<8} {'グリッド高さ':<12} {'穴数':<8}")
        print("-" * 80)
        
        for step in range(max_steps):
            # 状態から盤面統計を取得
            height, holes, bumpiness = env._calculate_board_stats()
            
            # アクション選択
            action = agent.select_action(state, training=True)
            
            # アクション実行
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            lines_cleared = info.get('lines_cleared', 0)
            episode_lines += lines_cleared
            
            # ステップごとの情報を表示
            if verbose or step < 10 or step % 50 == 0 or done:
                print(f"{step:<8} {action_names[action]:<12} {reward:>9.2f} {lines_cleared:>7} {height:>11} {holes:>7}")
            
            state = next_state
            
            if done:
                print("-" * 80)
                print(f"ゲームオーバー! ステップ {step + 1} で終了")
                break
        
        # エピソード統計
        print(f"\n【エピソード {episode + 1} 統計】")
        print(f"  総報酬: {episode_reward:.2f}")
        print(f"  消除ラン数: {episode_lines}")
        print(f"  実行ステップ: {step + 1}")
        print(f"  グリッド最大高さ: {height}")
        print(f"  穴の数: {holes}")
        
        # 最終的なグリッド状態を可視化
        print(f"\n  最終グリッド状態（#=ブロック、.=空）:")
        for y in range(env.grid_height):
            print("    ", end="")
            for x in range(env.grid_width):
                if env.grid[y][x]:
                    print("#", end="")
                else:
                    print(".", end="")
            print()
        
        agent.end_episode()
    
    print("\n" + "=" * 80)
    print("デバッグ完了")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug Tetris AI Agent')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to debug (default: 3)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Max steps per episode (default: 500)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print all steps (default: sample steps)')
    
    args = parser.parse_args()
    
    debug_agent(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=args.verbose
    )
