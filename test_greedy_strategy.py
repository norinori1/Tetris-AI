"""
統計的に最適なテトリス戦略をテスト
（学習前のベースラインとして機能）
"""
import numpy as np
from tetris_env import TetrisEnv

def greedy_tetris_strategy(env):
    """
    統計的に最適なテトリス戦略：
    - 穴を作らない
    - 高さを最小化する
    - 凹凸を最小化する
    """
    # 各列の高さを計算
    heights = []
    for x in range(env.grid_width):
        for y in range(env.grid_height):
            if env.grid[y][x]:
                heights.append(env.grid_height - y)
                break
        else:
            heights.append(0)
    
    # 最も低い列を狙う戦略
    best_column = np.argmin(heights)
    # 現在のピースの中心列を計算（ピースの幅は不定なので、xをそのまま使う）
    current_column = env.current_piece.x
    
    # 現在の列から目標列へ移動
    if current_column < best_column:
        return 1  # 右移動
    elif current_column > best_column:
        return 0  # 左移動
    else:
        # 列が合ったらハードドロップ
        return 3

def test_greedy_strategy(num_episodes=3, max_steps=500):
    """
    統計的最適戦略でテスト
    """
    env = TetrisEnv(render_mode=None)
    
    print("=" * 80)
    print("統計的最適戦略テスト（ベースライン）")
    print("=" * 80)
    
    total_lines = 0
    total_steps = 0
    
    action_names = ['左移動', '右移動', '回転', 'ハードドロップ', 'ソフトドロップ']
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_lines = 0
        
        print(f"\nエピソード {episode + 1}/{num_episodes}")
        print("-" * 80)
        
        for step in range(max_steps):
            # 統計的に最適なアクション選択
            action = greedy_tetris_strategy(env)
            
            # アクション実行
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            lines = info.get('lines_cleared', 0)
            episode_lines += lines
            total_lines += lines
            
            if lines > 0:
                height, holes, _ = env._calculate_board_stats()
                print(f"  Step {step}: {action_names[action]} → {lines}ラインクリア！"
                      f" (高さ: {height}, 穴: {holes})")
            
            state = next_state
            
            if done:
                print(f"  ゲームオーバー (ステップ {step + 1})")
                total_steps += step + 1
                break
        
        height, holes, _ = env._calculate_board_stats()
        print(f"  【エピソード統計】 消除: {episode_lines}ライン, 最終高さ: {height}, 穴: {holes}")
    
    print("\n" + "=" * 80)
    print(f"【全体統計】")
    print(f"  総消除ライン: {total_lines}")
    print(f"  平均ステップ: {total_steps // num_episodes if total_steps else 0}")
    print(f"  平均1エピソード消除ライン: {total_lines / num_episodes:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    test_greedy_strategy(num_episodes=3)
