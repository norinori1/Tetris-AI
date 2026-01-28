"""
Tetris Environment for Reinforcement Learning
Gym-compatible environment wrapper for the Tetris game
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import os

# Add Tetris-AI to path for importing the original game
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Tetris-AI'))
try:
    from Tetris_AI import Tetromino, SHAPES, COLORS, GRID_WIDTH, GRID_HEIGHT
except ImportError:
    # Fallback for different directory structures
    from Tetris_AI import Tetromino, SHAPES, COLORS, GRID_WIDTH, GRID_HEIGHT


# Constants for reward design
T_PIECE_SHAPE_ID = 2
# v14 報酬設計：ライン消去を最優先に（中間報酬を大幅削減）
HOLE_PENALTY = 3.0           # 穴のペナルティ
HEIGHT_PENALTY = 0.5         # 高さペナルティ
BUMPINESS_PENALTY = 0.3      # 凹凸ペナルティ
SURVIVAL_REWARD = 0.1        # 生存報酬を削減（1.0→0.1）
GAME_OVER_PENALTY = 100      # ゲームオーバーペナルティを増加（10→100）
PIECE_PLACEMENT_REWARD = 0.5 # 配置報酬を削減（3.0→0.5）
# 中間報酬：大幅に削減してライン消去を最優先に
SOME_FILLED_REWARD = 0.05        # 50%以上満杯の行（3.0→0.05）
MOST_FILLED_REWARD = 0.1         # 70%以上満杯の行（8.0→0.1）
ALMOST_FULL_LINE_REWARD = 0.3    # 80%以上満杯の行（20.0→0.3）
VERY_FULL_LINE_REWARD = 0.5      # 90%以上満杯の行（40.0→0.5）  
ONE_AWAY_FROM_CLEAR_REWARD = 1.0 # 9/10埋まった行（80.0→1.0）
# ライン消去報酬：大幅増加してライン消去を最優先に
LINE_CLEAR_BASE_REWARD = 1000.0  # 基本報酬を大幅増（300→1000）


class TetrisEnv(gym.Env):
    """
    Tetris Environment for RL agents
    
    Observation Space: Grid state (20 height x 10 width) + current piece encoding
    Action Space: 
        0: Move left
        1: Move right
        2: Rotate
        3: Hard drop
        4: Soft drop (move down)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space: grid + piece info
        # Grid: 20x10, Piece: 7 one-hot encoded, Position: x, y
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.grid_height * self.grid_width + 7 + 2,), 
            dtype=np.float32
        )
        
        # Initialize game state
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.back_to_back = False
        self.last_clear_difficult = False
        
        # For rendering
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((300, 600))
            pygame.display.set_caption("Tetris AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        else:
            self.screen = None
            self.clock = None
            self.font = None
        
        # Statistics for reward calculation
        self.prev_holes = 0
        self.prev_height = 0
        self.prev_bumpiness = 0
        
        # Auto-fall mechanism
        self.steps_since_fall = 0
        self.fall_frequency = 1  # Piece falls every N steps
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.grid = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.back_to_back = False
        self.last_clear_difficult = False
        
        # Reset statistics
        self.prev_holes = 0
        self.prev_height = 0
        self.prev_bumpiness = 0
        
        # Reset auto-fall counter
        self.steps_since_fall = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation state"""
        # Flatten grid (binary: filled or not)
        grid_flat = []
        for row in self.grid:
            for cell in row:
                grid_flat.append(1.0 if cell else 0.0)
        
        # One-hot encode current piece
        piece_encoding = [0.0] * 7
        piece_encoding[self.current_piece.shape_id] = 1.0
        
        # Normalize position
        pos_x = self.current_piece.x / self.grid_width
        pos_y = self.current_piece.y / self.grid_height
        
        return np.array(grid_flat + piece_encoding + [pos_x, pos_y], dtype=np.float32)
    
    def _check_collision(self, piece, offset_x=0, offset_y=0):
        """Check if piece collides with grid or boundaries"""
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + offset_x
                    new_y = piece.y + y + offset_y
                    if (new_x < 0 or new_x >= self.grid_width or 
                        new_y >= self.grid_height or
                        (new_y >= 0 and self.grid[new_y][new_x])):
                        return True
        return False
    
    def _lock_piece(self):
        """Lock current piece to grid"""
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell and self.current_piece.y + y >= 0:
                    self.grid[self.current_piece.y + y][self.current_piece.x + x] = self.current_piece.color
        
        reward = self._clear_lines()
        
        self.current_piece = self.next_piece
        self.next_piece = Tetromino()
        
        if self._check_collision(self.current_piece):
            self.game_over = True
            
        return reward
    
    def _check_tspin(self):
        """
        Check if current move is a T-Spin
        
        A T-Spin is detected when the T piece is rotated and at least 3 of its
        4 corner positions (in a 3x3 bounding box) are blocked or out of bounds.
        """
        if self.current_piece.shape_id != T_PIECE_SHAPE_ID:  # Not T piece
            return 0
        if not self.current_piece.last_rotation:
            return 0
        
        # Check corners of 3x3 bounding box (assumes T piece fits in 3x3)
        corners = [(0, 0), (2, 0), (0, 2), (2, 2)]
        filled = 0
        for dx, dy in corners:
            check_x = self.current_piece.x + dx
            check_y = self.current_piece.y + dy
            if (check_x < 0 or check_x >= self.grid_width or 
                check_y < 0 or check_y >= self.grid_height or 
                self.grid[check_y][check_x]):
                filled += 1
        
        return 1 if filled >= 3 else 0
    
    def _clear_lines(self):
        """Clear completed lines and calculate reward"""
        lines_to_clear = []
        for y in range(self.grid_height):
            if all(self.grid[y]):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            self.current_piece.last_rotation = False
            return 0
        
        num_lines = len(lines_to_clear)
        is_tspin = self._check_tspin()
        
        # v14: ライン消去を最優先（二次関数的増加）
        # 1ライン: 1000, 2ライン: 4000, 3ライン: 9000, 4ライン: 16000
        reward = 0
        is_difficult = False
        
        if is_tspin and num_lines > 0:
            is_difficult = True
            # T-Spin報酬は基本報酬の2倍
            reward = LINE_CLEAR_BASE_REWARD * num_lines * num_lines * 2
        elif num_lines >= 1:
            if num_lines == 4:
                is_difficult = True
            # 基本報酬: 300 * lines * lines (二次関数的増加)
            reward = LINE_CLEAR_BASE_REWARD * num_lines * num_lines
        
        # Back-to-Back bonus
        if is_difficult and self.last_clear_difficult and self.back_to_back:
            reward = int(reward * 1.5)
        
        if is_difficult:
            self.back_to_back = True
            self.last_clear_difficult = True
        else:
            self.last_clear_difficult = False
        
        self.lines_cleared += num_lines
        
        # Clear lines from grid
        for y in sorted(lines_to_clear, reverse=True):
            del self.grid[y]
            self.grid.insert(0, [0 for _ in range(self.grid_width)])
        
        self.current_piece.last_rotation = False
        return reward
    
    def _calculate_board_stats(self):
        """Calculate board statistics for reward shaping"""
        # Height: maximum column height
        heights = []
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.grid[y][x]:
                    heights.append(self.grid_height - y)
                    break
            else:
                heights.append(0)
        
        max_height = max(heights) if heights else 0
        
        # Holes: empty cells with filled cells above
        holes = 0
        for x in range(self.grid_width):
            block_found = False
            for y in range(self.grid_height):
                if self.grid[y][x]:
                    block_found = True
                elif block_found:
                    holes += 1
        
        # Bumpiness: sum of height differences between adjacent columns
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        # 行の充填率を計算（v8で追加）
        row_fill_rates = []
        for y in range(self.grid_height):
            filled = sum(1 for x in range(self.grid_width) if self.grid[y][x])
            fill_rate = filled / self.grid_width
            row_fill_rates.append(fill_rate)
        
        return max_height, holes, bumpiness, row_fill_rates
    
    def step(self, action):
        """Execute action and return observation, reward, done, info"""
        if self.game_over:
            return self._get_observation(), 0, True, False, {}
        
        reward = 0
        self.current_piece.last_rotation = False
        piece_locked = False
        
        # Execute action
        if action == 0:  # Move left
            if not self._check_collision(self.current_piece, -1, 0):
                self.current_piece.x -= 1
        elif action == 1:  # Move right
            if not self._check_collision(self.current_piece, 1, 0):
                self.current_piece.x += 1
        elif action == 2:  # Rotate
            original_shape = [row[:] for row in self.current_piece.shape]
            self.current_piece.shape = [list(row) for row in zip(*self.current_piece.shape[::-1])]
            self.current_piece.last_rotation = True
            
            # Wall kick
            kicks = [(0, 0), (-1, 0), (1, 0), (0, -1), (-1, -1), (1, -1)]
            success = False
            for dx, dy in kicks:
                if not self._check_collision(self.current_piece, dx, dy):
                    self.current_piece.x += dx
                    self.current_piece.y += dy
                    success = True
                    break
            
            if not success:
                self.current_piece.shape = original_shape
                self.current_piece.last_rotation = False
        elif action == 3:  # Hard drop
            while not self._check_collision(self.current_piece, 0, 1):
                self.current_piece.y += 1
            line_clear_reward = self._lock_piece()
            reward += line_clear_reward
            reward += PIECE_PLACEMENT_REWARD  # ブロック配置成功報酬
            piece_locked = True
        elif action == 4:  # Soft drop
            if not self._check_collision(self.current_piece, 0, 1):
                self.current_piece.y += 1
            else:
                line_clear_reward = self._lock_piece()
                reward += line_clear_reward
                reward += PIECE_PLACEMENT_REWARD  # ブロック配置成功報酬
                piece_locked = True
        
        # Auto-fall mechanism (if piece wasn't locked by action)
        if not piece_locked:
            self.steps_since_fall += 1
            if self.steps_since_fall >= self.fall_frequency:
                self.steps_since_fall = 0
                if not self._check_collision(self.current_piece, 0, 1):
                    self.current_piece.y += 1
                else:
                    # Piece hits bottom, lock it
                    line_clear_reward = self._lock_piece()
                    reward += line_clear_reward
                    reward += PIECE_PLACEMENT_REWARD  # ブロック配置成功報酬
        
        # Calculate board statistics for reward shaping
        height, holes, bumpiness, row_fill_rates = self._calculate_board_stats()
        
        # v14: 中間報酬を大幅削減してライン消去を誘導
        fill_reward = 0
        filled_rows_count = 0  # 埋まっている行の数
        
        for y, fill_rate in enumerate(row_fill_rates):
            filled_count = int(fill_rate * self.grid_width)
            
            # 下の方の行（画面下部）により高い報酬を与える
            # y=19が最下行、y=0が最上行なので、yが大きいほど下
            depth_multiplier = 1.0 + (y / self.grid_height) * 0.5  # 最大1.5倍
            
            # 段階的な報酬で細かく誘導（10マス幅のテトリス用に最適化）
            # 注: grid_width=10を前提とした条件分岐
            if filled_count == 9:  # 9/10マス：残り1マス！超高額報酬
                fill_reward += ONE_AWAY_FROM_CLEAR_REWARD * depth_multiplier
                filled_rows_count += 1.0
            elif fill_rate >= 0.9:  # 90%以上満杯（9マス以上）
                fill_reward += VERY_FULL_LINE_REWARD * depth_multiplier
                filled_rows_count += 0.8
            elif fill_rate >= 0.8:  # 80%以上満杯（8マス以上）
                fill_reward += ALMOST_FULL_LINE_REWARD * depth_multiplier
                filled_rows_count += 0.6
            elif fill_rate >= 0.7:  # 70%以上満杯（7マス以上）
                fill_reward += MOST_FILLED_REWARD * depth_multiplier
                filled_rows_count += 0.4
            elif fill_rate >= 0.5:  # 50%以上満杯（5マス以上）
                fill_reward += SOME_FILLED_REWARD * depth_multiplier
                filled_rows_count += 0.2
        
        # 複数行が同時に埋まっている場合にボーナス（連鎖を促す）
        if filled_rows_count >= 2:
            fill_reward *= 1.5
        
        reward += fill_reward
        
        # Reward shaping using defined constants
        reward -= (holes - self.prev_holes) * HOLE_PENALTY
        reward -= (height - self.prev_height) * HEIGHT_PENALTY
        reward -= (bumpiness - self.prev_bumpiness) * BUMPINESS_PENALTY
        
        self.prev_holes = holes
        self.prev_height = height
        self.prev_bumpiness = bumpiness
        
        # Game over penalty
        if self.game_over:
            reward -= GAME_OVER_PENALTY
        
        # Small survival reward
        reward += SURVIVAL_REWARD
        
        terminated = self.game_over
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, {
            'score': self.score,
            'lines_cleared': self.lines_cleared,
            'height': height,
            'holes': holes
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode != 'human':
            return
        
        BLOCK_SIZE = 30
        BLACK = (0, 0, 0)
        GRAY = (128, 128, 128)
        WHITE = (255, 255, 255)
        
        self.screen.fill(BLACK)
        
        # Draw grid
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                color = self.grid[y][x] if self.grid[y][x] else GRAY
                pygame.draw.rect(self.screen, color, 
                               (x * BLOCK_SIZE, y * BLOCK_SIZE, 
                                BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        
        # Draw current piece
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, self.current_piece.color,
                                   ((self.current_piece.x + x) * BLOCK_SIZE,
                                    (self.current_piece.y + y) * BLOCK_SIZE,
                                    BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        
        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.metadata['render_fps'])
    
    def close(self):
        """Close the environment"""
        if self.screen is not None:
            pygame.quit()
