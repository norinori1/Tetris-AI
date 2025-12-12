import pygame
import random
import sys

# 初期化
pygame.init()

# 定数
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = BLOCK_SIZE * (GRID_WIDTH + 8)
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT
FPS = 60

# 色定義
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
COLORS = [
    (0, 255, 255),  # I
    (255, 255, 0),  # O
    (128, 0, 128),  # T
    (0, 255, 0),    # S
    (255, 0, 0),    # Z
    (0, 0, 255),    # J
    (255, 165, 0),  # L
]

# テトリミノ形状
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[0, 1, 0], [1, 1, 1]],  # T
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]],  # L
]

class Tetromino:
    def __init__(self, shape_id=None):
        if shape_id is None:
            shape_id = random.randint(0, len(SHAPES) - 1)
        self.shape_id = shape_id
        self.shape = [row[:] for row in SHAPES[shape_id]]
        self.color = COLORS[shape_id]
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
        self.last_rotation = False  # T-Spin判定用

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]
        self.last_rotation = True

class TetrisGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris - T-Spin & BTB")
        self.clock = pygame.time.Clock()
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.game_over = False
        self.score = 0
        self.lines_cleared = 0
        self.fall_time = 0
        self.fall_speed = 500
        self.back_to_back = False
        self.last_clear_difficult = False
        self.font = pygame.font.Font(None, 24)
        self.message = ""
        self.message_time = 0

    def check_collision(self, piece, offset_x=0, offset_y=0):
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + offset_x
                    new_y = piece.y + y + offset_y
                    if (new_x < 0 or new_x >= GRID_WIDTH or 
                        new_y >= GRID_HEIGHT or
                        (new_y >= 0 and self.grid[new_y][new_x])):
                        return True
        return False

    def lock_piece(self):
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell and self.current_piece.y + y >= 0:
                    self.grid[self.current_piece.y + y][self.current_piece.x + x] = self.current_piece.color
        self.clear_lines()
        self.current_piece = self.next_piece
        self.next_piece = Tetromino()
        if self.check_collision(self.current_piece):
            self.game_over = True

    def check_tspin(self):
        if self.current_piece.shape_id != 2:  # Tピース以外
            return 0
        if not self.current_piece.last_rotation:
            return 0
        
        corners = [(0, 0), (2, 0), (0, 2), (2, 2)]
        filled = 0
        for dx, dy in corners:
            check_x = self.current_piece.x + dx
            check_y = self.current_piece.y + dy
            if (check_x < 0 or check_x >= GRID_WIDTH or 
                check_y < 0 or check_y >= GRID_HEIGHT or 
                self.grid[check_y][check_x]):
                filled += 1
        
        return 1 if filled >= 3 else 0

    def clear_lines(self):
        lines_to_clear = []
        for y in range(GRID_HEIGHT):
            if all(self.grid[y]):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            self.current_piece.last_rotation = False
            return
        
        num_lines = len(lines_to_clear)
        is_tspin = self.check_tspin()
        
        # スコア計算
        bonus_points = 0
        clear_name = ""
        is_difficult = False
        
        if is_tspin and num_lines > 0:
            is_difficult = True
            if num_lines == 1:
                bonus_points = 800
                clear_name = "T-Spin Single"
            elif num_lines == 2:
                bonus_points = 1200
                clear_name = "T-Spin Double"
            elif num_lines == 3:
                bonus_points = 1600
                clear_name = "T-Spin Triple"
        elif num_lines == 4:
            is_difficult = True
            bonus_points = 800
            clear_name = "Tetris"
        else:
            bonus_points = num_lines * 100
            clear_name = f"{num_lines} Line{'s' if num_lines > 1 else ''}"
        
        # Back-to-Back判定
        if is_difficult and self.last_clear_difficult and self.back_to_back:
            bonus_points = int(bonus_points * 1.5)
            clear_name += " BTB"
        
        if is_difficult:
            self.back_to_back = True
            self.last_clear_difficult = True
        else:
            self.last_clear_difficult = False
        
        self.score += bonus_points
        self.lines_cleared += num_lines
        self.message = f"{clear_name} +{bonus_points}"
        self.message_time = pygame.time.get_ticks()
        
        # ライン削除
        for y in sorted(lines_to_clear, reverse=True):
            del self.grid[y]
            self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
        
        self.current_piece.last_rotation = False

    def move(self, dx, dy):
        if not self.check_collision(self.current_piece, dx, dy):
            self.current_piece.x += dx
            self.current_piece.y += dy
            self.current_piece.last_rotation = False
            return True
        return False

    def rotate_piece(self):
        original_shape = [row[:] for row in self.current_piece.shape]
        self.current_piece.rotate()
        
        # Wall kick試行
        kicks = [(0, 0), (-1, 0), (1, 0), (0, -1), (-1, -1), (1, -1)]
        for dx, dy in kicks:
            if not self.check_collision(self.current_piece, dx, dy):
                self.current_piece.x += dx
                self.current_piece.y += dy
                return
        
        self.current_piece.shape = original_shape
        self.current_piece.last_rotation = False

    def hard_drop(self):
        while not self.check_collision(self.current_piece, 0, 1):
            self.current_piece.y += 1
        self.lock_piece()

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                color = self.grid[y][x] if self.grid[y][x] else GRAY
                pygame.draw.rect(self.screen, color, 
                               (x * BLOCK_SIZE, y * BLOCK_SIZE, 
                                BLOCK_SIZE - 1, BLOCK_SIZE - 1))

    def draw_piece(self, piece, offset_x=0, offset_y=0):
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, piece.color,
                                   ((piece.x + x + offset_x) * BLOCK_SIZE,
                                    (piece.y + y + offset_y) * BLOCK_SIZE,
                                    BLOCK_SIZE - 1, BLOCK_SIZE - 1))

    def draw_next_piece(self):
        text = self.font.render("Next:", True, WHITE)
        self.screen.blit(text, (GRID_WIDTH * BLOCK_SIZE + 10, 50))
        
        for y, row in enumerate(self.next_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, self.next_piece.color,
                                   ((GRID_WIDTH + 1 + x) * BLOCK_SIZE,
                                    (3 + y) * BLOCK_SIZE,
                                    BLOCK_SIZE - 1, BLOCK_SIZE - 1))

    def draw_ui(self):
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        lines_text = self.font.render(f"Lines: {self.lines_cleared}", True, WHITE)
        btb_text = self.font.render(f"BTB: {'ON' if self.back_to_back else 'OFF'}", True, 
                                   (255, 255, 0) if self.back_to_back else WHITE)
        
        self.screen.blit(score_text, (GRID_WIDTH * BLOCK_SIZE + 10, 200))
        self.screen.blit(lines_text, (GRID_WIDTH * BLOCK_SIZE + 10, 230))
        self.screen.blit(btb_text, (GRID_WIDTH * BLOCK_SIZE + 10, 260))
        
        # メッセージ表示
        if pygame.time.get_ticks() - self.message_time < 2000:
            msg = self.font.render(self.message, True, (255, 255, 0))
            self.screen.blit(msg, (GRID_WIDTH * BLOCK_SIZE + 10, 300))

    def run(self):
        while not self.game_over:
            dt = self.clock.tick(FPS)
            self.fall_time += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.move(1, 0)
                    elif event.key == pygame.K_DOWN:
                        self.move(0, 1)
                    elif event.key == pygame.K_UP:
                        self.rotate_piece()
                    elif event.key == pygame.K_SPACE:
                        self.hard_drop()

            if self.fall_time > self.fall_speed:
                if not self.move(0, 1):
                    self.lock_piece()
                self.fall_time = 0

            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_piece(self.current_piece)
            self.draw_next_piece()
            self.draw_ui()
            pygame.display.flip()

        # ゲームオーバー
        game_over_text = self.font.render("GAME OVER", True, WHITE)
        self.screen.blit(game_over_text, 
                        (SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2))
        pygame.display.flip()
        pygame.time.wait(3000)

if __name__ == "__main__":
    game = TetrisGame()
    game.run()
    pygame.quit()