

import pygame
import numpy as np
import random
import time
import math
from collections import namedtuple

# ------------------------------- CONFIG -------------------------------
BOARD_SIZE = 8
SQUARE_SIZE = 80
WIDTH = HEIGHT = BOARD_SIZE * SQUARE_SIZE
FPS = 60

# AI settings
DIFFICULTY_LEVELS = {
    'easy': {'depth': 2, 'blunder_chance': 0.25, 'think_time': (0.3, 0.8)},
    'medium': {'depth': 4, 'blunder_chance': 0.12, 'think_time': (0.6, 1.4)},
    'hard': {'depth': 6, 'blunder_chance': 0.04, 'think_time': (1.0, 2.0)}
}
DEFAULT_DIFFICULTY = 'medium'

# Piece values
MAN_VALUE = 100
KING_VALUE = 175

# colors
WHITE = (245, 245, 245)
BLACK = (40, 40, 40)
LIGHT_SQUARE = (232, 235, 239)
DARK_SQUARE = (90, 110, 125)
HIGHLIGHT = (50, 200, 50, 140)
RED = (200, 30, 30)
BLUE = (60, 90, 200)

# Named tuple for moves
Move = namedtuple('Move', ['start', 'sequence'])

# ------------------------------- BOARD REPRESENTATION -------------------------------
# 0 = empty
# 1 = red man
# 2 = red king
# -1 = black man
# -2 = black king

class CheckersBoard:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.reset()

    def reset(self):
        self.board.fill(0)
        # Place black on top rows (negative)
        for r in range(3):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    self.board[r, c] = -1
        # Place red on bottom rows
        for r in range(5, 8):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    self.board[r, c] = 1

    def inside(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def get_piece(self, r, c):
        return self.board[r, c]

    def move_piece(self, move: Move):
        # move.sequence is a list of positions (r,c) the piece visits, including captures
        sr, sc = move.start
        piece = self.board[sr, sc]
        self.board[sr, sc] = 0
        tr, tc = move.sequence[-1]
        self.board[tr, tc] = piece
        # Remove captured pieces if jump
        if len(move.sequence) > 1:
            r0, c0 = sr, sc
            for (r1, c1) in move.sequence:
                dr = r1 - r0
                dc = c1 - c0
                if abs(dr) == 2 and abs(dc) == 2:
                    midr = (r0 + r1) // 2
                    midc = (c0 + c1) // 2
                    self.board[midr, midc] = 0
                r0, c0 = r1, c1
        # King me if reach end
        if piece == 1 and tr == 0:
            self.board[tr, tc] = 2
        if piece == -1 and tr == BOARD_SIZE - 1:
            self.board[tr, tc] = -2

    def copy(self):
        b = CheckersBoard()
        b.board = self.board.copy()
        return b

    def pieces(self, color):
        # color = 1 for red, -1 for black
        coords = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] * color > 0:
                    coords.append((r, c))
        return coords

# ------------------------------- MOVE GENERATION -------------------------------

def get_legal_moves(board: CheckersBoard, color: int):
    # Return list of Move(start, sequence)
    captures = []
    quiet = []
    for (r, c) in board.pieces(color):
        piece = board.get_piece(r, c)
        is_king = abs(piece) == 2
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if not is_king:
            # men move forward only
            if color == 1:
                directions = [(-1, -1), (-1, 1)]
            else:
                directions = [(1, -1), (1, 1)]
        # look for captures (jumps) using DFS for multi-capture
        def dfs_capture(r0, c0, visited_board, path, results):
            found = False
            for dr, dc in directions:
                r1, c1 = r0 + dr, c0 + dc
                r2, c2 = r0 + 2 * dr, c0 + 2 * dc
                if visited_board.inside(r2, c2) and visited_board.inside(r1, c1):
                    if visited_board.get_piece(r2, c2) == 0 and visited_board.get_piece(r1, c1) * color < 0:
                        # perform jump on a copied visited_board
                        nb = visited_board.copy()
                        nb.board[r0, c0] = 0
                        nb.board[r1, c1] = 0
                        nb.board[r2, c2] = piece
                        # handle kinging mid-capture: allow further jumps only if king
                        new_is_king = is_king or (piece == 1 and r2 == 0) or (piece == -1 and r2 == BOARD_SIZE - 1)
                        # For multi-captures we should consider new directions if kinged
                        subdirs = directions
                        if new_is_king and not is_king:
                            subdirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                        # DFS continues
                        found = True
                        dfs_capture(r2, c2, nb, path + [(r2, c2)], results)
            if not found and len(path) > 0:
                results.append(Move(start=(r, c), sequence=path))

        results = []
        dfs_capture(r, c, board, [], results)
        captures.extend(results)
        if len(results) == 0:
            # check normal moves
            for dr, dc in directions:
                r1, c1 = r + dr, c + dc
                if board.inside(r1, c1) and board.get_piece(r1, c1) == 0:
                    quiet.append(Move(start=(r, c), sequence=[(r1, c1)]))
    # If any captures exist, rules demand capture moves only
    return captures if len(captures) > 0 else quiet

# ------------------------------- EVALUATION -------------------------------

def evaluate(board: CheckersBoard, color: int):
    # Positive means good for `color` player
    val = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board.board[r, c]
            if p == 0:
                continue
            sign = 1 if p > 0 else -1
            piece_val = KING_VALUE if abs(p) == 2 else MAN_VALUE
            # positional bonus: center control and advancement
            center_bonus = (3 - abs(3.5 - c)) * 3
            advance_bonus = 0
            if p > 0:
                advance_bonus = (7 - r)  # red wants to go up (r smaller)
            else:
                advance_bonus = r  # black wants to go down
            val += sign * (piece_val + center_bonus + advance_bonus)
    return val * color

# ------------------------------- MINIMAX / AI -------------------------------

def minimax(board: CheckersBoard, depth: int, alpha: float, beta: float, maximizing: bool, player_color: int):
    # returning (score, move)
    # Terminal checks: no moves -> loss
    moves = get_legal_moves(board, player_color if maximizing else -player_color)
    if depth == 0 or len(moves) == 0:
        return evaluate(board, player_color), None

    best_move = None
    if maximizing:
        max_eval = -math.inf
        for mv in moves:
            nb = board.copy()
            nb.move_piece(mv)
            score, _ = minimax(nb, depth - 1, alpha, beta, False, player_color)
            if score > max_eval:
                max_eval = score
                best_move = mv
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for mv in moves:
            nb = board.copy()
            nb.move_piece(mv)
            score, _ = minimax(nb, depth - 1, alpha, beta, True, player_color)
            if score < min_eval:
                min_eval = score
                best_move = mv
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_eval, best_move


def ai_select_move(board: CheckersBoard, color: int, difficulty='medium'):
    settings = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS[DEFAULT_DIFFICULTY])
    depth = settings['depth']
    blunder_chance = settings['blunder_chance']
    think_time_min, think_time_max = settings['think_time']

    # Simulate human thinking time (non-blocking in UI — here we'll block for simplicity, but add visual animation in UI)
    think_time = random.uniform(think_time_min, think_time_max)
    time.sleep(0.05)  # tiny pause

    # Get candidate moves with search
    score, best_move = minimax(board, depth, -math.inf, math.inf, True, color)
    moves = get_legal_moves(board, color)
    if not moves:
        return None

    # Build move list with heuristic ordering
    scored_moves = []
    for mv in moves:
        nb = board.copy()
        nb.move_piece(mv)
        s = evaluate(nb, color)
        # small randomness to emulate noisy human evaluation
        s += random.uniform(-5, 5)
        scored_moves.append((s, mv))
    scored_moves.sort(key=lambda x: x[0], reverse=True)

    # Blunder: sometimes choose a worse move
    if random.random() < blunder_chance and len(scored_moves) > 1:
        # pick from bottom half sometimes
        idx = random.randint(len(scored_moves) // 2, len(scored_moves) - 1)
        chosen = scored_moves[idx][1]
    else:
        # mostly pick top choices, but sometimes second-best
        r = random.random()
        if r < 0.85:
            chosen = scored_moves[0][1]
        elif len(scored_moves) > 1:
            chosen = scored_moves[1][1]
        else:
            chosen = scored_moves[0][1]

    # Occasionally do a "hesitation": simulate change of mind by returning a move after a tiny second pause
    if random.random() < 0.12:
        time.sleep(random.uniform(0.15, 0.35))
    return chosen

# ------------------------------- PYGAME UI -------------------------------

class GameUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Checkers')
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + 60))
        self.clock = pygame.time.Clock()
        self.board = CheckersBoard()
        self.selected = None
        self.legal_moves_cache = []
        self.turn = 1  # 1 = red (human by default), -1 = black
        self.running = True
        self.ai_enabled = True
        self.difficulty = DEFAULT_DIFFICULTY
        self.font = pygame.font.SysFont(None, 22)
        self.large_font = pygame.font.SysFont(None, 26)
        self.ai_thinking = False
        self.ai_animation_start = 0
        self.ai_animation_duration = 0

    def board_to_screen(self, pos):
        r, c = pos
        return c * SQUARE_SIZE, r * SQUARE_SIZE

    def screen_to_board(self, x, y):
        return y // SQUARE_SIZE, x // SQUARE_SIZE

    def draw_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = DARK_SQUARE if (r + c) % 2 else LIGHT_SQUARE
                rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

    def draw_pieces(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board.get_piece(r, c)
                if p == 0:
                    continue
                cx = c * SQUARE_SIZE + SQUARE_SIZE // 2
                cy = r * SQUARE_SIZE + SQUARE_SIZE // 2
                radius = SQUARE_SIZE // 2 - 6
                if p > 0:
                    pygame.draw.circle(self.screen, RED, (cx, cy), radius)
                    if p == 2:
                        pygame.draw.circle(self.screen, WHITE, (cx, cy), radius - 14)
                else:
                    pygame.draw.circle(self.screen, BLUE, (cx, cy), radius)
                    if p == -2:
                        pygame.draw.circle(self.screen, WHITE, (cx, cy), radius - 14)

    def highlight_moves(self, moves):
        s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        for mv in moves:
            sr, sc = mv.start
            # highlight start square
            rect = pygame.Rect(sc * SQUARE_SIZE, sr * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.screen, (80, 200, 120, 60), rect)
            # highlight targets
            for (r, c) in mv.sequence:
                rect2 = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, (50, 180, 240, 80), rect2)

    def draw_ui_bar(self):
        bar_rect = pygame.Rect(0, HEIGHT, WIDTH, 60)
        pygame.draw.rect(self.screen, BLACK, bar_rect)
        status = f"Turn: {'Red (You)' if self.turn == 1 else 'Black (AI)'} | AI: {'On' if self.ai_enabled else 'Off'} | Difficulty: {self.difficulty}"
        text = self.font.render(status, True, WHITE)
        self.screen.blit(text, (8, HEIGHT + 8))
        help_text = self.font.render("Click & drag to move. Keys: R=Reset, A=Toggle AI, D=Cycle Difficulty", True, WHITE)
        self.screen.blit(help_text, (8, HEIGHT + 32))

        if self.ai_thinking:
            elapsed = time.time() - self.ai_animation_start
            dots = int((elapsed * 3) % 4)
            thinking_text = self.large_font.render('AI thinking' + '.' * dots, True, WHITE)
            self.screen.blit(thinking_text, (WIDTH - 200, HEIGHT + 18))

    def recompute_legal_moves_for_selected(self):
        if self.selected is None:
            self.legal_moves_cache = []
            return
        smoves = get_legal_moves(self.board, self.turn)
        # filter to moves that start at selected
        self.legal_moves_cache = [m for m in smoves if m.start == self.selected]

    def handle_click(self, pos):
        r, c = pos
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return
        piece = self.board.get_piece(r, c)
        if piece * self.turn > 0:
            # select piece
            self.selected = (r, c)
            self.recompute_legal_moves_for_selected()
        elif self.selected is not None:
            # try to play move if legal
            chosen = None
            for mv in self.legal_moves_cache:
                if mv.sequence and mv.sequence[-1] == (r, c):
                    chosen = mv
                    break
            if chosen:
                self.board.move_piece(chosen)
                self.selected = None
                self.legal_moves_cache = []
                self.turn *= -1

    def ai_move(self):
        # kick off AI move with humanized animations
        self.ai_thinking = True
        self.ai_animation_start = time.time()
        settings = DIFFICULTY_LEVELS.get(self.difficulty, DIFFICULTY_LEVELS[DEFAULT_DIFFICULTY])
        think_time_min, think_time_max = settings['think_time']
        # Non-blocking loop with small sleeps to allow animation
        think_until = time.time() + random.uniform(think_time_min, think_time_max)
        while time.time() < think_until:
            # update display so dots animate
            self.render()
            pygame.display.flip()
            self.clock.tick(FPS)
        # compute move (blocking)
        mv = ai_select_move(self.board, self.turn, self.difficulty)
        self.ai_thinking = False
        if mv:
            self.board.move_piece(mv)
        self.turn *= -1

    def cycle_difficulty(self):
        keys = list(DIFFICULTY_LEVELS.keys())
        idx = keys.index(self.difficulty)
        self.difficulty = keys[(idx + 1) % len(keys)]

    def render(self):
        self.screen.fill(WHITE)
        self.draw_board()
        # highlight legal moves for selected piece
        if self.selected:
            self.highlight_moves(self.legal_moves_cache)
        self.draw_pieces()
        self.draw_ui_bar()

    def run(self):
        drag = False
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.board.reset()
                        self.turn = 1
                        self.selected = None
                    elif event.key == pygame.K_a:
                        self.ai_enabled = not self.ai_enabled
                    elif event.key == pygame.K_d:
                        self.cycle_difficulty()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if my < HEIGHT:
                        r, c = self.screen_to_board(mx, my)
                        self.handle_click((r, c))
                        drag = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    drag = False
            # If it's AI's turn and AI enabled, let it move
            if self.turn == -1 and self.ai_enabled and not self.ai_thinking:
                # small delay to feel natural
                self.ai_move()

            self.render()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

# ------------------------------- MAIN -------------------------------

if __name__ == '__main__':
    ui = GameUI()
    ui.run()
