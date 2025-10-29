'''
    @Info 
    @Assignment: Tic Tac Toe - MinMax
    @Course: MITE 436
    @Author: Ratul Sikder
    @Roll: 2506102
    @Email: ratulsikder.dev@gmail.com
'''
import pygame
import sys
import math
import time
from typing import List, Optional, Tuple, Dict

pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 750
BOARD_SIZE = 500
CELL_SIZE = BOARD_SIZE // 3
BOARD_X = 150
BOARD_Y = 120
ANALYSIS_Y = BOARD_Y + BOARD_SIZE + 30
ANALYSIS_HEIGHT = 150
WHITE = (255, 255, 255)
BLACK = (40, 40, 40)
RED = (231, 76, 60)
BLUE = (52, 152, 219)
GREEN = (46, 204, 113)
GRAY = (149, 165, 166)
LIGHT_GRAY = (236, 240, 241)
GOLD = (241, 196, 15)
BACKGROUND = (247, 249, 250)
font_title = pygame.font.Font(None, 42)
font_large = pygame.font.Font(None, 36)
font_medium = pygame.font.Font(None, 28)
font_small = pygame.font.Font(None, 22)
font_tiny = pygame.font.Font(None, 18)




class TicTacToeAI:
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe  - MinMax")
        self.clock = pygame.time.Clock()
        
        self.board = [" " for _ in range(9)]
        self.human_player = "X"
        self.ai_player = "O"
        self.current_player = self.human_player
        self.game_over = False
        self.winner = None
        
        self.human_wins = 0
        self.ai_wins = 0
        self.ties = 0
        
        self.ai_thinking = False
        self.thinking_start_time = 0
        self.move_evaluations = {}
        self.best_move = None
        self.calculation_tree = []
        self.current_depth = 0
        self.nodes_evaluated = 0
        
        self.user_best_move = None
        self.user_move_evaluations = {}
        self.processing_steps = []
        
        self.thinking_dots = 0
        self.last_dot_time = 0
        
    def reset_game(self):
        self.board = [" " for _ in range(9)]
        self.current_player = self.human_player
        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        self.move_evaluations = {}
        self.best_move = None
        self.calculation_tree = []
        self.nodes_evaluated = 0
        self.user_best_move = None
        self.user_move_evaluations = {}
        self.processing_steps = []
        
    def available_moves(self) -> List[int]:
        return [i for i, spot in enumerate(self.board) if spot == " "]
    
    def make_move(self, position: int, player: str) -> bool:
        if 0 <= position <= 8 and self.board[position] == " ":
            self.board[position] = player
            return True
        return False
    
    def check_winner(self) -> Optional[str]:
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for combo in winning_combinations:
            if (self.board[combo[0]] == self.board[combo[1]] == 
                self.board[combo[2]] != " "):
                return self.board[combo[0]]
        return None
    
    def is_board_full(self) -> bool:
        return " " not in self.board
    
    def is_game_over(self) -> bool:
        return self.check_winner() is not None or self.is_board_full()
    
    def min_max_run(self, depth: int, is_maximizing: bool, alpha=float('-inf'), beta=float('inf')) -> int:
        self.nodes_evaluated += 1
        
        indent = "  " * depth
        player = "AI (MAX)" if is_maximizing else "Human (MIN)"
        print(f"{indent}Depth {depth}: {player} evaluating...")
        
        winner = self.check_winner()
        if winner == self.ai_player:
            print(f"{indent}AI wins! Score: {10 - depth}")
            return 10 - depth
        if winner == self.human_player:
            print(f"{indent}Human wins! Score: {depth - 10}")
            return depth - 10
        if self.is_board_full():
            print(f"{indent}Tie game! Score: 0")
            return 0
        
        if is_maximizing:
            max_eval = float('-inf')
            print(f"{indent}AI trying moves: {[f'({m//3+1},{m%3+1})' for m in self.available_moves()]}")
            
            for move in self.available_moves():
                row, col = move // 3 + 1, move % 3 + 1
                print(f"{indent}AI tries move ({row},{col})")
                
                self.board[move] = self.ai_player
                eval_score = self.min_max_run(depth + 1, False, alpha, beta)
                self.board[move] = " "
                
                print(f"{indent}Move ({row},{col}) score: {eval_score}")
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    print(f"{indent}Pruning branch (alpha={alpha}, beta={beta})")
                    break
            
            print(f"{indent}AI best score at depth {depth}: {max_eval}")
            return max_eval
        else:
            min_eval = float('inf')
            print(f"{indent}Human considering moves: {[f'({m//3+1},{m%3+1})' for m in self.available_moves()]}")
            
            for move in self.available_moves():
                row, col = move // 3 + 1, move % 3 + 1
                print(f"{indent}Human simulates move ({row},{col})")
                
                self.board[move] = self.human_player
                eval_score = self.min_max_run(depth + 1, True, alpha, beta)
                self.board[move] = " "
                
                print(f"{indent}Move ({row},{col}) score: {eval_score}")
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    print(f"{indent}Pruning branch (alpha={alpha}, beta={beta})")
                    break
            
            print(f"{indent}Human best score at depth {depth}: {min_eval}")
            return min_eval
    
    def get_best_move(self) -> Optional[int]:
        if not self.available_moves():
            return None
            
        print("\n" + "="*50)
        print("AI MINIMAX DECISION PROCESS")
        print("="*50)
        
        self.move_evaluations = {}
        self.nodes_evaluated = 0
        best_score = float('-inf')
        best_move = None
        
        available = self.available_moves()
        print(f"Available moves: {[f'({m//3+1},{m%3+1})' for m in available]}")
        print("-" * 50)
        
        for i, move in enumerate(available):
            row, col = move // 3 + 1, move % 3 + 1
            print(f"\nEVALUATING MOVE {i+1}/{len(available)}: ({row},{col})")
            print("-" * 30)
            
            self.board[move] = self.ai_player
            score = self.min_max_run(0, False)
            self.move_evaluations[move] = score
            self.board[move] = " "
            
            print(f"Final score for ({row},{col}): {score}")
            
            if score > best_score:
                best_score = score
                best_move = move
                print(f"NEW BEST MOVE: ({row},{col}) with score {score}")
        
        print("\n" + "="*50)
        print("FINAL DECISION")
        print("="*50)
        if best_move is not None:
            row, col = best_move // 3 + 1, best_move % 3 + 1
            print(f"AI chooses: ({row},{col}) with score {best_score}")
            print(f"Total nodes evaluated: {self.nodes_evaluated}")
        print("="*50 + "\n")
        
        self.best_move = best_move
        self.best_score = best_score
        return best_move
    
    def get_cell_from_mouse(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        
        if (BOARD_X <= x <= BOARD_X + BOARD_SIZE and 
            BOARD_Y <= y <= BOARD_Y + BOARD_SIZE):
            
            col = (x - BOARD_X) // CELL_SIZE
            row = (y - BOARD_Y) // CELL_SIZE
            
            if 0 <= row < 3 and 0 <= col < 3:
                return row * 3 + col
        
        return None
    
    def draw_title(self):
        title_text = "Tic-Tac-Toe: AI vs Human"
        title_surface = font_large.render(title_text, True, BLACK)
        title_rect = title_surface.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(title_surface, title_rect)
    
    def draw_game_status(self):
        score_text = f"Score - You: {self.human_wins}  |  AI: {self.ai_wins}  |  Ties: {self.ties}"
        score_surface = font_medium.render(score_text, True, BLACK)
        self.screen.blit(score_surface, (BOARD_X, BOARD_Y - 50))
        
        if self.game_over:
            if self.winner == self.human_player:
                status_text = "You Win!"
                color = GREEN
            elif self.winner == self.ai_player:
                status_text = "AI Wins!"
                color = RED
            else:
                status_text = "It's a Tie!"
                color = GRAY
        elif self.ai_thinking:
            status_text = "AI is thinking..."
            color = BLUE
        elif self.current_player == self.human_player:
            status_text = "Your turn (X)"
            color = GREEN
        else:
            status_text = "AI's turn (O)"
            color = BLUE
        
        status_surface = font_medium.render(status_text, True, color)
        self.screen.blit(status_surface, (BOARD_X, BOARD_Y - 25))

    def draw_board(self):
        board_rect = pygame.Rect(BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, WHITE, board_rect)
        pygame.draw.rect(self.screen, BLACK, board_rect, 3)
        
        for i in range(1, 3):
            pygame.draw.line(self.screen, BLACK,
                           (BOARD_X + i * CELL_SIZE, BOARD_Y),
                           (BOARD_X + i * CELL_SIZE, BOARD_Y + BOARD_SIZE), 3)
            pygame.draw.line(self.screen, BLACK,
                           (BOARD_X, BOARD_Y + i * CELL_SIZE),
                           (BOARD_X + BOARD_SIZE, BOARD_Y + i * CELL_SIZE), 3)
        
        for i in range(9):
            row = i // 3
            col = i % 3
            center_x = BOARD_X + col * CELL_SIZE + CELL_SIZE // 2
            center_y = BOARD_Y + row * CELL_SIZE + CELL_SIZE // 2
            
            if self.board[i] == " ":
                if i == self.best_move:
                    highlight_rect = pygame.Rect(
                        BOARD_X + col * CELL_SIZE + 3,
                        BOARD_Y + row * CELL_SIZE + 3,
                        CELL_SIZE - 6, CELL_SIZE - 6
                    )
                    pygame.draw.rect(self.screen, GOLD, highlight_rect, 4)
                
                elif (i == self.user_best_move and 
                      self.current_player == self.human_player and 
                      not self.ai_thinking):
                    highlight_rect = pygame.Rect(
                        BOARD_X + col * CELL_SIZE + 3,
                        BOARD_Y + row * CELL_SIZE + 3,
                        CELL_SIZE - 6, CELL_SIZE - 6
                    )
                    pygame.draw.rect(self.screen, GREEN, highlight_rect, 3)
            
            if self.board[i] == "X":
                pygame.draw.line(self.screen, RED,
                               (center_x - 40, center_y - 40),
                               (center_x + 40, center_y + 40), 8)
                pygame.draw.line(self.screen, RED,
                               (center_x + 40, center_y - 40),
                               (center_x - 40, center_y + 40), 8)
            elif self.board[i] == "O":
                pygame.draw.circle(self.screen, BLUE, (center_x, center_y), 45, 8)
            
            if (i in self.move_evaluations and self.board[i] == " " and 
                not self.game_over):
                score = self.move_evaluations[i]
                
                if score > 0:
                    color = GREEN
                    text = f"+{score}"
                elif score < 0:
                    color = RED
                    text = str(score)
                else:
                    color = GRAY
                    text = "0"
                
                score_surface = font_medium.render(text, True, color)
                score_rect = score_surface.get_rect(center=(center_x, center_y))
                self.screen.blit(score_surface, score_rect)
    
    def draw_analysis_panel(self):
        y_offset = ANALYSIS_Y + 20
        
        title_text = "AI MINIMAX ANALYSIS"
        title_surface = font_medium.render(title_text, True, BLACK)
        title_rect = title_surface.get_rect(center=(WINDOW_WIDTH // 2, y_offset))
        self.screen.blit(title_surface, title_rect)
        y_offset += 40
        
        center_x = WINDOW_WIDTH // 2
        
        if self.ai_thinking:
            features = [
                "AI is calculating optimal move...",
                f"Game States Evaluated: {self.nodes_evaluated}",
                "Alpha-Beta Pruning Active",
                "Decision Tree Generation in Progress"
            ]
            
            for i, feature in enumerate(features):
                feature_surface = font_small.render(feature, True, BLUE)
                feature_rect = feature_surface.get_rect(center=(center_x, y_offset + i * 25))
                self.screen.blit(feature_surface, feature_rect)
        
        elif self.best_move is not None:
            row, col = self.best_move // 3 + 1, self.best_move % 3 + 1
            
            result_features = [
                f"Optimal Move: Position ({row},{col})",
                f"Score: {getattr(self, 'best_score', 'N/A')} | Nodes: {self.nodes_evaluated}",
                "------",
                ""
            ]
            
            for i, feature in enumerate(result_features):
                color = BLUE if i == 0 else BLACK
                feature_surface = font_small.render(feature, True, color)
                feature_rect = feature_surface.get_rect(center=(center_x, y_offset + i * 25))
                self.screen.blit(feature_surface, feature_rect)
        
        else:
            info_features = [
                "",
                "",
                "",
                ""
            ]
            
            for i, feature in enumerate(info_features):
                feature_surface = font_small.render(feature, True, BLACK)
                feature_rect = feature_surface.get_rect(center=(center_x, y_offset + i * 25))
                self.screen.blit(feature_surface, feature_rect)
    
    def draw_instructions(self):
        y_start = ANALYSIS_Y + ANALYSIS_HEIGHT + 10
        
        if not self.game_over:
            if self.current_player == self.human_player and not self.ai_thinking:
                instruction = "Click on empty square to place X  |  R: Restart  |  Q: Quit"
            else:
                instruction = "AI is making optimal move...  |  R: Restart  |  Q: Quit"
        else:
            instruction = "Game Over!  |  R: New Game  |  Q: Quit"
        
        instruction_surface = font_tiny.render(instruction, True, GRAY)
        instruction_rect = instruction_surface.get_rect(center=(WINDOW_WIDTH // 2, y_start))
        self.screen.blit(instruction_surface, instruction_rect)
    
    def handle_human_move(self, position: int):
        if self.make_move(position, self.human_player):
            winner = self.check_winner()
            if winner:
                self.winner = winner
                self.game_over = True
                if winner == self.human_player:
                    self.human_wins += 1
                else:
                    self.ai_wins += 1
            elif self.is_board_full():
                self.game_over = True
                self.ties += 1
            else:
                self.current_player = self.ai_player
    
    def handle_ai_move(self):
        if not self.ai_thinking:
            self.ai_thinking = True
            self.thinking_start_time = time.time()
            return
        
        if time.time() - self.thinking_start_time < 1.5:
            return
        
        best_move = self.get_best_move()
        
        if best_move is not None:
            self.make_move(best_move, self.ai_player)
            
            winner = self.check_winner()
            if winner:
                self.winner = winner
                self.game_over = True
                if winner == self.ai_player:
                    self.ai_wins += 1
                else:
                    self.human_wins += 1
            elif self.is_board_full():
                self.game_over = True
                self.ties += 1
            else:
                self.current_player = self.human_player
        
        self.ai_thinking = False
    
    def run(self):
        running = True
        
        while running:
            current_time = time.time()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if (not self.game_over and 
                        self.current_player == self.human_player and 
                        not self.ai_thinking):
                        
                        cell = self.get_cell_from_mouse(event.pos)
                        if cell is not None:
                            self.handle_human_move(cell)
            
            if (not self.game_over and 
                self.current_player == self.ai_player):
                self.handle_ai_move()
            
            if self.ai_thinking and current_time - self.last_dot_time > 0.5:
                self.thinking_dots = (self.thinking_dots + 1) % 4
                self.last_dot_time = current_time
            
            self.screen.fill(BACKGROUND)
            self.draw_title()
            self.draw_game_status()
            self.draw_board()
            self.draw_analysis_panel()
            self.draw_instructions()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    print("Starting Tic-Tac-Toe with Minimax AI")
    print("Check console for detailed algorithm output")
    print("=" * 40)
    
    game = TicTacToeAI()
    game.run()

if __name__ == "__main__":
    main()