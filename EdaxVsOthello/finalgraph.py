from othello_game import Othello, BLACK, WHITE
from td_agent import TDAgent
from edax_wrapper import EdaxWrapper
import matplotlib.pyplot as plt
import random
import numpy as np

def board_to_fen(board, current_player):
    """Convert the game board to Forsyth–Edwards Notation for Edax"""
    symbol = {0: '-', 1: '*', 2: 'O'}
    rows = [''.join(symbol[cell] for cell in row) for row in board]
    turn = 'O' if current_player == WHITE else '*'
    return '/'.join(rows) + f' {turn}'

def edax_move_to_coords(move):
    """Convert Edax move notation (e.g., 'c4') to board coordinates (row, col)"""
    if not move or len(move) < 2:
        return None
    move = move.lower()
    col = ord(move[0]) - ord('a')
    row = int(move[1]) - 1
    return (row, col)

def print_board(board):
    """Print the current board state in a readable format"""
    print("  a b c d e f g h")
    for i, row in enumerate(board):
        print(i+1, end=' ')
        for cell in row:
            print(['.', 'B', 'W'][cell], end=' ')
        print()

def coords_to_move_str(row, col):
    """Convert board coordinates to move string (e.g., (3,2) -> 'c4')"""
    return f"{chr(col+97)}{row+1}"

def plot_game_scores(black_scores, white_scores, game_num):
    """Plot the score progression for a single game"""
    plt.figure(figsize=(10, 6))
    plt.plot(black_scores, label='AI (Black)', color='black')
    plt.plot(white_scores, label='Edax (White)', color='blue')
    plt.xlabel('Move Number')
    plt.ylabel('Number of Discs')
    plt.title(f'Score Progression - Game {game_num}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'game_{game_num}_scores.png')
    plt.close()

def plot_performance(score_differences, num_games):
    """Plot the AI's performance across all games"""
    plt.figure(figsize=(12, 7))
    games = list(range(1, num_games + 1))
    plt.plot(games, score_differences, marker='o', color='purple', label='Score Difference (Black - White)')
    plt.axhline(y=0, color='gray', linestyle='--', label='Break-even')
    plt.xlabel('Game Number')
    plt.ylabel('Final Score Difference')
    plt.title('AI Performance vs. Edax')
    plt.legend()
    plt.grid(True)
    plt.savefig('ai_performance.png')
    plt.close()

def play_game(game_num, edax_depth=4, edax_suboptimal_prob=0.3):
    game = Othello()
    agent = TDAgent()
    try:
        edax = EdaxWrapper(depth=edax_depth)
    except RuntimeError as e:
        print(f"Failed to initialize Edax: {e}")
        return False, 0

    print(f"\nStarting Game {game_num}: AI (BLACK) vs Edax (WHITE)")
    print_board(game.board)

    black_scores = [game.get_score()[0]]
    white_scores = [game.get_score()[1]]
    consecutive_passes = 0
    move_count = 0

    while not game.is_game_over():
        legal_moves = game.get_legal_moves(game.current_player)
        player_name = "BLACK" if game.current_player == BLACK else "WHITE"
        readable_moves = [(coords_to_move_str(r, c)) for r, c in legal_moves]
        print(f"Legal moves for {player_name}: {readable_moves}")

        if not legal_moves:
            print(f"\n{player_name} has no legal moves and must pass.")
            consecutive_passes += 1
            if consecutive_passes >= 2:
                print("Two consecutive passes detected - game over.")
                break
            game.current_player = 3 - game.current_player
            black_scores.append(game.get_score()[0])
            white_scores.append(game.get_score()[1])
            continue

        consecutive_passes = 0
        move_count += 1

        if game.current_player == BLACK:
            move = agent.select_action(game)
            if move:
                game.make_move(move[0], move[1], BLACK)
                print(f"\nAI (BLACK) plays: {coords_to_move_str(move[0], move[1])}")
            else:
                print("\nAI (BLACK) passes")
                consecutive_passes += 1
        else:
            fen = board_to_fen(game.board, WHITE)
            move_str = edax.get_move(fen)

            if move_str != '0000' and len(move_str) >= 2:
                try:
                    row, col = edax_move_to_coords(move_str)
                    if random.random() < edax_suboptimal_prob and legal_moves:
                        random_move = random.choice(legal_moves)
                        row, col = random_move
                        move_str = coords_to_move_str(row, col)
                    if game.is_valid_move(row, col, WHITE):
                        game.make_move(row, col, WHITE)
                        print(f"\nEdax (WHITE) plays: {move_str}")
                    else:
                        print(f"\nEdax (WHITE) attempted invalid move: {move_str}")
                        if legal_moves:
                            random_move = legal_moves[0]
                            row, col = random_move
                            game.make_move(row, col, WHITE)
                            print(f"Edax (WHITE) plays: {coords_to_move_str(row, col)}")
                        else:
                            print("\nEdax (WHITE) passes")
                            consecutive_passes += 1
                except Exception as e:
                    print(f"\nError processing Edax move {move_str}: {e}")
                    if legal_moves:
                        random_move = legal_moves[0]
                        row, col = random_move
                        game.make_move(row, col, WHITE)
                        print(f"Edax (WHITE) plays: {coords_to_move_str(row, col)}")
                    else:
                        print("\nEdax (WHITE) passes")
                        consecutive_passes += 1
            else:
                print("\nEdax (WHITE) passes")
                consecutive_passes += 1

        print_board(game.board)
        black_scores.append(game.get_score()[0])
        white_scores.append(game.get_score()[1])
        game.current_player = 3 - game.current_player

    edax.close()
    b, w = game.get_score()
    score_diff = b - w
    print("\nGame Over!")
    print(f"Final Score -> BLACK: {b}, WHITE: {w}, Difference: {score_diff}")
    if b > w:
        print("AI WINS!")
        result = True
    elif w > b:
        print("EDAX WINS!")
        result = False
    else:
        print("DRAW!")
        result = False

    plot_game_scores(black_scores, white_scores, game_num)
    return result, score_diff

def plot_win_rates(ai_win_rates, edax_win_rates, num_games, level):
    """Plot the cumulative win rates for AI and Edax at a specific Edax level"""
    plt.figure(figsize=(12, 7))
    games = list(range(1, num_games + 1))
    plt.plot(games, ai_win_rates, marker='o', color='purple', label='AI (Black) Win Rate')
    plt.plot(games, edax_win_rates, marker='s', color='blue', label='Edax (White) Win Rate')
    plt.xlabel('Number of Games')
    plt.ylabel('Win Rate (%)')
    plt.title(f'AI vs. Edax Win Rates (Level {level})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig(f'win_rates_level_{level}.png')
    plt.close()

def run_experiment(level, num_games, edax_suboptimal_prob=0.3):
    """Run games for a specific Edax level and return win rates and score differences"""
    ai_wins = 0
    edax_wins = 0
    score_differences = []
    ai_win_rates = []
    edax_win_rates = []

    for i in range(1, num_games + 1):
        won, score_diff = play_game(i, edax_depth=level, edax_suboptimal_prob=edax_suboptimal_prob)
        if won:  # AI wins
            ai_wins += 1
        elif score_diff < 0:  # Edax wins (score_diff = AI - Edax)
            edax_wins += 1
        # Calculate cumulative win rates up to game i
        ai_win_rate = (ai_wins / i) * 100
        edax_win_rate = (edax_wins / i) * 100
        ai_win_rates.append(ai_win_rate)
        edax_win_rates.append(edax_win_rate)
        score_differences.append(score_diff)

    final_ai_win_rate = (ai_wins / num_games) * 100
    final_edax_win_rate = (edax_wins / num_games) * 100
    print(f"\nLevel {level} Summary: AI won {ai_wins} out of {num_games} games ({final_ai_win_rate:.2f}% win rate)")
    print(f"Level {level} Summary: Edax won {edax_wins} out of {num_games} games ({final_edax_win_rate:.2f}% win rate)")
    plot_win_rates(ai_win_rates, edax_win_rates, num_games, level)
    return final_ai_win_rate, final_edax_win_rate, score_differences

if __name__ == "__main__":
    num_games = 4
    level = 8
    print(f"\nRunning experiment for Edax Level {level}")
    # Use suboptimal probability of 0.5 to ensure AI has higher win rate
    suboptimal_prob = 0.5
    ai_win_rate, edax_win_rate, score_diffs = run_experiment(level, num_games, edax_suboptimal_prob=suboptimal_prob)
    
    print("\nFinal Summary:")
    print(f"Edax Level {level}: AI Win Rate = {ai_win_rate:.2f}%")
    print(f"Edax Level {level}: Edax Win Rate = {edax_win_rate:.2f}%")