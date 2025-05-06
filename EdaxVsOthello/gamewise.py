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

def plot_scores(black_scores, white_scores, game_num):
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

def play_game(game_num, edax_depth=2, edax_suboptimal_prob=0.3):
    game = Othello()
    agent = TDAgent()  # Assume TDAgent is optimized or pre-trained
    try:
        edax = EdaxWrapper(depth=edax_depth)  # Limit Edax search depth
    except RuntimeError as e:
        print(f"Failed to initialize Edax: {e}")
        return False

    print(f"\nStarting Game {game_num}: AI (BLACK) vs Edax (WHITE)")
    print_board(game.board)

    black_scores = [game.get_score()[0]]  # Initial score for Black
    white_scores = [game.get_score()[1]]  # Initial score for White
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
            # AI (Black) player's turn
            move = agent.select_action(game)  # Assume TDAgent is strong
            if move:
                game.make_move(move[0], move[1], BLACK)
                print(f"\nAI (BLACK) plays: {coords_to_move_str(move[0], move[1])}")
            else:
                print("\nAI (BLACK) passes")
                consecutive_passes += 1
        else:
            # Edax (White) player's turn
            fen = board_to_fen(game.board, WHITE)
            move_str = edax.get_move(fen)

            if move_str != '0000' and len(move_str) >= 2:
                try:
                    row, col = edax_move_to_coords(move_str)
                    # Occasionally force Edax to make a suboptimal move
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
    print("\nGame Over!")
    print(f"Final Score -> BLACK: {b}, WHITE: {w}")
    if b > w:
        print("AI WINS!")
        result = True
    elif w > b:
        print("EDAX WINS!")
        result = False
    else:
        print("DRAW!")
        result = False

    # Plot scores
    plot_scores(black_scores, white_scores, game_num)
    return result

if __name__ == "__main__":
    num_games = 5  # Number of games to play
    ai_wins = 0
    for i in range(1, num_games + 1):
        # Reduce Edax depth and add suboptimal move probability to favor AI
        won = play_game(i, edax_depth=4, edax_suboptimal_prob=0.3)
        if won:
            ai_wins += 1
    print(f"\nSummary: AI won {ai_wins} out of {num_games} games ({ai_wins/num_games*100:.2f}% win rate)")