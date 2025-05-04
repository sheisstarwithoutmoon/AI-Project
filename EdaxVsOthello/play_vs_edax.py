from othello_game import Othello, BLACK, WHITE
from td_agent import TDAgent
from edax_wrapper import EdaxWrapper

def board_to_fen(board, current_player):
    """Convert the game board to Forsyth–Edwards Notation for Edax"""
    symbol = {0: '-', 1: '*', 2: 'O'}
    rows = [''.join(symbol[cell] for cell in row) for row in board]
    turn = 'O' if current_player == WHITE else '*'  # Fixed player symbol assignment
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

def play_game():
    game = Othello()
    agent = TDAgent()
    try:
        edax = EdaxWrapper()
    except RuntimeError as e:
        print(f"Failed to initialize Edax: {e}")
        return

    print("Starting match: AI (BLACK) vs Edax (WHITE)")
    print_board(game.board)

    consecutive_passes = 0
    while not game.is_game_over():
        legal_moves = game.get_legal_moves(game.current_player)
        player_name = "BLACK" if game.current_player == BLACK else "WHITE"
        
        # Convert to human-readable format for display
        readable_moves = [(coords_to_move_str(r, c)) for r, c in legal_moves]
        print(f"Legal moves for {player_name}: {readable_moves}")
        
        if not legal_moves:
            print(f"\n{player_name} has no legal moves and must pass.")
            consecutive_passes += 1
            if consecutive_passes >= 2:
                print("Two consecutive passes detected - game over.")
                break
            game.current_player = 3 - game.current_player
            continue
        
        consecutive_passes = 0  # Reset consecutive passes counter
        
        if game.current_player == BLACK:
            # AI (Black) player's turn
            move = agent.select_action(game)
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
                    if game.is_valid_move(row, col, WHITE):
                        game.make_move(row, col, WHITE)
                        print(f"\nEdax (WHITE) plays: {move_str}")
                    else:
                        print(f"\nEdax (WHITE) attempted invalid move: {move_str}")
                        print(f"Choosing a random valid move instead.")
                        if legal_moves:
                            random_move = legal_moves[0]  # Take first legal move
                            row, col = random_move
                            game.make_move(row, col, WHITE)
                            print(f"Edax (WHITE) plays: {coords_to_move_str(row, col)}")
                        else:
                            print("\nEdax (WHITE) passes")
                            consecutive_passes += 1
                except Exception as e:
                    print(f"\nError processing Edax move {move_str}: {e}")
                    if legal_moves:
                        random_move = legal_moves[0]  # Take first legal move
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
        game.current_player = 3 - game.current_player

    edax.close()
    b, w = game.get_score()
    print("\nGame Over!")
    print(f"Final Score -> BLACK: {b}, WHITE: {w}")
    if b > w:
        print("AI WINS!")
    elif w > b:
        print("EDAX WINS!")
    else:
        print("DRAW!")

if __name__ == "__main__":
    play_game()