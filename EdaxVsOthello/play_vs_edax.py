from othello_game import Othello, BLACK, WHITE
from td_agent import TDAgent
from edax_wrapper import EdaxWrapper

def board_to_fen(board, current_player):
    symbol = {0: '-', 1: '*', 2: 'O'}
    rows = [''.join(symbol[cell] for cell in row) for row in board]
    turn = '*' if current_player == BLACK else 'O'
    return '/'.join(rows) + f' {turn}'

def edax_move_to_coords(move):
    move = move.lower()
    col = ord(move[0]) - ord('a')
    row = int(move[1]) - 1
    return (row, col)

def print_board(board):
    print("  a b c d e f g h")
    for i, row in enumerate(board):
        print(i+1, end=' ')
        for cell in row:
            print(['.', 'B', 'W'][cell], end=' ')
        print()

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
        print(f"Legal moves for {['WHITE', 'BLACK'][game.current_player-1]}: {[(r+1, chr(c+97)) for r, c in legal_moves]}")
        if game.current_player == BLACK:
            move = agent.select_action(game)
            if move:
                game.make_move(*move, BLACK)
                print(f"\nAI (BLACK) plays: {chr(move[1]+97)}{move[0]+1}")
                consecutive_passes = 0
            else:
                print("\nAI (BLACK) passes")
                consecutive_passes += 1
        else:
            fen = board_to_fen(game.board, WHITE)
            move_str = edax.get_move(fen)
            if move_str != '0000':
                try:
                    row, col = edax_move_to_coords(move_str)
                    if game.is_valid_move(row, col, WHITE):
                        game.make_move(row, col, WHITE)
                        print(f"\nEdax (WHITE) plays: {move_str}")
                        consecutive_passes = 0
                    else:
                        print(f"\nEdax (WHITE) attempted invalid move: {move_str}")
                        consecutive_passes += 1
                except Exception as e:
                    print(f"\nError processing Edax move {move_str}: {e}")
                    consecutive_passes += 1
            else:
                print("\nEdax (WHITE) passes")
                consecutive_passes += 1
        print_board(game.board)
        game.current_player = 3 - game.current_player
        if consecutive_passes >= 2:
            print("Two consecutive passes detected")
            break

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