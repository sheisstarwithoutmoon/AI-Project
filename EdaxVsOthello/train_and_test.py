import argparse
import time
from othello_game import Othello, BLACK, WHITE
from td_agent_improved import TDAgent
from edax_wrapper import EdaxWrapper

def train_agent(num_episodes, resume=False):
    """Train the TD agent for the specified number of episodes"""
    agent = TDAgent(
        learning_rate=0.2,
        lambda_trace=0.5,
        epsilon_start=0.2,
        epsilon_end=0.1,
        use_tcl=True
    )
    
    model_file = "td_agent_trained.pkl"
    if resume:
        agent.load(model_file)
    
    start_time = time.time()
    agent.train(num_episodes=num_episodes, save_interval=10000, save_path=model_file)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print(f"Win rate: {agent.wins/num_episodes*100:.1f}%")
    print(f"Loss rate: {agent.losses/num_episodes*100:.1f}%")
    print(f"Draw rate: {agent.draws/num_episodes*100:.1f}%")

def evaluate_against_random(num_games=100):
    """Evaluate the trained agent against a random player"""
    agent = TDAgent()
    if not agent.load("td_agent_trained.pkl"):
        print("No trained model found. Please train the agent first.")
        return
    
    print(f"Evaluating against random player for {num_games} games...")
    wins, losses, draws = 0, 0, 0
    
    for i in range(num_games):
        game = Othello()
        while not game.is_game_over():
            legal_moves = game.get_legal_moves(game.current_player)
            if not legal_moves:
                game.current_player = 3 - game.current_player
                continue
                
            if game.current_player == BLACK:
                # Agent plays as BLACK
                move = agent.select_action(game, epsilon=0)  # No exploration during evaluation
                if move:
                    game.make_move(move[0], move[1], BLACK)
            else:
                # Random player as WHITE
                import random
                move = random.choice(legal_moves)
                game.make_move(move[0], move[1], WHITE)
            
            game.current_player = 3 - game.current_player
        
        b, w = game.get_score()
        if b > w:
            wins += 1
        elif w > b:
            losses += 1
        else:
            draws += 1
        
        if (i+1) % 10 == 0:
            print(f"Progress: {i+1}/{num_games} games")
    
    print("\nEvaluation Results:")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")

def play_against_edax(num_games=10, edax_depth=5):
    """Test the trained agent against Edax at specified depth"""
    agent = TDAgent()
    if not agent.load("td_agent_trained.pkl"):
        print("No trained model found. Please train the agent first.")
        return
    
    try:
        edax = EdaxWrapper(depth=edax_depth)
    except RuntimeError as e:
        print(f"Failed to initialize Edax: {e}")
        return
    
    print(f"Playing against Edax (depth {edax_depth}) for {num_games} games...")
    wins, losses, draws = 0, 0, 0
    
    from play_vs_edax_fix import board_to_fen, edax_move_to_coords, print_board, coords_to_move_str
    
    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        game = Othello()
        print_board(game.board)
        
        while not game.is_game_over():
            legal_moves = game.get_legal_moves(game.current_player)
            player_name = "BLACK" if game.current_player == BLACK else "WHITE"
            
            if not legal_moves:
                print(f"{player_name} has no legal moves and must pass.")
                game.current_player = 3 - game.current_player
                continue
            
            if game.current_player == BLACK:
                # Agent plays as BLACK
                move = agent.select_action(game, epsilon=0)  # No exploration during evaluation
                if move:
                    game.make_move(move[0], move[1], BLACK)
                    print(f"Agent (BLACK) plays: {coords_to_move_str(move[0], move[1])}")
            else:
                # Edax plays as WHITE
                fen = board_to_fen(game.board, WHITE)
                move_str = edax.get_move(fen)
                
                if move_str != '0000' and len(move_str) >= 2:
                    try:
                        row, col = edax_move_to_coords(move_str)
                        if game.is_valid_move(row, col, WHITE):
                            game.make_move(row, col, WHITE)
                            print(f"Edax (WHITE) plays: {move_str}")
                        else:
                            print(f"Edax attempted invalid move: {move_str}")
                            # Choose first legal move as fallback
                            row, col = legal_moves[0]
                            game.make_move(row, col, WHITE)
                            print(f"Edax (WHITE) plays: {coords_to_move_str(row, col)} (fallback)")
                    except Exception as e:
                        print(f"Error processing Edax move: {e}")
                        # Choose first legal move as fallback
                        row, col = legal_moves[0]
                        game.make_move(row, col, WHITE)
                        print(f"Edax (WHITE) plays: {coords_to_move_str(row, col)} (fallback)")
                else:
                    print("Edax failed to provide a valid move.")
                    # Choose first legal move as fallback
                    row, col = legal_moves[0]
                    game.make_move(row, col, WHITE)
                    print(f"Edax (WHITE) plays: {coords_to_move_str(row, col)} (fallback)")
            
            game.current_player = 3 - game.current_player
            print_board(game.board)
        
        b, w = game.get_score()
        print(f"Game {game_num + 1} result: BLACK (Agent) {b} - {w} WHITE (Edax)")
        
        if b > w:
            wins += 1
            print("Agent WINS!")
        elif w > b:
            losses += 1
            print("Edax WINS!")
        else:
            draws += 1
            print("DRAW!")
    
    edax.close()
    
    print("\nOverall Results vs Edax:")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='TD Learning for Othello')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the TD agent')
    train_parser.add_argument('--episodes', type=int, default=250000, help='Number of episodes to train')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from saved model')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the agent against random player')
    eval_parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    
    # Play against Edax command
    edax_parser = subparsers.add_parser('edax', help='Play against Edax')
    edax_parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    edax_parser.add_argument('--depth', type=int, default=5, help='Edax search depth')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_agent(args.episodes, args.resume)
    elif args.command == 'evaluate':
        evaluate_against_random(args.games)
    elif args.command == 'edax':
        play_against_edax(args.games, args.depth)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()