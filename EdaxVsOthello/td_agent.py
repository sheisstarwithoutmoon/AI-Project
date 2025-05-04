import random
import numpy as np
import copy
import pickle
import os
from othello_game import Othello, BLACK, WHITE

class TDAgent:
    def __init__(self, 
                 learning_rate=0.2, 
                 lambda_trace=0.5, 
                 epsilon_start=0.2, 
                 epsilon_end=0.1,
                 discount_factor=0.99,
                 use_tcl=True):
        """
        Initialize a TD-learning agent for Othello
        
        Parameters:
        - learning_rate: Alpha parameter for TD updates
        - lambda_trace: Lambda for eligibility traces
        - epsilon_start: Starting exploration rate
        - epsilon_end: Ending exploration rate
        - discount_factor: Gamma for future rewards
        - use_tcl: Whether to use Temporal Coherence Learning
        """
        # Learning parameters
        self.alpha = learning_rate
        self.lambda_trace = lambda_trace
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.gamma = discount_factor
        self.use_tcl = use_tcl
        
        # N-tuple configuration
        # Use more sophisticated n-tuples for better learning
        self.setup_ntuples()
        
        # Model parameters
        self.weights = {}
        self.eligibility = {}
        self.tcl_step_sizes = {}  # For TCL
        self.tcl_gradients = {}   # For TCL
        
        # Initialize weights and traces
        self.initialize_parameters()
        
        # Training stats
        self.episodes_trained = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def setup_ntuples(self):
        """Define n-tuple configurations for feature extraction"""
        self.n_tuples = []
        
        # Basic 2x2 squares throughout the board
        for i in range(7):
            for j in range(7):
                self.n_tuples.append([(i, j), (i, j+1), (i+1, j), (i+1, j+1)])
        
        # Horizontal lines
        for i in range(8):
            self.n_tuples.append([(i, j) for j in range(8)])
        
        # Vertical lines
        for j in range(8):
            self.n_tuples.append([(i, j) for i in range(8)])
        
        # Diagonals
        self.n_tuples.append([(i, i) for i in range(8)])
        self.n_tuples.append([(i, 7-i) for i in range(8)])
        
        # Edge configurations (important in Othello)
        for i in range(8):
            self.n_tuples.append([(0, i), (1, i)])  # Top edge
            self.n_tuples.append([(7, i), (6, i)])  # Bottom edge
            self.n_tuples.append([(i, 0), (i, 1)])  # Left edge
            self.n_tuples.append([(i, 7), (i, 6)])  # Right edge
            
        # Corner configurations (very important in Othello)
        self.n_tuples.append([(0, 0), (0, 1), (1, 0), (1, 1)])  # Top-left
        self.n_tuples.append([(0, 7), (0, 6), (1, 7), (1, 6)])  # Top-right
        self.n_tuples.append([(7, 0), (7, 1), (6, 0), (6, 1)])  # Bottom-left
        self.n_tuples.append([(7, 7), (7, 6), (6, 7), (6, 6)])  # Bottom-right
    
    def initialize_parameters(self):
        """Initialize weights, eligibility traces, and TCL parameters"""
        # Initialize weights with small random values
        for i, n_tuple in enumerate(self.n_tuples):
            tuple_key = f"tuple_{i}"
            # 3^len(n_tuple) possible states for each n-tuple (0, 1, 2 for empty, black, white)
            n_states = 3 ** len(n_tuple)
            self.weights[tuple_key] = np.random.uniform(-0.1, 0.1, n_states)
            self.eligibility[tuple_key] = np.zeros(n_states)
            
            if self.use_tcl:
                self.tcl_step_sizes[tuple_key] = np.ones(n_states) * 0.1  # Initial step sizes
                self.tcl_gradients[tuple_key] = np.zeros(n_states)
    
    def get_tuple_index(self, board, n_tuple):
        """Convert a tuple's board positions to a single index"""
        base = 1
        index = 0
        for r, c in n_tuple:
            index += board[r][c] * base
            base *= 3
        return index
    
    def get_features(self, board):
        """Extract features from the board using n-tuples"""
        features = {}
        for i, n_tuple in enumerate(self.n_tuples):
            tuple_key = f"tuple_{i}"
            features[tuple_key] = self.get_tuple_index(board, n_tuple)
        return features
    
    def evaluate(self, board, player):
        """Evaluate board position using n-tuple weights"""
        features = self.get_features(board)
        value = 0
        for tuple_key, index in features.items():
            value += self.weights[tuple_key][index]
        
        # Scale the value using sigmoid to keep it between -1 and 1
        return 2.0 / (1.0 + np.exp(-value)) - 1.0
    
    def select_action(self, game, epsilon=None):
        """Select an action using epsilon-greedy policy"""
        if epsilon is None:
            # Calculate current epsilon based on training progress
            progress = min(1.0, self.episodes_trained / 250000)
            epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
        
        moves = game.get_legal_moves(game.current_player)
        if not moves:
            return None
            
        if random.random() < epsilon:
            return random.choice(moves)
            
        best_value = float('-inf')
        best_move = None
        
        for move in moves:
            new_game = copy.deepcopy(game)
            new_game.make_move(move[0], move[1], game.current_player)
            value = self.evaluate(new_game.board, game.current_player)
            if value > best_value:
                best_value = value
                best_move = move
                
        return best_move
    
    def update_weights(self, old_features, new_features, td_error):
        """Update weights using TD error and eligibility traces"""
        for tuple_key in self.weights:
            old_idx = old_features[tuple_key]
            
            # Update eligibility traces
            self.eligibility[tuple_key] *= self.lambda_trace
            self.eligibility[tuple_key][old_idx] += 1
            
            if self.use_tcl:
                # Temporal Coherence Learning update
                self.tcl_gradients[tuple_key][old_idx] = 0.95 * self.tcl_gradients[tuple_key][old_idx] + \
                                                       0.05 * td_error
                
                # Update step sizes
                if self.tcl_gradients[tuple_key][old_idx] * td_error > 0:
                    self.tcl_step_sizes[tuple_key][old_idx] *= 1.1
                else:
                    self.tcl_step_sizes[tuple_key][old_idx] *= 0.9
                
                # Ensure step sizes remain in reasonable range
                self.tcl_step_sizes[tuple_key][old_idx] = min(1.0, 
                                                           max(0.01, self.tcl_step_sizes[tuple_key][old_idx]))
                
                # Use TCL step sizes to update weights
                self.weights[tuple_key] += self.tcl_step_sizes[tuple_key] * \
                                        td_error * self.eligibility[tuple_key]
            else:
                # Standard TD update
                self.weights[tuple_key] += self.alpha * td_error * self.eligibility[tuple_key]
    
    def train_episode(self, game=None):
        """Train the agent for one episode"""
        if game is None:
            game = Othello()
        
        states = []
        features = []
        player = BLACK  # Agent plays as BLACK
        
        while not game.is_game_over():
            # Save current state
            board_copy = copy.deepcopy(game.board)
            states.append(board_copy)
            current_features = self.get_features(board_copy)
            features.append(current_features)
            
            # Get legal moves
            legal_moves = game.get_legal_moves(game.current_player)
            
            if not legal_moves:
                # Player must pass
                game.current_player = 3 - game.current_player
                continue
                
            if game.current_player == player:
                # Agent's turn
                move = self.select_action(game)
                if move:
                    game.make_move(move[0], move[1], player)
            else:
                # Opponent's turn - random policy
                move = random.choice(legal_moves)
                game.make_move(move[0], move[1], game.current_player)
            
            game.current_player = 3 - game.current_player
        
        # Game over - determine final reward
        black_score, white_score = game.get_score()
        if player == BLACK:
            outcome = 1 if black_score > white_score else (-1 if white_score > black_score else 0)
        else:
            outcome = 1 if white_score > black_score else (-1 if black_score > white_score else 0)
        
        # Update game statistics
        if outcome == 1:
            self.wins += 1
        elif outcome == -1:
            self.losses += 1
        else:
            self.draws += 1
        
        # TD updates
        if len(states) > 1:
            for i in range(len(states) - 1, 0, -1):
                current_val = self.evaluate(states[i], player)
                
                if i == len(states) - 1:
                    # Terminal state - use actual outcome
                    td_error = outcome - current_val
                else:
                    next_val = self.evaluate(states[i+1], player)
                    td_error = self.gamma * next_val - current_val
                
                self.update_weights(features[i], features[i-1], td_error)
        
        self.episodes_trained += 1
        
        # Reset eligibility traces for next episode
        for tuple_key in self.eligibility:
            self.eligibility[tuple_key].fill(0)
        
        return outcome
    
    def train(self, num_episodes=250000, save_interval=10000, save_path="td_agent.pkl"):
        """Train the agent for a specified number of episodes"""
        print(f"Starting training for {num_episodes} episodes")
        print(f"Parameters: α={self.alpha}, λ={self.lambda_trace}, ε={self.epsilon_start}->{self.epsilon_end}, TCL={self.use_tcl}")
        
        for episode in range(1, num_episodes + 1):
            outcome = self.train_episode()
            
            # Print progress
            if episode % 1000 == 0:
                win_rate = self.wins / episode * 100
                loss_rate = self.losses / episode * 100
                draw_rate = self.draws / episode * 100
                
                progress = min(1.0, episode / num_episodes)
                current_epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
                
                print(f"Episode {episode}/{num_episodes} - Win: {win_rate:.1f}%, Loss: {loss_rate:.1f}%, Draw: {draw_rate:.1f}%, ε: {current_epsilon:.3f}")
            
            # Save checkpoints
            if episode % save_interval == 0:
                checkpoint_path = f"{save_path.split('.')[0]}_{episode}.pkl"
                self.save(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save final model
        self.save(save_path)
        print(f"Training complete. Final model saved to {save_path}")
        print(f"Results - Win: {self.wins}, Loss: {self.losses}, Draw: {self.draws}")
    
    def save(self, filename):
        """Save the agent's parameters to a file"""
        state = {
            'weights': self.weights,
            'tcl_step_sizes': self.tcl_step_sizes,
            'episodes_trained': self.episodes_trained,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'alpha': self.alpha,
            'lambda_trace': self.lambda_trace,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'use_tcl': self.use_tcl
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filename):
        """Load agent parameters from a file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            self.weights = state['weights']
            if 'tcl_step_sizes' in state:
                self.tcl_step_sizes = state['tcl_step_sizes']
            if 'episodes_trained' in state:
                self.episodes_trained = state['episodes_trained']
            if 'wins' in state:
                self.wins = state['wins']
            if 'losses' in state:
                self.losses = state['losses']
            if 'draws' in state:
                self.draws = state['draws']
            
            print(f"Loaded model from {filename}")
            print(f"Episodes trained: {self.episodes_trained}")
            if self.episodes_trained > 0:
                win_rate = self.wins / self.episodes_trained * 100
                print(f"Win rate: {win_rate:.1f}%")
            return True
        else:
            print(f"Model file {filename} not found")
            return False


# For training the agent
if __name__ == "__main__":
    agent = TDAgent(
        learning_rate=0.2,
        lambda_trace=0.5,
        epsilon_start=0.2,
        epsilon_end=0.1,
        use_tcl=True
    )
    
    # Try to load existing model first
    model_file = "td_agent_trained.pkl"
    if not agent.load(model_file):
        print("Training new model...")
        agent.train(num_episodes=250000, save_path=model_file)