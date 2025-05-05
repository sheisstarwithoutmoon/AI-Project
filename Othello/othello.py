import tkinter as tk
import random
import math
import copy
import numpy as np
import pickle
import os

# Constants
BLACK = 1
WHITE = 2
EMPTY = 0
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Othello Game Class
class Othello:
    def __init__(self):
        self.board = [[EMPTY for _ in range(8)] for _ in range(8)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.current_player = BLACK

    def get_legal_moves(self, player):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == EMPTY and self.is_valid_move(i, j, player):
                    moves.append((i, j))
        return moves

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != EMPTY:
            return False
        opponent = 3 - player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if not (0 <= r < 8 and 0 <= c < 8) or self.board[r][c] != opponent:
                continue
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                r += dr
                c += dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player:
                return True
        return False

    def make_move(self, row, col, player):
        self.board[row][col] = player
        opponent = 3 - player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player:
                for fr, fc in to_flip:
                    self.board[fr][fc] = player

    def is_game_over(self):
        return not self.get_legal_moves(BLACK) and not self.get_legal_moves(WHITE)

    def get_score(self):
        black_count = sum(row.count(BLACK) for row in self.board)
        white_count = sum(row.count(WHITE) for row in self.board)
        return black_count, white_count

# TD-FARL Agent with TCL
class TDAgent:
    def __init__(self):
        self.n_tuples = [[(i, j) for j in range(4)] for i in range(4)]  # 4x4 n-tuples
        self.weights = [random.uniform(-1, 1) for _ in range(len(self.n_tuples))]
        # TCL parameters
        self.Ni = [0.0] * len(self.weights)  # Sum of weight changes
        self.Ai = [0.0] * len(self.weights)  # Sum of absolute weight changes
        self.alpha = 0.2  # Global learning rate (Othello from paper)
        self.lambda_ = 0.5  # Eligibility trace factor (Othello from paper)
        self.eligibility = [0.0] * len(self.weights)  # Eligibility traces

    def state_to_features(self, board, player):
        features = []
        for n_tuple in self.n_tuples:
            value = 0
            for r, c in n_tuple:
                if 0 <= r < 8 and 0 <= c < 8:
                    if board[r][c] == player:
                        value += 1
                    elif board[r][c] == 3 - player:
                        value -= 1
            features.append(value)
        return features

    def evaluate(self, board, player):
        features = self.state_to_features(board, player)
        return sum(w * f for w, f in zip(self.weights, features))

    def get_prior_probabilities(self, board, player, legal_moves):
        if not legal_moves:
            return []
        values = []
        for move in legal_moves:
            temp_game = Othello()
            temp_game.board = [row[:] for row in board]
            temp_game.make_move(move[0], move[1], player)
            value = self.evaluate(temp_game.board, player)
            values.append(value)
        exp_values = [math.exp(v) for v in values]
        total = sum(exp_values)
        if total == 0:
            return [1 / len(values)] * len(values)
        return [ev / total for ev in exp_values]

    def select_action(self, game, epsilon):
        legal_moves = game.get_legal_moves(game.current_player)
        if not legal_moves:
            return None
        if random.random() < epsilon:
            return random.choice(legal_moves)
        values = []
        for move in legal_moves:
            temp_game = copy.deepcopy(game)
            temp_game.make_move(move[0], move[1], game.current_player)
            value = self.evaluate(temp_game.board, temp_game.current_player)
            values.append(value)
        return legal_moves[np.argmax(values)]

    def update_weights(self, state, next_state, reward, player, is_final=False):
        features = self.state_to_features(state, player)
        value = self.evaluate(state, player)
        if is_final:
            td_error = reward - value
        else:
            next_value = self.evaluate(next_state, player)
            td_error = reward + 0.9 * next_value - value  # Gamma = 0.9
        # Update eligibility traces
        for i, f in enumerate(features):
            self.eligibility[i] = self.lambda_ * self.eligibility[i] + f
            # TCL: Adjust individual learning rate
            alpha_i = 1.0 if self.Ai[i] == 0 else abs(self.Ni[i]) / self.Ai[i]
            delta_theta = self.alpha * alpha_i * td_error * self.eligibility[i]
            self.weights[i] += delta_theta
            self.Ni[i] += delta_theta
            self.Ai[i] += abs(delta_theta)
            
    def save(self, filename="othello_agent.pkl", total_episodes=0):
        with open(filename, "wb") as f:
            pickle.dump({
                "weights": self.weights,
                "Ni": self.Ni,
                "Ai": self.Ai,
                "total_episodes": total_episodes
            }, f)
        print("Agent saved to", filename)

    def load(self, filename="othello_agent.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.weights = data["weights"]
                self.Ni = data["Ni"]
                self.Ai = data["Ai"]
                # Handle backward compatibility with older saved files
                total_episodes = data.get("total_episodes", 0)
            print(f"Pre-trained agent loaded from {filename} (Trained for {total_episodes} episodes)")
            return True, total_episodes
        else:
            print("No pre-trained agent found.")
            return False, 0

# MCTS Node
class MCTSNode:
    def __init__(self, game, move=None, parent=None):
        self.game = game
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = 0
        self.expanded = False

# MCTS Wrapper
class MCTS:
    def __init__(self, td_agent, iterations=100):
        self.td_agent = td_agent
        self.iterations = iterations
        self.cpuct = 1.0

    def select_child(self, node):
        if not node.expanded:
            return None, None
        scores = []
        for child in node.children:
            if child.visits == 0:
                score = float('inf')
            else:
                u = (self.cpuct * child.prior * math.sqrt(node.visits)) / (1 + child.visits)
                q = child.value / child.visits if child.visits > 0 else 0
                score = q + u
            scores.append(score)
        if not scores:
            return None, None
        idx = np.argmax(scores)
        return node.children[idx].move, node.children[idx]

    def mcts_iteration(self, node):
        if node.game.is_game_over():
            black_score, white_score = node.game.get_score()
            score = black_score - white_score if node.game.current_player == BLACK else white_score - black_score
            return -score

        if not node.expanded:
            legal_moves = node.game.get_legal_moves(node.game.current_player)
            if not legal_moves:
                new_game = Othello()
                new_game.board = [row[:] for row in node.game.board]
                new_game.current_player = 3 - node.game.current_player
                child = MCTSNode(new_game, move=None, parent=node)
                child.prior = 0
                node.children.append(child)
                node.expanded = True
                value = self.td_agent.evaluate(node.game.board, node.game.current_player)
                return -value

            priors = self.td_agent.get_prior_probabilities(node.game.board, node.game.current_player, legal_moves)
            value = self.td_agent.evaluate(node.game.board, node.game.current_player)
            for move, prior in zip(legal_moves, priors):
                new_game = Othello()
                new_game.board = [row[:] for row in node.game.board]
                new_game.current_player = node.game.current_player
                new_game.make_move(move[0], move[1], node.game.current_player)
                new_game.current_player = 3 - new_game.current_player
                child = MCTSNode(new_game, move=move, parent=node)
                child.prior = prior
                node.children.append(child)
            node.expanded = True
            return -value

        move, child = self.select_child(node)
        if child is None:
            return 0
        value = self.mcts_iteration(child)
        child.value += value
        child.visits += 1
        return -value

    def get_best_move(self, game):
        root = MCTSNode(copy.deepcopy(game))
        for _ in range(self.iterations):
            self.mcts_iteration(root)
        if not root.children:
            return None
        best_child = max(root.children, key=lambda c: c.visits, default=None)
        return best_child.move if best_child else None

# Training Function
def train_agent(num_episodes=250000, start_episode=0):
    agent = TDAgent()
    loaded, total_episodes = agent.load()  # Try to load existing agent
    
    # Start from where we left off
    current_total_episodes = total_episodes
    
    epsilon_start, epsilon_end = 0.2, 0.1  # Exploration rate decay
    print(f"Starting training from episode {start_episode}...")
    for episode in range(num_episodes):
        game = Othello()
        # Calculate epsilon based on total episodes trained
        relative_progress = (current_total_episodes + episode) / (current_total_episodes + num_episodes)
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * relative_progress
        
        state_history = {BLACK: [], WHITE: []}
        while not game.is_game_over():
            player = game.current_player
            state = copy.deepcopy(game.board)
            move = agent.select_action(game, epsilon)
            if move:
                game.make_move(move[0], move[1], player)
            next_state = copy.deepcopy(game.board)
            state_history[player].append((state, next_state, 0))  # Reward 0 during game
            game.current_player = 3 - player

        # Game over, assign rewards and update weights
        black_score, white_score = game.get_score()
        reward = 1 if black_score > white_score else -1 if white_score > black_score else 0
        for player in [BLACK, WHITE]:
            for i, (state, next_state, _) in enumerate(state_history[player][:-1]):
                agent.update_weights(state, next_state, 0, player)  # Mid-game updates
            # FARL: Final adaptation for the last state
            last_state, _, _ = state_history[player][-1] if state_history[player] else (game.board, game.board, 0)
            agent.update_weights(last_state, last_state, reward if player == BLACK else -reward, player, is_final=True)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")
            
        # Save periodically (every 100 episodes)
        if (episode + 1) % 100 == 0:
            new_total = current_total_episodes + episode + 1
            agent.save(total_episodes=new_total)
            
    # Final save with updated episode count
    new_total_episodes = current_total_episodes + num_episodes
    print(f"Training completed. Total episodes: {new_total_episodes}")
    agent.save(total_episodes=new_total_episodes)
    return agent

# UI with Tkinter
class OthelloUI:
    def __init__(self, root, td_agent):
        self.root = root
        self.game = Othello()
        self.td_agent = td_agent
        self.mcts = MCTS(self.td_agent, iterations=100)
        
        # Create main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = tk.Label(self.main_frame, text="Your turn (Black)", font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        # Game canvas
        self.canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.canvas.pack(pady=10)
        
        # Control buttons
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)
        
        self.new_game_btn = tk.Button(self.button_frame, text="New Game", command=self.new_game)
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        self.draw_board()
        self.canvas.bind("<Button-1>", self.human_move)

    def new_game(self):
        self.game = Othello()
        self.draw_board()
        self.status_label.config(text="Your turn (Black)")

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(8):
            for j in range(8):
                x1, y1 = j * 50, i * 50
                x2, y2 = x1 + 50, y1 + 50
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="green")
                if self.game.board[i][j] == BLACK:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="black")
                elif self.game.board[i][j] == WHITE:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="white")
        
        # Highlight legal moves for current player
        legal_moves = self.game.get_legal_moves(self.game.current_player)
        for r, c in legal_moves:
            x, y = c * 50 + 25, r * 50 + 25
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="red")
        
        # Display current score
        black_score, white_score = self.game.get_score()
        score_text = f"Black: {black_score}  White: {white_score}"
        self.canvas.create_text(200, 380, text=score_text, font=("Arial", 12), fill="white")

    def human_move(self, event):
        if self.game.current_player != BLACK or self.game.is_game_over():
            return
        col = event.x // 50
        row = event.y // 50
        if (row, col) in self.game.get_legal_moves(BLACK):
            self.game.make_move(row, col, BLACK)
            self.game.current_player = WHITE
            self.draw_board()
            self.status_label.config(text="AI thinking...")
            self.root.update()
            self.root.after(100, self.ai_move)

    def ai_move(self):
        if self.game.is_game_over():
            self.game_over()
            return
            
        if self.game.current_player != WHITE:
            return
            
        legal_moves = self.game.get_legal_moves(WHITE)
        if not legal_moves:
            self.game.current_player = BLACK
            self.draw_board()
            self.status_label.config(text="Your turn (Black)")
            
            # Check if game is now over
            if self.game.is_game_over():
                self.game_over()
            return
            
        move = self.mcts.get_best_move(self.game)
        if move:
            self.game.make_move(move[0], move[1], WHITE)
        self.game.current_player = BLACK
        self.draw_board()
        
        # Check if the game is over
        if self.game.is_game_over():
            self.game_over()
        else:
            # Check if human has legal moves
            if not self.game.get_legal_moves(BLACK):
                self.status_label.config(text="No legal moves - AI's turn")
                self.game.current_player = WHITE
                self.root.after(100, self.ai_move)
            else:
                self.status_label.config(text="Your turn (Black)")

    def game_over(self):
        black_score, white_score = self.game.get_score()
        if black_score > white_score:
            result = f"Game Over! You win! Black: {black_score}, White: {white_score}"
        elif white_score > black_score:
            result = f"Game Over! AI wins! Black: {black_score}, White: {white_score}"
        else:
            result = f"Game Over! It's a draw! Black: {black_score}, White: {white_score}"
        self.status_label.config(text=result)
        self.canvas.create_rectangle(50, 180, 350, 220, fill="blue")
        self.canvas.create_text(200, 200, text=result, font=("Arial", 14), fill="yellow")

def main():
    # Create a simple menu window first
    menu_window = tk.Tk()
    menu_window.title("Othello Game")
    menu_window.geometry("350x200")
    
    # Center the window
    menu_window.eval('tk::PlaceWindow . center')
    
    # Create and initialize the agent
    agent = TDAgent()
    agent_exists, total_episodes = agent.load()  # Try to load existing agent
    
    def start_game():
        menu_window.destroy()
        game_window = tk.Tk()
        game_window.title("Othello Game")
        app = OthelloUI(game_window, agent)
        game_window.mainloop()
    
    def start_training():
        # Fixed number of episodes
        episodes = 250000 # You can adjust this value
            
        train_btn.config(state=tk.DISABLED, text="Training in progress...")
        menu_window.update()
        
        # Train with specified episodes
        train_agent(num_episodes=episodes)
        
        # Reload the trained agent
        agent_exists, new_total = agent.load()
        
        # Automatically start the game after training
        start_game()
    
    # Create menu components
    title_label = tk.Label(menu_window, text="Othello with TD-FARL and MCTS", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)
    
    status_text = f"Agent {'loaded successfully' if agent_exists else 'not found'}"
    status_label = tk.Label(menu_window, text=status_text)
    status_label.pack(pady=5)
    
    # Display total episodes trained (if agent exists)
    if agent_exists:
        episodes_trained_label = tk.Label(menu_window, text=f"Total episodes trained: {total_episodes}")
        episodes_trained_label.pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(menu_window)
    button_frame.pack(pady=20)
    
    if agent_exists:
        play_btn = tk.Button(button_frame, text="Play Game", command=start_game)
        play_btn.pack(side=tk.LEFT, padx=10)
        
        train_btn = tk.Button(button_frame, text="Train More & Play", command=start_training)
        train_btn.pack(side=tk.LEFT, padx=10)
    else:
        # If no agent exists, only show train button
        train_btn = tk.Button(button_frame, text="Train Agent & Play", command=start_training, width=20)
        train_btn.pack(padx=10)
    
    menu_window.mainloop()

if __name__ == "__main__":
    main()