import tkinter as tk
import random
import math
import copy
import numpy as np
import pickle
import os

# Constants
RED = 1
YELLOW = 2
EMPTY = 0

# ConnectFour Game Class
class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = [[EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = RED

    def get_legal_moves(self, player):
        return [col for col in range(self.cols) if self.board[0][col] == EMPTY]

    def is_valid_move(self, col):
        return 0 <= col < self.cols and self.board[0][col] == EMPTY

    def make_move(self, col, player):
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = player
                break

    def check_win(self, player):
        # Check horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r][c + i] == player for i in range(4)):
                    return True
        # Check vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if all(self.board[r + i][c] == player for i in range(4)):
                    return True
        # Check diagonal (positive slope)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(self.board[r + i][c + i] == player for i in range(4)):
                    return True
        # Check diagonal (negative slope)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r - i][c + i] == player for i in range(4)):
                    return True
        return False

    def is_game_over(self):
        return (self.check_win(RED) or self.check_win(YELLOW) or
                not any(self.board[0][c] == EMPTY for c in range(self.cols)))

    def get_score(self):
        if self.check_win(RED):
            return 1, -1
        elif self.check_win(YELLOW):
            return -1, 1
        return 0, 0

# TD-FARL Agent with TCL
class TDAgent:
    def __init__(self):
        self.n_tuples = [
            [(r, c) for r in range(4) for c in range(4)],
            [(r, c) for r in range(4) for c in range(4)],
            [(r, c + r) for r in range(4) for c in range(4)],
            [(r, c - r) for r in range(4) for c in range(3, 7)]
        ]
        self.weights = [random.uniform(-1, 1) for _ in range(len(self.n_tuples))]
        self.Ni = [0.0] * len(self.weights)
        self.Ai = [0.0] * len(self.weights)
        self.alpha = 0.1
        self.lambda_ = 0.0
        self.eligibility = [0.0] * len(self.weights)

    def state_to_features(self, board, player):
        features = []
        for n_tuple in self.n_tuples:
            value = 0
            for r, c in n_tuple:
                if 0 <= r < 6 and 0 <= c < 7:
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
        for col in legal_moves:
            temp_game = ConnectFour()
            temp_game.board = [row[:] for row in board]
            temp_game.make_move(col, player)
            value = self.evaluate(temp_game.board, player)
            value = max(min(value, 50), -50)
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
        for col in legal_moves:
            temp_game = copy.deepcopy(game)
            temp_game.make_move(col, game.current_player)
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
            td_error = reward + 0.9 * next_value - value
        for i, f in enumerate(features):
            self.eligibility[i] = self.lambda_ * self.eligibility[i] + f
            alpha_i = 1.0 if self.Ai[i] == 0 else abs(self.Ni[i]) / self.Ai[i]
            delta_theta = self.alpha * alpha_i * td_error * self.eligibility[i]
            self.weights[i] += delta_theta
            self.Ni[i] += delta_theta
            self.Ai[i] += abs(delta_theta)

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
    def __init__(self, td_agent, iterations=1000):
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
            red_score, yellow_score = node.game.get_score()
            score = red_score if node.game.current_player == RED else yellow_score
            return -score
        if not node.expanded:
            legal_moves = node.game.get_legal_moves(node.game.current_player)
            if not legal_moves:
                return 0
            priors = self.td_agent.get_prior_probabilities(node.game.board, node.game.current_player, legal_moves)
            value = self.td_agent.evaluate(node.game.board, node.game.current_player)
            for move, prior in zip(legal_moves, priors):
                new_game = ConnectFour()
                new_game.board = [row[:] for row in node.game.board]
                new_game.current_player = node.game.current_player
                new_game.make_move(move, node.game.current_player)
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
        return best_child.move if best_child else random.choice(game.get_legal_moves(game.current_player))

# Training Function
def train_agent(num_episodes=6000):
    agent = TDAgent()
    epsilon_start, epsilon_end = 0.1, 0.0
    print("Starting training...")
    for episode in range(num_episodes):
        game = ConnectFour()
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / num_episodes
        state_history = {RED: [], YELLOW: []}
        while not game.is_game_over():
            player = game.current_player
            state = copy.deepcopy(game.board)
            move = agent.select_action(game, epsilon)
            if move is not None:
                game.make_move(move, player)
            next_state = copy.deepcopy(game.board)
            state_history[player].append((state, next_state, 0))
            game.current_player = 3 - player
        red_score, yellow_score = game.get_score()
        for player in [RED, YELLOW]:
            for i, (state, next_state, _) in enumerate(state_history[player][:-1]):
                agent.update_weights(state, next_state, 0, player)
            last_state, _, _ = state_history[player][-1] if state_history[player] else (game.board, game.board, 0)
            reward = red_score if player == RED else yellow_score
            agent.update_weights(last_state, last_state, reward, player, is_final=True)
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes} completed")
    print("Training completed.")
    return agent

# Function to load or train and save the model
def get_trained_agent(filename='ConnectFourModel.pkl', num_episodes=20000):
    if os.path.exists(filename):
        print(f"Loading existing model from {filename}...")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"No existing model found. Training new model...")
        trained_agent = train_agent(num_episodes=num_episodes)
        print(f"Saving trained model to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(trained_agent, f)
        return trained_agent

# Enhanced UI with Tkinter
class ConnectFourUI:
    def __init__(self, root, td_agent):
        self.root = root
        self.game = ConnectFour()
        self.td_agent = td_agent
        self.mcts = MCTS(self.td_agent, iterations=1000)
        self.cell_size = 60  # Initial cell size, will be updated
        self.min_cell_size = 40  # Minimum cell size
        self.max_cell_size = 100  # Maximum cell size

        # Configure root window
        self.root.configure(bg="#f5f0e1")
        self.root.minsize(400, 400)

        # Create main frame
        self.main_frame = tk.Frame(self.root, bg="#f5f0e1")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for game board
        self.canvas = tk.Canvas(self.main_frame, bg="#1e3d59", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status label
        self.status_var = tk.StringVar(value="Red's Turn")
        self.status_label = tk.Label(self.main_frame, textvariable=self.status_var, font=("Arial", 14), bg="#f5f0e1", fg="#1e3d59")
        self.status_label.pack(pady=5)

        # Button frame
        self.button_frame = tk.Frame(self.main_frame, bg="#f5f0e1")
        self.button_frame.pack(pady=5)

        # Restart button
        self.restart_button = tk.Button(self.button_frame, text="Restart Game", command=self.restart, font=("Arial", 12), bg="#ff6f61", fg="white", relief="flat", padx=10, pady=5)
        self.restart_button.pack()
        self.restart_button.bind("<Enter>", lambda e: self.restart_button.config(bg="#e65b51"))
        self.restart_button.bind("<Leave>", lambda e: self.restart_button.config(bg="#ff6f61"))

        # Title label
        self.title_label = tk.Label(self.main_frame, text="Connect Four", font=("Arial", 20, "bold"), bg="#f5f0e1", fg="#1e3d59")
        self.title_label.pack(pady=5)

        # Bind events
        self.canvas.bind("<Button-1>", self.human_move)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Configure>", self.on_resize)
        self.hovered_col = None

        # Initialize board
        self.draw_board()
        self.animate_arrows()

    def on_resize(self, event):
        """Handle window resize by updating cell size and redrawing board."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 1 or canvas_height < 1:
            return

        cell_width = canvas_width / self.game.cols
        cell_height = canvas_height / (self.game.rows + 1)
        self.cell_size = max(self.min_cell_size, min(self.max_cell_size, min(cell_width, cell_height)))

        status_font_size = max(10, int(self.cell_size / 5))
        title_font_size = max(14, int(self.cell_size / 3))
        button_font_size = max(10, int(self.cell_size / 6))

        self.status_label.config(font=("Arial", status_font_size))
        self.title_label.config(font=("Arial", title_font_size, "bold"))
        self.restart_button.config(font=("Arial", button_font_size))

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        board_width = self.cell_size * self.game.cols
        board_height = self.cell_size * (self.game.rows + 1)
        offset_x = (canvas_width - board_width) / 2 if canvas_width > board_width else 0
        offset_y = (canvas_height - board_height) / 2 if canvas_height > board_height else 0

        for r in range(self.game.rows + 1):
            y = offset_y + (r + 1) * self.cell_size
            self.canvas.create_line(offset_x, y, offset_x + self.game.cols * self.cell_size, y, fill="#2f4f4f", width=1)
        for c in range(self.game.cols + 1):
            x = offset_x + c * self.cell_size
            self.canvas.create_line(x, offset_y + self.cell_size, x, offset_y + (self.game.rows + 1) * self.cell_size, fill="#2f4f4f", width=1)

        for r in range(self.game.rows):
            for c in range(self.game.cols):
                x1 = offset_x + c * self.cell_size
                y1 = offset_y + (r + 1) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#4682b4", outline="#2f4f4f")
                if self.game.board[r][c] == RED:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="#ff4040", outline="#cc3333")
                elif self.game.board[r][c] == YELLOW:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="#ffd700", outline="#cca300")

        legal_moves = self.game.get_legal_moves(self.game.current_player)
        for col in legal_moves:
            x = offset_x + col * self.cell_size + self.cell_size // 2
            self.canvas.create_line(x, offset_y + self.cell_size // 2, x, offset_y + self.cell_size - 10, fill="yellow", width=4, tags=f"arrow_{col}")

    def animate_arrows(self):
        for col in self.game.get_legal_moves(self.game.current_player):
            self.canvas.itemconfig(f"arrow_{col}", fill="yellow" if random.random() > 0.5 else "white")
        if not self.game.is_game_over():
            self.root.after(500, self.animate_arrows)

    def on_mouse_move(self, event):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 1 or canvas_height < 1:
            return

        board_width = self.cell_size * self.game.cols
        board_height = self.cell_size * (self.game.rows + 1)
        offset_x = (canvas_width - board_width) / 2 if canvas_width > board_width else 0
        offset_y = (canvas_height - board_height) / 2 if canvas_height > board_height else 0

        col = int((event.x - offset_x) // self.cell_size)
        if self.game.is_valid_move(col) and col != self.hovered_col:
            if self.hovered_col is not None:
                self.canvas.delete(f"hover_{self.hovered_col}")
            x1 = offset_x + col * self.cell_size
            y1 = offset_y
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="", outline="#ffffff", width=2, tags=f"hover_{col}")
            self.hovered_col = col
        elif not self.game.is_valid_move(col):
            if self.hovered_col is not None:
                self.canvas.delete(f"hover_{self.hovered_col}")
                self.hovered_col = None

    def human_move(self, event):
        if self.game.current_player != RED or self.game.is_game_over():
            return
        canvas_width = self.canvas.winfo_width()
        board_width = self.cell_size * self.game.cols
        offset_x = (canvas_width - board_width) / 2 if canvas_width > board_width else 0
        col = int((event.x - offset_x) // self.cell_size)
        if self.game.is_valid_move(col):
            self.game.make_move(col, RED)
            self.game.current_player = YELLOW
            self.status_var.set("Yellow (AI)'s Turn")
            self.draw_board()
            if self.game.check_win(RED):
                self.show_winner("Red Wins!", "#ff4040")
            elif self.game.is_game_over():
                self.show_winner("It's a Draw!", "#4682b4")
            else:
                self.root.after(100, self.ai_move)

    def ai_move(self):
        if self.game.current_player != YELLOW or self.game.is_game_over():
            return
        move = self.mcts.get_best_move(self.game)
        if move is not None:
            self.game.make_move(move, YELLOW)
            self.draw_board()  # Redraw immediately after move
            if self.game.check_win(YELLOW):  # Check win condition right after move
                self.show_winner("Yellow (AI) Wins!", "#ffd700")
                return
            elif self.game.is_game_over():
                self.show_winner("It's a Draw!", "#4682b4")
                return
        self.game.current_player = RED
        self.status_var.set("Red's Turn")
        self.draw_board()
        if not self.game.is_game_over():
            self.root.after(100, self.ai_move)

    def show_winner(self, message, color):
        self.canvas.delete("game_over", "game_over_bg", "game_over_elements")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#000000", stipple="gray50", tags="game_over_bg")

        text_size = max(16, int(self.cell_size / 2))
        self.canvas.create_text(canvas_width / 2, canvas_height / 2 - self.cell_size, text=message, font=("Arial", text_size, "bold"), fill="white", tags="game_over_elements", anchor="center")

        button_width = max(100, int(self.cell_size * 2))
        button_height = max(40, int(self.cell_size / 2))
        button_x = canvas_width / 2 - button_width / 2
        button_y = canvas_height / 2 + self.cell_size / 2
        button = self.canvas.create_rectangle(button_x, button_y, button_x + button_width, button_y + button_height, fill="#4CAF50", outline="#388E3C", width=2, tags="game_over_elements")
        button_text = self.canvas.create_text(canvas_width / 2, button_y + button_height / 2, text="Play Again", font=("Arial", max(12, int(self.cell_size / 4))), fill="white", tags="game_over_elements", anchor="center")

        self.canvas.tag_bind("game_over_elements", "<Button-1>", lambda e: self.restart())
        self.canvas.tag_bind("game_over_elements", "<Enter>", lambda e: self.canvas.itemconfig(button, fill="#66BB6A"))
        self.canvas.tag_bind("game_over_elements", "<Leave>", lambda e: self.canvas.itemconfig(button, fill="#4CAF50"))

        self.canvas.itemconfig("game_over_elements", state="hidden")
        self.fade_in()

    def fade_in(self):
        current = self.canvas.itemcget("game_over_elements", "state")
        if current == "hidden":
            self.canvas.itemconfig("game_over_elements", state="normal")
            try:
                self.canvas.itemconfig("game_over_elements", alpha=0.0)
            except:
                pass
        else:
            try:
                alpha = float(self.canvas.itemcget("game_over_elements", "alpha"))
                if alpha < 1.0:
                    self.canvas.itemconfig("game_over_elements", alpha=min(1.0, alpha + 0.1))
                    self.root.after(50, self.fade_in)
            except:
                self.canvas.itemconfig("game_over_elements", state="normal")

    def restart(self):
        self.game = ConnectFour()
        self.hovered_col = None
        self.status_var.set("Red's Turn")
        self.canvas.delete("game_over", "game_over_bg", "game_over_elements")
        self.draw_board()

if __name__ == "__main__":
    trained_agent = get_trained_agent(filename='ConnectFourModel.pkl', num_episodes=20000)
    root = tk.Tk()
    root.title("ConnectFour with TD-FARL and MCTS")
    app = ConnectFourUI(root, trained_agent)
    root.mainloop()