import tkinter as tk
import random
import math
import copy
import numpy as np

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
def train_agent(num_episodes=250000):
    agent = TDAgent()
    epsilon_start, epsilon_end = 0.2, 0.1  # Exploration rate decay
    print("Starting training...")
    for episode in range(num_episodes):
        game = Othello()
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / num_episodes
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

        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} completed")
    print("Training completed.")
    return agent

# UI with Tkinter
class OthelloUI:
    def __init__(self, root, td_agent):
        self.root = root
        self.game = Othello()
        self.td_agent = td_agent
        self.mcts = MCTS(self.td_agent, iterations=100)
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        self.draw_board()
        self.canvas.bind("<Button-1>", self.human_move)

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
        legal_moves = self.game.get_legal_moves(self.game.current_player)
        for r, c in legal_moves:
            x, y = c * 50 + 25, r * 50 + 25
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="red")

    def human_move(self, event):
        if self.game.current_player != BLACK or self.game.is_game_over():
            return
        col = event.x // 50
        row = event.y // 50
        if (row, col) in self.game.get_legal_moves(BLACK):
            self.game.make_move(row, col, BLACK)
            self.game.current_player = WHITE
            self.draw_board()
            self.root.after(100, self.ai_move)

    def ai_move(self):
        if self.game.current_player != WHITE or self.game.is_game_over():
            return
        move = self.mcts.get_best_move(self.game)
        if move:
            self.game.make_move(move[0], move[1], WHITE)
        self.game.current_player = BLACK
        self.draw_board()
        if self.game.is_game_over():
            black_score, white_score = self.game.get_score()
            result = f"Game Over! Black: {black_score}, White: {white_score}"
            self.canvas.create_text(200, 200, text=result, font=("Arial", 20), fill="yellow")
        else:
            self.root.after(100, self.ai_move)

if __name__ == "__main__":
    # Train the agent first
    trained_agent = train_agent(num_episodes=200)  # Matches paper's Othello setting 250000
    # Start the game UI with the trained agent
    root = tk.Tk()
    root.title("Othello with Trained TD-FARL and MCTS")
    app = OthelloUI(root, trained_agent)
    root.mainloop()