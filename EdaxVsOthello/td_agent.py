# td_agent.py

import random
import numpy as np
import copy
from othello_game import Othello, BLACK, WHITE

class TDAgent:
    def __init__(self):
        self.n_tuples = [[(i, j) for j in range(4)] for i in range(4)]
        self.weights = [random.uniform(-1, 1) for _ in self.n_tuples]

    def evaluate(self, board, player):
        value = 0
        for idx, n_tuple in enumerate(self.n_tuples):
            count = sum(1 if board[r][c] == player else -1 if board[r][c] == 3 - player else 0
                        for r, c in n_tuple)
            value += self.weights[idx] * count
        return value

    def select_action(self, game, epsilon=0):
        moves = game.get_legal_moves(game.current_player)
        if not moves:
            return None
        if random.random() < epsilon:
            return random.choice(moves)
        values = []
        for move in moves:
            new_game = copy.deepcopy(game)
            new_game.make_move(move[0], move[1], game.current_player)
            values.append(self.evaluate(new_game.board, game.current_player))
        return moves[np.argmax(values)]
