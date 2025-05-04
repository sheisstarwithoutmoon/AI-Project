BLACK = 1
WHITE = 2

class Othello:
    def __init__(self):
        self.board = [[0]*8 for _ in range(8)]
        self.board[3][3] = self.board[4][4] = WHITE
        self.board[3][4] = self.board[4][3] = BLACK
        self.current_player = BLACK

    def get_legal_moves(self, player):
        moves = []
        for r in range(8):
            for c in range(8):
                if self.is_valid_move(r, c, player):
                    moves.append((r, c))
        return moves

    def is_valid_move(self, row, col, player):
        if row < 0 or row >= 8 or col < 0 or col >= 8 or self.board[row][col] != 0:
            return False
        opponent = 3 - player
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opponent = False
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                found_opponent = True
                r += dr
                c += dc
                while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                    r += dr
                    c += dc
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player and found_opponent:
                    return True
        return False

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            return False
        self.board[row][col] = player
        opponent = 3 - player
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        flipped = False
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc
                while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                    to_flip.append((r, c))
                    r += dr
                    c += dc
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player:
                    for fr, fc in to_flip:
                        self.board[fr][fc] = player
                    flipped = True
        return flipped

    def is_game_over(self):
        black_moves = self.get_legal_moves(BLACK)
        white_moves = self.get_legal_moves(WHITE)
        return not black_moves and not white_moves

    def get_score(self):
        black = sum(row.count(BLACK) for row in self.board)
        white = sum(row.count(WHITE) for row in self.board)
        return black, white