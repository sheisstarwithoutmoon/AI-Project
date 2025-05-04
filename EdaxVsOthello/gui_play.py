import tkinter as tk
from tkinter import messagebox
from othello_game import Othello, BLACK, WHITE

class OthelloUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Othello (Reversi)")
        
        self.game = Othello()
        self.board_buttons = [[None for _ in range(8)] for _ in range(8)]
        
        self.create_board()
        
    def create_board(self):
        for row in range(8):
            for col in range(8):
                btn = tk.Button(self.root, text='', width=4, height=2, command=lambda r=row, c=col: self.on_click(r, c))
                btn.grid(row=row, column=col)
                self.board_buttons[row][col] = btn
        self.update_board()

    def update_board(self):
        for row in range(8):
            for col in range(8):
                cell = self.game.board[row][col]
                if cell == BLACK:
                    self.board_buttons[row][col].config(text='B', bg='black', fg='white', state='normal')
                elif cell == WHITE:
                    self.board_buttons[row][col].config(text='W', bg='white', fg='black', state='normal')
                else:
                    self.board_buttons[row][col].config(text='', bg='green', fg='black', state='normal')
        
        # Highlight valid moves
        self.highlight_valid_moves()

        if self.game.is_game_over():
            self.show_game_over()

    def highlight_valid_moves(self):
        valid_moves = self.game.get_legal_moves(self.game.current_player)
        for row in range(8):
            for col in range(8):
                if (row, col) in valid_moves:
                    self.board_buttons[row][col].config(bg='yellow')

    def on_click(self, row, col):
        if self.game.is_valid_move(row, col, self.game.current_player):
            self.game.make_move(row, col, self.game.current_player)
            self.update_board()
            if self.game.is_game_over():
                self.show_game_over()
            else:
                self.game.current_player = 3 - self.game.current_player  # Switch player

    def show_game_over(self):
        b, w = self.game.get_score()
        winner = "DRAW"
        if b > w:
            winner = "BLACK WINS"
        elif w > b:
            winner = "WHITE WINS"
        
        messagebox.showinfo("Game Over", f"Final Score: Black: {b} - White: {w}\n{winner}")
        self.root.quit()  # Close the game window

def main():
    root = tk.Tk()
    app = OthelloUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
