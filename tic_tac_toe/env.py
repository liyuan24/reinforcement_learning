import numpy as np


class TicTacToeEnvironment:
    def __init__(self, current_player=1):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = current_player  # 1 for X, -1 for O

    def reset(self, current_player=1):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = current_player
        return self.get_state()

    def get_state(self):
        """Convert board to hashable state representation"""
        return tuple(self.board.flatten())

    def get_available_actions(self):
        """Get list of available moves (row, col)"""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions

    def make_move(self, action):
        """Make a move and return (next_state, reward, done)"""
        if self.board[action[0], action[1]] != 0:
            raise ValueError(
                f"Invalid move: position ({action[0]}, {action[1]}) is already occupied"
            )

        self.board[action[0], action[1]] = self.current_player

        # Check for win
        winner = self.check_winner()
        if winner is not None:
            reward = 1 if winner == self.current_player else -1
            return self.get_state(), reward, True

        # Check for draw
        if len(self.get_available_actions()) == 0:
            return self.get_state(), 0, True

        # Switch player
        self.current_player *= -1
        return self.get_state(), 0, False

    def check_winner(self):
        """Check if there's a winner. Returns 1, -1, or None"""
        # Check rows
        for row in self.board:
            if abs(sum(row)) == 3:
                return row[0]

        # Check columns
        for col in range(3):
            if abs(sum(self.board[:, col])) == 3:
                return self.board[0, col]

        # Check diagonals
        if abs(sum(self.board.diagonal())) == 3:
            return self.board[0, 0]
        if abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return self.board[0, 2]

        return None

    def print_board(self):
        """Print the current board state"""
        symbols = {0: " ", 1: "X", -1: "O"}
        print("\n  0   1   2")
        for i in range(3):
            print(
                f"{i} {symbols[self.board[i,0]]} | {symbols[self.board[i,1]]} | {symbols[self.board[i,2]]}"
            )
            if i < 2:
                print("  ---------")


if __name__ == "__main__":
    env = TicTacToeEnvironment()
    env.print_board()
    env.make_move((0, 0))
    print(f"state: {env.get_state()}")
