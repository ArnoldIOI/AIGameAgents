class Connect4:
    def __init__(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.turn = 'X'
        self.current_winner = None

    def reset(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.turn = 'X'
        self.current_winner = None

    def make_move(self, col, turn):
        if not 0 <= col < 7:
            return False
        if self.board[0][col] != ' ':
            return False
        for row in range(5, -1, -1):
            if self.board[row][col] == ' ':
                self.board[row][col] = turn
                self.current_winner = self.get_winner()
                return [row, col]
        return False

    def change_turn(self):
        self.turn = 'O' if self.turn == 'X' else 'X'

    def empty_squares(self):
        return ' ' in [cell for row in self.board for cell in row]

    def num_empty_squares(self):
        return sum(row.count(' ') for row in self.board)

    def available_moves(self):
        """
        Get the available moves for the current board
        :return:
        """
        return [col for col in range(7) if self.board[0][col] == ' ']

    def get_winner(self):
        for row in range(6):
            for col in range(7):
                if self.board[row][col] == ' ':
                    continue
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    for i in range(1, 4):
                        r, c = row + i * dr, col + i * dc
                        if not (0 <= r < 6 and 0 <= c < 7):
                            break
                        try:
                            if self.board[r][c] != self.board[row][col]:
                                break
                        except Exception as e:
                            print(self.board)
                            print(r, c)
                            print(row, col)
                            raise e
                    else:
                        return self.board[row][col]
        return None

    def __str__(self):
        return '\n'.join(['|'.join(row) for row in self.board])

    def copy(self):
        new_board = Connect4()
        new_board.board = [row.copy() for row in self.board]
        new_board.turn = self.turn
        new_board.current_winner = self.current_winner
        return new_board

    def undo_move(self, col):
        for row in range(6):
            if self.board[row][col] != ' ':
                self.board[row][col] = ' '
                return True
        return False

    def evaluate(self, player):
        """Evaluate the board for a specific player to assign a heuristic score."""
        score = 0
        opponent = 'O' if player == 'X' else 'X'
        sequences = {'2': 10, '3': 100, '4': 1000}

        def count_sequences(player, sequence_length):
            count = 0
            for row in self.board:
                # Count horizontal sequences
                for col in range(7 - sequence_length + 1):
                    if all(cell == player for cell in row[col:col + sequence_length]):
                        count += 1
            # Count vertical sequences
            for col in range(7):
                for row in range(6 - sequence_length + 1):
                    if all(self.board[r][col] == player for r in range(row, row + sequence_length)):
                        count += 1
            # Count diagonal sequences
            for row in range(6 - sequence_length + 1):
                for col in range(7 - sequence_length + 1):
                    if all(self.board[row + i][col + i] == player for i in range(sequence_length)):
                        count += 1
                    if col >= sequence_length - 1:
                        if all(self.board[row + i][col - i] == player for i in range(sequence_length)):
                            count += 1
            return count

        # Sum up the sequence scores for the player
        for seq, seq_score in sequences.items():
            length = int(seq)
            score += count_sequences(player, length) * seq_score
            score -= count_sequences(opponent, length) * seq_score

        return score

    def game_over(self):
        return self.current_winner is not None or not self.empty_squares()
