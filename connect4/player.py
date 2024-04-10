"""
This file is for defining the player of Connect4
A Player Class
A HumanPlayer Class
A RandomComputerPlayer Class
A SmartRandomComputerPlayer Class
A MiniMaxPlayer Class
A QLearningPlayer Class
"""
import random


def other_letter(letter):
    return 'O' if letter == 'X' else 'X'


class Player:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        pass


class RandomComputerPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        square = random.choice(game.available_moves())
        return square


class SmartRandomComputerPlayer(Player):
    """
    This player will try to make a winning move if it can
    And it will block the opponent from winning if it can
    Otherwise, it will just make a random move
    """

    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        # if there is a move that can win, take that move
        for move in game.available_moves():
            game_copy = game.copy()
            game_copy.make_move(move, self.letter)
            if game_copy.get_winner() == self.letter:
                return move

        # if the opponent can win, block that move
        for move in game.available_moves():
            game_copy = game.copy()
            game_copy.make_move(move, other_letter(self.letter))
            if game_copy.get_winner() == other_letter(self.letter):
                return move

        # otherwise, take a random move
        return random.choice(game.available_moves())


class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-6): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val


class MiniMaxPlayer(Player):
    def __init__(self, letter, pruning=True, depth=4):
        super().__init__(letter)
        self.pruning = pruning
        self.depth = depth  # Max depth limit

    def get_move(self, game):
        if len(game.available_moves()) == 0:
            return None
        if self.pruning:
            return self.minimax_with_alpha_beta_pruning(game, self.letter, -float('inf'), float('inf'), self.depth)[
                'position']
        else:
            return self.minimax(game, self.letter, self.depth)['position']

    def minimax(self, state, player, depth):
        if depth == 0 or state.current_winner is not None or not state.empty_squares():
            return {'position': None, 'score': state.evaluate(player)}

        max_player = self.letter
        other_player = other_letter(player)
        best = {'position': None, 'score': -float('inf') if player == max_player else float('inf')}

        for possible_move in state.available_moves():
            position = state.make_move(possible_move, player)
            sim_score = self.minimax(state, other_player, depth - 1)
            state.board[position[0]][position[1]] = ' '  # reset the board
            state.current_winner = None
            sim_score['position'] = possible_move

            if player == max_player and sim_score['score'] > best['score'] or player != max_player and sim_score[
                'score'] < best['score']:
                best = sim_score

        return best

    def minimax_with_alpha_beta_pruning(self, state, player, alpha, beta, depth):
        if depth == 0 or state.current_winner is not None or not state.empty_squares():
            return {'position': None, 'score': state.evaluate(player)}

        max_player = self.letter
        other_player = other_letter(player)
        best = {'position': None, 'score': -float('inf') if player == max_player else float('inf')}

        for possible_move in state.available_moves():
            position = state.make_move(possible_move, player)
            sim_score = self.minimax_with_alpha_beta_pruning(state, other_player, alpha, beta, depth - 1)
            state.board[position[0]][position[1]] = ' '  # reset the board
            state.current_winner = None
            sim_score['position'] = possible_move

            if player == max_player:
                if sim_score['score'] > best['score']:
                    best = sim_score
                    alpha = max(alpha, sim_score['score'])
                    if beta <= alpha:
                        break
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
                    beta = min(beta, sim_score['score'])
                    if beta <= alpha:
                        break

        return best


class QLearningPlayer(Player):
    def __init__(self, letter, q_table=None, training_mode=False, alpha=0.9, alpha_decay=0.999, alpha_min=0.1,
                 gamma=0.8, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(letter)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {} if q_table is None else q_table
        self.state_history = []
        self.training_mode = training_mode

    def save_q_table(self, filename):
        with open(filename, 'w') as f:
            for key, value in self.q_table.items():
                state, action = key
                f.write(f"{state};{action};{value}\n")

    def load_q_table(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                state, action, value = line.strip().split(';')
                self.q_table[(eval(state), int(action))] = float(value)

    def get_state(self, game):
        return tuple([tuple(row) for row in game.board])

    def get_move(self, game):
        state = self.get_state(game)
        available_moves = game.available_moves()

        if self.training_mode and random.random() < self.epsilon:
            move = random.choice(available_moves)
        else:
            move = self.choose_best_move(state, available_moves)

        self.state_history.append((state, move))

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return move

    def choose_best_move(self, state, available_moves):
        best_value = -float('inf')
        best_move = None
        for move in available_moves:
            value = self.q_table.get((state, move), 0)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move if best_move is not None else random.choice(available_moves)

    def update_q_values(self, reward):
        self.delta = 0
        for state, action in reversed(self.state_history):
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            old_q_value = self.q_table.get((state, action), 0)
            future_rewards = []
            available_moves = [col for col in range(7) if state[0][col] == ' ']
            for next_move in available_moves:
                future_state = list(state)
                for row in range(5, -1, -1):
                    if future_state[row][next_move] == ' ':
                        future_state[row] = list(future_state[row])
                        future_state[row][next_move] = self.letter
                        future_state[row] = tuple(future_state[row])
                        break
                future_state = tuple(future_state)
                future_rewards.append(self.q_table.get((future_state, next_move), 0))
            max_future_reward = max(future_rewards) if future_rewards else 0
            new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_reward - old_q_value)
            self.delta += abs(new_q_value - old_q_value)
            self.q_table[(state, action)] = new_q_value

        self.state_history = []
