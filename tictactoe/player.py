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
            if game_copy.current_winner == self.letter:
                return move

        # if the opponent can win, block that move
        for move in game.available_moves():
            game_copy = game.copy()
            game_copy.make_move(move, other_letter(self.letter))
            if game_copy.current_winner == other_letter(self.letter):
                return move

        # otherwise, take a random move
        return random.choice(game.available_moves())


class MiniMaxPlayer(Player):
    def __init__(self, letter, pruning=True):
        super().__init__(letter)
        self.pruning = pruning

    def get_move(self, game):
        if len(game.available_moves()) == 9:
            random.choice(game.available_moves())
        if self.pruning:
            return self.minimax_with_alpha_beta_pruning(game, self.letter, -float('inf'), float('inf'))['position']
        else:
            return self.minimax(game, self.letter)['position']

    def minimax(self, state, player):
        max_player = self.letter
        other_player = other_letter(player)

        if state.current_winner == other_player:
            return {'position': None,
                    'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                            state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -float('inf')}
        else:
            best = {'position': None, 'score': float('inf')}

        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimax(state, other_player)  # alternate players
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move

            if player == max_player:  # max
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:  # min
                if sim_score['score'] < best['score']:
                    best = sim_score
        return best

    def minimax_with_alpha_beta_pruning(self, state, player, alpha, beta):
        max_player = self.letter
        other_player = other_letter(player)

        if state.current_winner == other_player:
            return {'position': None,
                    'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                            state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -float('inf')}
        else:
            best = {'position': None, 'score': float('inf')}

        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimax_with_alpha_beta_pruning(state, other_player, alpha, beta)
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move

            if player == max_player:
                if sim_score['score'] > best['score']:
                    best = sim_score
                    alpha = max(alpha, best['score'])
                    if beta <= alpha:
                        break
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
                    beta = min(beta, best['score'])
                    if beta <= alpha:
                        break
        return best


class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-8): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val


class QLearningPlayer(Player):
    def __init__(self, letter, q_table=None, training_mode=False, alpha=0.9, alpha_decay=0.999, alpha_min=0.1,
                 gamma=0.8, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, debug=False):
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
        self.delta = 0
        self.training_mode = training_mode
        self.debug = debug

    def save_q_table(self, filename):
        with open(filename, 'w') as f:
            for key, value in self.q_table.items():
                f.write(f'({key[0]}, {key[1]}):{value}\n')
        print(f'Saved {len(self.q_table)} Q-values to {filename}')

    def load_q_table(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                key, value = line.split(':')
                key = ((key[3], key[8], key[13], key[18], key[23], key[28], key[33], key[38], key[43]), int(key[-2]))
                self.q_table[key] = float(value)

    def get_move(self, game):
        state = self.get_state(game)
        available_moves = game.available_moves()

        if self.training_mode and random.random() < self.epsilon:
            if self.debug:
                print("Random move")
            move = random.choice(available_moves)
        else:
            if self.debug:
                print("Best move")
            move = self.choose_best_move(state, available_moves)

        self.state_history.append((self.get_state(game), move))

        # Decrement epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return move

    def get_state(self, game):
        return tuple(game.board)

    def choose_best_move(self, state, available_moves):
        if self.debug:
            values = {move: self.q_table.get((state, move), 0) for move in available_moves}
            print(values)
        best_value = -float('inf')
        best_move = random.choice(available_moves)
        for move in available_moves:
            value = self.q_table.get((state, move), 0)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def update_q_values(self, reward):
        self.delta = 0
        for state, action in reversed(self.state_history):
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            old_q_value = self.q_table.get((state, action), 0)
            future_rewards = []
            for next_move in range(9):
                future_state = list(state)
                if future_state[next_move] == ' ':
                    future_state[next_move] = self.letter
                    future_state = tuple(future_state)
                    future_rewards.append(self.q_table.get((future_state, next_move), 0))
            max_future_reward = max(future_rewards) if future_rewards else 0
            new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_reward - old_q_value)
            self.delta += abs(new_q_value - old_q_value)
            self.q_table[(state, action)] = new_q_value
        if self.debug:
            print("Delta:", self.delta)
        self.state_history = []
