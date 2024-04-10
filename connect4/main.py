"""
This file is for the main game logic of Connect4
"""
import time

from .game import Connect4
from .player import HumanPlayer, RandomComputerPlayer, SmartRandomComputerPlayer, MiniMaxPlayer, QLearningPlayer

Q_TABLE_PATH = "./connect4/q_table.txt"

# Record results
timer = {}  # response time
moves = {}  # move count
results = {}  # winning rate


def init_record(x, o):
    global timer, moves, results
    x_name = x.__class__.__name__
    o_name = o.__class__.__name__
    timer = {x_name: 0, o_name: 0}
    moves = {x_name: 0, o_name: 0}
    results = {x_name: 0, o_name: 0, 'tie': 0}


def play(game, x_player, o_player, print_game=True):
    x_player.letter = 'X'
    o_player.letter = 'O'
    if print_game:
        print(game)

    while game.available_moves():
        s = time.time()
        if game.turn == 'O':
            col = o_player.get_move(game)
            name = o_player.__class__.__name__
        else:
            name = x_player.__class__.__name__
            col = x_player.get_move(game)
        e = time.time()

        if game.make_move(col, game.turn):
            timer[name] += e - s
            moves[name] += 1
            if print_game:
                print(f'{game.turn} makes a move to column {col}')
                print(game)
                print('')

            winner = game.get_winner()
            if winner:
                if print_game:
                    print(f'{winner} wins!')
                wining_player = x_player if winner == 'X' else o_player
                results[wining_player.__class__.__name__] += 1
                return winner

            game.change_turn()
        else:
            print('Invalid move! Try again.')
            continue
    if print_game:
        print('It\'s a tie!')
    results['tie'] += 1
    return 'tie'


def train_q_learning_player(q_player, opponent, game, num_episodes=1000):
    for episode in range(num_episodes):
        # Reset the game at the start of each new game episode
        game.reset()

        # This is for training against a random player
        # Randomly choose who goes first
        # if random.randint(0, 1) == 0:
        #     q_player.letter = 'X'
        #     opponent.letter = 'O'
        #     current_player = q_player
        # else:
        #     q_player.letter = 'O'
        #     opponent.letter = 'X'
        #     current_player = opponent

        # This is for training against itself
        current_player = q_player
        while not game.game_over():
            move = current_player.get_move(game)
            game.make_move(move, current_player.letter)

            # Switch turns
            if current_player == q_player:
                current_player = opponent
            else:
                current_player = q_player

        if game.current_winner == q_player.letter:
            reward = 1
        elif game.current_winner is None:
            reward = 0.1
        else:
            reward = -1

        q_player.update_q_values(reward)
        # training against itself
        opponent.update_q_values(-reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Q-Player learns with reward {reward}")
            print("delta:", q_player.delta)
    # Save the Q-table
    q_player.save_q_table(Q_TABLE_PATH)
    print("alpha:", q_player.alpha)
    print("epsilon:", q_player.epsilon)


def train(num_episodes=100000):
    q_table = {}
    q_player_1 = QLearningPlayer('X', q_table, training_mode=True)
    q_player_2 = QLearningPlayer('O', q_table, training_mode=True)
    game = Connect4()
    train_q_learning_player(q_player_1, q_player_2, game, num_episodes=num_episodes)


def minimaxVSrandom(n=100, minimax_first=True):
    mini = MiniMaxPlayer('', pruning=True, depth=5)
    random = SmartRandomComputerPlayer('')
    init_record(mini, random)
    for i in range(n):
        c = Connect4()
        if minimax_first:
            play(c, mini, random, print_game=False)
        else:
            play(c, random, mini, print_game=False)
    print(results, timer, moves)
    print("winning rate of MiniMax player:", results[mini.__class__.__name__] / n)
    print("tie rate:", results['tie'] / n)
    print("Average Move Count for MiniMax Player:", moves[mini.__class__.__name__] / n)
    print("Average Response Time for MiniMax Player:", timer[mini.__class__.__name__] / moves[mini.__class__.__name__])


def qVSrandom(n=100, q_first=True):
    q_player = QLearningPlayer('', training_mode=False)
    q_player.load_q_table(Q_TABLE_PATH)
    random = SmartRandomComputerPlayer('')
    init_record(q_player, random)
    for i in range(n):
        c = Connect4()
        if q_first:
            play(c, q_player, random, print_game=False)
        else:
            play(c, random, q_player, print_game=False)
    # print(results, timer, moves)
    print("winning rate of Q player:", results[q_player.__class__.__name__] / n)
    print("tie rate:", results['tie'] / n)
    print("Average Move Count for Q Player:", moves[q_player.__class__.__name__] / n)
    print("Average Response Time for Q Player:",
          timer[q_player.__class__.__name__] / moves[q_player.__class__.__name__])


def minimaxVSq(n=100, minimax_first=True):
    mini = MiniMaxPlayer('', pruning=True, depth=5)
    q_player = QLearningPlayer('', training_mode=False)
    q_player.load_q_table(Q_TABLE_PATH)
    init_record(mini, q_player)
    for i in range(n):
        c = Connect4()
        if minimax_first:
            play(c, mini, q_player, print_game=False)
        else:
            play(c, q_player, mini, print_game=False)
    # print(results, timer, moves)
    print("winning rate of MiniMax player:", results[mini.__class__.__name__] / n)
    print("winning rate of Q player:", results[q_player.__class__.__name__] / n)
    print("tie rate:", results['tie'] / n)
    print("Average Move Count for MiniMax Player:", moves[mini.__class__.__name__] / n)
    print("Average Move Count for Q Player:", moves[q_player.__class__.__name__] / n)
    print("Average Response Time for MiniMax Player:", timer[mini.__class__.__name__] / moves[mini.__class__.__name__])
    print("Average Response Time for Q Player:",
          timer[q_player.__class__.__name__] / moves[q_player.__class__.__name__])


if __name__ == '__main__':
    # Compare the performance of different players
    # ==============================================================================================================
    print("MiniMax vs Random\n")
    print("MiniMax First")
    minimaxVSrandom(2, minimax_first=True)
    # print("\nRandom First")
    # minimaxVSrandom(50, minimax_first=False)
    #
    # print("\n\n\nQ vs Random\n")
    # print("Q First")
    # qVSrandom(50, q_first=True)
    # print("\nRandom First")
    # qVSrandom(50, q_first=False)
    #
    # print("\n\n\nMiniMax vs Q\n")
    # print("MiniMax First")
    # minimaxVSq(50, minimax_first=True)
    # print("\nQ First")
    # minimaxVSq(50, minimax_first=False)

    # ==============================================================================================================
    # Train the Q-learning player
    # train(100000)
