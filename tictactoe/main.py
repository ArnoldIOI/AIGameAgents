import random
import time

from .game import TicTacToe
from .player import HumanPlayer, RandomComputerPlayer, SmartRandomComputerPlayer, MiniMaxPlayer, QLearningPlayer

Q_TABLE_PATH = "tictactoe/q_table.txt"

timer = {}
moves = {}
results = {}


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
    # returns the winner of the game! or None for a tie
    if print_game:
        game.print_board_nums()

    letter = 'X'  # starting letter
    # iterate while the game still has empty squares
    # (we don't have to worry about winner because we'll just return that
    # which breaks the loop)
    while not game.game_over():
        # get the move from the appropriate player
        if letter == 'O':
            s = time.time()
            square = o_player.get_move(game)
            player_name = o_player.__class__.__name__
            timer[player_name] += time.time() - s
            moves[player_name] += 1
        else:
            s = time.time()
            square = x_player.get_move(game)
            player_name = x_player.__class__.__name__
            timer[player_name] += time.time() - s
            moves[player_name] += 1

        # let's define a function to make a move
        if game.make_move(square, letter):
            if print_game:
                print(letter + f' makes a move to square {square}')
                game.print_board()
                print('')  # empty line

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                winning_player = x_player if letter == 'X' else o_player
                results[winning_player.__class__.__name__] += 1
                return letter

            # after we made our move, we need to alternate letters
            letter = 'O' if letter == 'X' else 'X'  # switches player
        else:
            if print_game:
                print('Square is already occupied! Please choose another.')
                # whose turn?
                print(letter + '\'s turn.')
                game.print_board()
            continue

        # tiny break to make things a little easier to read
        if print_game:
            # time.sleep(0.8)
            print('-------------------')
    if print_game:
        print('It\'s a tie!')
    results['tie'] += 1
    return 'tie'


def train_q_learning_player(q_player, opponent, game, num_episodes=1000):
    for episode in range(num_episodes):
        # Reset the game at the start of each new game episode
        game.board = [' ' for _ in range(9)]
        game.current_winner = None

        # Randomly choose who goes first
        if random.randint(0, 1) == 0:
            q_player.letter = 'X'
            opponent.letter = 'O'
            current_player = q_player
        else:
            q_player.letter = 'O'
            opponent.letter = 'X'
            current_player = opponent

        while not game.game_over():
            move = current_player.get_move(game)
            game.make_move(move, current_player.letter)

            # Switch turns
            if current_player == q_player:
                current_player = opponent
            else:
                current_player = q_player

        # After the game is over, we need to update Q-values
        # Reward: 1 for win, -1 for loss, 0.5 for tie
        if game.current_winner == q_player.letter:
            reward = 1  # Q-player wins
        elif game.current_winner is None:
            reward = 0.1
        else:
            reward = -1  # Q-player loses

        q_player.update_q_values(reward)
        opponent.update_q_values(-reward)

        # Print some information
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Q-Player learns with reward {reward}")
            print("delta:", q_player.delta)
    # Save the Q-table
    q_player.save_q_table(Q_TABLE_PATH)
    print("alpha:", q_player.alpha)
    print("epsilon:", q_player.epsilon)


def train(num_episodes=1000):
    q_table = {}
    q_player = QLearningPlayer('X', q_table, training_mode=True)
    q_player_2 = QLearningPlayer('O', q_table, training_mode=True)
    # random_player = SmartRandomComputerPlayer('O')
    # random_player = RandomComputerPlayer('O')
    game = TicTacToe()
    train_q_learning_player(q_player, q_player_2, game, num_episodes=num_episodes)
    return q_player, q_player_2


def minimaxVSrandom(n=100, minimax_first=True):
    """
    Play a game of Tic-Tac-Toe between a MiniMax player and a Random player
    1. n games are played
    2. half of the games are played with the MiniMax player as 'X' and the Random player as 'O'
    3. the other half of the games are played with the MiniMax player as 'O' and the Random player as 'X'
    :return:
    """
    mini = MiniMaxPlayer('', pruning=True)
    random = SmartRandomComputerPlayer('')
    init_record(mini, random)
    for i in range(n):
        t = TicTacToe()
        if minimax_first:
            play(t, mini, random, print_game=False)
        else:
            play(t, random, mini, print_game=False)
    # print(results, timer, moves)
    # print wining rate of the MiniMax player
    print("winning rate of MiniMax player:", results[mini.__class__.__name__] / n)
    print("tie rate:", results['tie'] / n)
    # Average Move Count for MiniMax Player
    print("Average Move Count for MiniMax Player:", moves[mini.__class__.__name__] / n)
    # Average Response Time for MiniMax Player
    print("Average Response Time for MiniMax Player:", timer[mini.__class__.__name__] / moves[mini.__class__.__name__])


def minimaxVSq(n=100, minimax_first=True):
    mini = MiniMaxPlayer('', pruning=True)
    q_player = QLearningPlayer('', training_mode=False)
    q_player.load_q_table(Q_TABLE_PATH)
    init_record(mini, q_player)
    for i in range(n):
        t = TicTacToe()
        if minimax_first:
            play(t, mini, q_player, print_game=False)
        else:
            play(t, q_player, mini, print_game=False)
    print(results, timer, moves)
    # print wining rate of the MiniMax player
    print("winning rate of MiniMax player:", results[mini.__class__.__name__] / n)
    print("winning rate of Q player:", results[q_player.__class__.__name__] / n)
    print("tie rate:", results['tie'] / n)
    # Average Move Count for MiniMax Player
    print("Average Move Count for MiniMax Player:", moves[mini.__class__.__name__] / n)
    print("Average Move Count for Q Player:", moves[q_player.__class__.__name__] / n)
    # Average Response Time for MiniMax Player
    print("Average Response Time for MiniMax Player:", timer[mini.__class__.__name__] / moves[mini.__class__.__name__])
    print("Average Response Time for Q Player:",
          timer[q_player.__class__.__name__] / moves[q_player.__class__.__name__])


def qVSrandom(n=100, q_first=True):
    q_player = QLearningPlayer('', training_mode=False)
    q_player.load_q_table(Q_TABLE_PATH)
    random = SmartRandomComputerPlayer('')
    init_record(q_player, random)
    for i in range(n):
        t = TicTacToe()
        if q_first:
            play(t, q_player, random, print_game=False)
        else:
            play(t, random, q_player, print_game=False)
    # print(results, timer, moves)
    # print wining rate of the MiniMax player
    print("winning rate of Q player:", results[q_player.__class__.__name__] / n)
    print("tie rate:", results['tie'] / n)
    # Average Move Count for MiniMax Player
    print("Average Move Count for Q Player:", moves[q_player.__class__.__name__] / n)
    # Average Response Time for MiniMax Player
    print("Average Response Time for Q Player:",
          timer[q_player.__class__.__name__] / moves[q_player.__class__.__name__])


if __name__ == '__main__':
    print("MiniMax vs Random\n")
    minimaxVSrandom(500, minimax_first=True)
    minimaxVSrandom(500, minimax_first=False)
    print("\n\n\nQ vs Random\n")
    qVSrandom(500, q_first=True)
    qVSrandom(500, q_first=False)
    print("\n\n\nMiniMax vs Q\n")
    minimaxVSq(500, minimax_first=True)
    minimaxVSq(500, minimax_first=False)
    # If you don't have a q_table.txt, you can train a Q-learning player
    # train(1000000)
