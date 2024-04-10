from consts import GAMES, PLAYERS, RANDOM, SMART_RANDOM, Q_LEARNING, MINIMAX, HUMAN
import sys


def play(game, first_mover, second_mover, print_game=True):
    print("Playing", game)
    print("First mover (X) is", first_mover)
    print("Second mover (O) is", second_mover)
    if game == "ttt":
        _play_ttt(first_mover, second_mover, print_game)
    elif game == "connect4":
        _play_connect4(first_mover, second_mover, print_game)
    print("First mover (X) is", first_mover)
    print("Second mover (O) is", second_mover)


def _play_ttt(first_mover, second_mover, print_game):
    from tictactoe.main import play, init_record
    from tictactoe.player import RandomComputerPlayer, SmartRandomComputerPlayer, QLearningPlayer, MiniMaxPlayer, HumanPlayer
    from consts import TTT_Q_TABLE_PATH
    from tictactoe.game import TicTacToe
    if first_mover == Q_LEARNING or second_mover == Q_LEARNING:
        # check if Q-table exists
        try:
            with open(TTT_Q_TABLE_PATH, "r") as f:
                pass
        except FileNotFoundError:
            print("Q-table not found.")
            is_training_now = input("Do you want to train now? (y/n) ")
            if is_training_now == "y":
                from tictactoe.main import train
                train(100000)
                print("Training complete.")
            else:
                print("Training aborted.")
                return
    players = {
        RANDOM: RandomComputerPlayer,
        SMART_RANDOM: SmartRandomComputerPlayer,
        Q_LEARNING: QLearningPlayer,
        MINIMAX: MiniMaxPlayer,
        HUMAN: HumanPlayer
    }
    first_mover_class = players[first_mover]
    second_mover_class = players[second_mover]
    if first_mover == Q_LEARNING:
        first_mover = QLearningPlayer('')
        print("Loading Q-table...")
        first_mover.load_q_table(TTT_Q_TABLE_PATH)
    else:
        first_mover = first_mover_class('')
    if second_mover == Q_LEARNING:
        second_mover = QLearningPlayer('')
        print("Loading Q-table...")
        second_mover.load_q_table(TTT_Q_TABLE_PATH)
    else:
        second_mover = second_mover_class('')
    init_record(first_mover, second_mover)
    play(TicTacToe(), first_mover, second_mover, print_game)


def _play_connect4(first_mover, second_mover, print_game):
    from connect4.main import play, init_record
    from connect4.player import RandomComputerPlayer, SmartRandomComputerPlayer, QLearningPlayer, MiniMaxPlayer, HumanPlayer
    from consts import CONNECT4_Q_TABLE_PATH
    from connect4.game import Connect4
    if first_mover == Q_LEARNING or second_mover == Q_LEARNING:
        # check if Q-table exists
        try:
            with open(CONNECT4_Q_TABLE_PATH, "r") as f:
                pass
        except FileNotFoundError:
            print("Q-table not found. Training Q-learning player...")
            is_training_now = input("Do you want to train now? (y/n) ")
            if is_training_now == "y":
                from connect4.main import train
                train(10000)
                print("Training complete.")
            else:
                print("Training aborted.")
                return
    players = {
        RANDOM: RandomComputerPlayer,
        SMART_RANDOM: SmartRandomComputerPlayer,
        Q_LEARNING: QLearningPlayer,
        MINIMAX: MiniMaxPlayer,
        HUMAN: HumanPlayer
    }
    first_mover_class = players[first_mover]
    second_mover_class = players[second_mover]
    if first_mover == Q_LEARNING:
        first_mover = QLearningPlayer('')
        print("Loading Q-table...")
        first_mover.load_q_table(CONNECT4_Q_TABLE_PATH)
    else:
        first_mover = first_mover_class('')
    if second_mover == Q_LEARNING:
        second_mover = QLearningPlayer('')
        print("Loading Q-table...")
        second_mover.load_q_table(CONNECT4_Q_TABLE_PATH)
    else:
        second_mover = second_mover_class('')
    init_record(first_mover, second_mover)
    play(Connect4(), first_mover, second_mover, print_game)


def _parse_input():
    if len(sys.argv) < 4:
        raise ValueError("Usage: python play.py <game> <first_mover> <second_mover> [print_game](y/n)")
    game = sys.argv[1]
    first_mover = sys.argv[2]
    second_mover = sys.argv[3]
    print_game = "y" == sys.argv[4] if len(sys.argv) == 5 else True
    if game not in GAMES:
        raise ValueError("Game must be either 'ttt' or 'connect4'")
    if first_mover not in PLAYERS:
        raise ValueError("First mover must be one of the following: ", PLAYERS)
    if second_mover not in PLAYERS:
        raise ValueError("Second mover must be one of the following: ", PLAYERS)
    return game, first_mover, second_mover, print_game


def main():
    game, first_mover, second_mover, print_game = _parse_input()
    play(game, first_mover, second_mover, print_game)


if __name__ == "__main__":
    main()
