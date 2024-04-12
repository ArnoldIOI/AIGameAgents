# AIGameAgents
Play tic-tac-toe and connect4 with Search(Minimax) Agents and Reinforcement Learning(Q Learning) Agents.

## Requirements
- Python 3.12
## How to run

```python
# params: <game> <first_mover> <second_mover> [print_game](y/n)
# <game>: ttt, connect4
# <first_mover>: Random, SmartRandom, Q-learning, Minimax, Human
# <second_mover>: Random, SmartRandom, Q-learning, Minimax, Human
# [print_game]: y(default), n
python play.py <game> <first_mover> <second_mover> [print_game](y/n)
```

## Example
```python
# Examples
# A training process will be started if no q-table is found for Q-learning agent
python play.py ttt Minimax Random
python play.py ttt Q-learning SmartRandom
python play.py ttt Human Q-learning
python play.py connect4 Q-learning Random
python play.py connect4 Minimax SmartRandom
python play.py connect4 Human Q-learning
```
