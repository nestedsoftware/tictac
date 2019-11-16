Demo project for different approaches for playing tic-tac-toe.

Code requires python 3, numpy, and pytest.

Install using pipenv:

* `pipenv shell`
* `pipenv install --dev`

Make sure to set `PYTHONPATH` to main project directory:

* In windows, run `path.bat`
* In bash run `source path.sh`

Run tests and demo:

* Run tests: `pytest`
* Run demo: `python -m tictac.main`
* Run neural net demo: `python -m tictac.main_qneural`

Below are the most recent demo results. The current qtable agent plays near-perfect games as O against itself, minimax, and random. Getting good result for the X player was pretty straightforward, but for O it took quite a bit of fiddling with the hyperparameters.

Latest results:

```
C:\Dev\python\tictac>python -m tictac.main
Playing random vs random:
-------------------------
x wins: 60.10%
o wins: 28.90%
draw  : 11.00%

Playing minimax not random vs minimax random:
---------------------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing minimax random vs minimax not random:
---------------------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing minimax not random vs minimax not random:
-------------------------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing minimax random vs minimax random:
-----------------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing minimax random vs random:
---------------------------------
x wins: 96.80%
o wins: 0.00%
draw  : 3.20%

Playing random vs minimax random:
---------------------------------
x wins: 0.00%
o wins: 78.10%
draw  : 21.90%

Training qtable X vs. random...
700/7000 games, using epsilon=0.6...
1400/7000 games, using epsilon=0.5...
2100/7000 games, using epsilon=0.4...
2800/7000 games, using epsilon=0.30000000000000004...
3500/7000 games, using epsilon=0.20000000000000004...
4200/7000 games, using epsilon=0.10000000000000003...
4900/7000 games, using epsilon=2.7755575615628914e-17...
5600/7000 games, using epsilon=0...
6300/7000 games, using epsilon=0...
7000/7000 games, using epsilon=0...
Training qtable O vs. random...
700/7000 games, using epsilon=0.6...
1400/7000 games, using epsilon=0.5...
2100/7000 games, using epsilon=0.4...
2800/7000 games, using epsilon=0.30000000000000004...
3500/7000 games, using epsilon=0.20000000000000004...
4200/7000 games, using epsilon=0.10000000000000003...
4900/7000 games, using epsilon=2.7755575615628914e-17...
5600/7000 games, using epsilon=0...
6300/7000 games, using epsilon=0...
7000/7000 games, using epsilon=0...

Playing qtable vs random:
-------------------------
x wins: 95.40%
o wins: 0.00%
draw  : 4.60%

Playing qtable vs minimax random:
---------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing qtable vs minimax:
--------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing random vs qtable:
-------------------------
x wins: 0.00%
o wins: 78.10%
draw  : 21.90%

Playing minimax random vs qtable:
---------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing minimax vs qtable:
--------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing qtable vs qtable:
-------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

number of items in qtable = 624

Training MCTS...
400/4000 playouts...
800/4000 playouts...
1200/4000 playouts...
1600/4000 playouts...
2000/4000 playouts...
2400/4000 playouts...
2800/4000 playouts...
3200/4000 playouts...
3600/4000 playouts...
4000/4000 playouts...

Playing random vs MCTS:
-----------------------
x wins: 0.00%
o wins: 63.50%
draw  : 36.50%

Playing minimax vs MCTS:
------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing minimax random vs MCTS:
-------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing MCTS vs random:
-----------------------
x wins: 81.80%
o wins: 0.00%
draw  : 18.20%

Playing MCTS vs minimax:
------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing MCTS vs minimax random:
-------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing MCTS vs MCTS:
---------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%
```

```
C:\Dev\python\tictac>python -m tictac.main_qneural
Training qlearning X vs. random...
700000/7000000 games, using epsilon=0.6...
1400000/7000000 games, using epsilon=0.5...
2100000/7000000 games, using epsilon=0.4...
2800000/7000000 games, using epsilon=0.30000000000000004...
3500000/7000000 games, using epsilon=0.20000000000000004...
4200000/7000000 games, using epsilon=0.10000000000000003...
4900000/7000000 games, using epsilon=2.7755575615628914e-17...
5600000/7000000 games, using epsilon=0...
6300000/7000000 games, using epsilon=0...
7000000/7000000 games, using epsilon=0...
Training qlearning O vs. random...
700000/7000000 games, using epsilon=0.6...
1400000/7000000 games, using epsilon=0.5...
2100000/7000000 games, using epsilon=0.4...
2800000/7000000 games, using epsilon=0.30000000000000004...
3500000/7000000 games, using epsilon=0.20000000000000004...
4200000/7000000 games, using epsilon=0.10000000000000003...
4900000/7000000 games, using epsilon=2.7755575615628914e-17...
5600000/7000000 games, using epsilon=0...
6300000/7000000 games, using epsilon=0...
7000000/7000000 games, using epsilon=0...

Playing qneural vs random:
--------------------------
x wins: 96.70%
o wins: 0.90%
draw  : 2.40%

Playing qneural vs minimax random:
----------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing qneural vs minimax:
---------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing random vs qneural:
--------------------------
x wins: 0.00%
o wins: 73.10%
draw  : 26.90%

Playing minimax random vs qneural:
----------------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing minimax vs qneural:
---------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%

Playing qneural vs qneural:
---------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%
```