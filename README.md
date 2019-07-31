Demo project for different approaches for playing tic-tac-toe.

Code requires python 3, numpy, and pytest.

* Make sure to set `PYTHONPATH` to main project directory: In windows, run `path.bat`, or in bash run `$ source path.sh` in the project directory.
* Run tests: `$ pytest`
* Run demo: `$ python -m tictac.main`

Below are the most recent demo results. The current qtable agent plays near-perfect games as O against itself, minimax, and random. Getting good result for the X player was pretty straightforward, but for O it took quite a bit of fiddling with the hyperparameters.

Latest results:

```
C:\Dev\python\tictac>python -m tictac.main
Playing random vs random:
-------------------------
x wins: 58.80%
o wins: 29.00%
draw  : 12.20%

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
x wins: 97.60%
o wins: 0.00%
draw  : 2.40%

Playing random vs minimax random:
---------------------------------
x wins: 0.00%
o wins: 78.70%
draw  : 21.30%

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
x wins: 99.60%
o wins: 0.00%
draw  : 0.40%

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
o wins: 91.00%
draw  : 9.00%

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

number of items in qtable = 626

Training MCTS...
4000/40000 playouts...
8000/40000 playouts...
12000/40000 playouts...
16000/40000 playouts...
20000/40000 playouts...
24000/40000 playouts...
28000/40000 playouts...
32000/40000 playouts...
36000/40000 playouts...
40000/40000 playouts...

Playing random vs MCTS:
-----------------------
x wins: 0.00%
o wins: 73.60%
draw  : 26.40%

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
x wins: 87.90%
o wins: 0.00%
draw  : 12.10%

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

Playing MCTS vs minimax MCTS:
-----------------------------
x wins: 0.00%
o wins: 0.00%
draw  : 100.00%
```
