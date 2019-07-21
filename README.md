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
x wins: 55.60%
o wins: 30.30%
draw  : 14.10%

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
x wins: 96.90%
o wins: 0.00%
draw  : 3.10%

Playing random vs minimax random:
---------------------------------
x wins: 0.00%
o wins: 76.60%
draw  : 23.40%

Training qtable X vs. random...
played 700 games, using epsilon=0.6...
played 1400 games, using epsilon=0.5...
played 2100 games, using epsilon=0.4...
played 2800 games, using epsilon=0.30000000000000004...
played 3500 games, using epsilon=0.20000000000000004...
played 4200 games, using epsilon=0.10000000000000003...
played 4900 games, using epsilon=2.7755575615628914e-17...
played 5600 games, using epsilon=0...
played 6300 games, using epsilon=0...
played 7000 games, using epsilon=0...
Training qtable O vs. random...
played 700 games, using epsilon=0.6...
played 1400 games, using epsilon=0.5...
played 2100 games, using epsilon=0.4...
played 2800 games, using epsilon=0.30000000000000004...
played 3500 games, using epsilon=0.20000000000000004...
played 4200 games, using epsilon=0.10000000000000003...
played 4900 games, using epsilon=2.7755575615628914e-17...
played 5600 games, using epsilon=0...
played 6300 games, using epsilon=0...
played 7000 games, using epsilon=0...

Playing qtable vs random:
-------------------------
x wins: 99.10%
o wins: 0.00%
draw  : 0.90%

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
o wins: 92.60%
draw  : 7.40%

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

number of items in qtable = 625
```
