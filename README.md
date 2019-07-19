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
x wins: 61.00%
o wins: 28.10%
draw  : 10.90%

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
x wins: 96.50%
o wins: 0.00%
draw  : 3.50%

Playing random vs minimax random:
---------------------------------
x wins: 0.00%
o wins: 75.90%
draw  : 24.10%

Training qtable X vs. random...
played 600 games, using epsilon=0.85...
played 1200 games, using epsilon=0.75...
played 1800 games, using epsilon=0.65...
played 2400 games, using epsilon=0.55...
played 3000 games, using epsilon=0.45000000000000007...
played 3600 games, using epsilon=0.3500000000000001...
played 4200 games, using epsilon=0.2500000000000001...
played 4800 games, using epsilon=0.1500000000000001...
played 5400 games, using epsilon=0.0500000000000001...
played 6000 games, using epsilon=0...
Training qtable O vs. random...
played 600 games, using epsilon=0.85...
played 1200 games, using epsilon=0.75...
played 1800 games, using epsilon=0.65...
played 2400 games, using epsilon=0.55...
played 3000 games, using epsilon=0.45000000000000007...
played 3600 games, using epsilon=0.3500000000000001...
played 4200 games, using epsilon=0.2500000000000001...
played 4800 games, using epsilon=0.1500000000000001...
played 5400 games, using epsilon=0.0500000000000001...
played 6000 games, using epsilon=0...

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
o wins: 91.70%
draw  : 8.30%

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

number of items in qtable = 627
```
