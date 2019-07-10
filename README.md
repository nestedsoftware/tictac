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
x wins: 58.70%
o wins: 27.90%
draw  : 13.40%

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
x wins: 96.70%
o wins: 0.00%
draw  : 3.30%

Playing random vs minimax random:
---------------------------------
x wins: 0.00%
o wins: 78.10%
draw  : 21.90%

Training qtable X vs. random and minimax random...
played 5000 games, using epsilon=0.85...
played 10000 games, using epsilon=0.75...
played 15000 games, using epsilon=0.65...
played 20000 games, using epsilon=0.55...
played 25000 games, using epsilon=0.45000000000000007...
played 30000 games, using epsilon=0.3500000000000001...
played 35000 games, using epsilon=0.2500000000000001...
played 40000 games, using epsilon=0.1500000000000001...
played 45000 games, using epsilon=0.0500000000000001...
played 50000 games, using epsilon=0...
Training qtable O vs. random and minimax random...
played 5000 games, using epsilon=0.85...
played 10000 games, using epsilon=0.75...
played 15000 games, using epsilon=0.65...
played 20000 games, using epsilon=0.55...
played 25000 games, using epsilon=0.45000000000000007...
played 30000 games, using epsilon=0.3500000000000001...
played 35000 games, using epsilon=0.2500000000000001...
played 40000 games, using epsilon=0.1500000000000001...
played 45000 games, using epsilon=0.0500000000000001...
played 50000 games, using epsilon=0...

Playing qtable vs random:
-------------------------
x wins: 99.70%
o wins: 0.00%
draw  : 0.30%

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
o wins: 91.30%
draw  : 8.70%

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
```
