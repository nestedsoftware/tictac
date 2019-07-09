Demo project for different approaches for playing tic-tac-toe.

Code requires python 3, numpy, and pytest.

* Make sure to set `PYTHONPATH` to main project directory: In windows, run `path.bat`, or in bash run `$ source path.sh` in the project directory.
* Run tests: `$ pytest`
* Run demo: `$ python -m tictac.main`

Below are the most recent demo results. It took quite a bit of fiddling, the current qtable agent plays near-perfect games as O against itself, minimax, and random. Getting good result for the X player was pretty straightforward, but for O it took quite a bit of fiddling with the hyperparameters. 

Latest results:

```
C:\Dev\python\tictac>python -m tictac.main
Playing random vs random:
-------------------------
x wins: 59.70%
o wins: 27.50%
draw  : 12.80%

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
x wins: 96.10%
o wins: 0.00%
draw  : 3.90%

Playing random vs minimax random:
---------------------------------
x wins: 0.00%
o wins: 76.60%
draw  : 23.40%

Training qtable X vs. random/minimax random...
played 5000 games, using epsilon=0.7000000000000001...
played 10000 games, using epsilon=0.6000000000000001...
played 15000 games, using epsilon=0.5000000000000001...
played 20000 games, using epsilon=0.40000000000000013...
played 25000 games, using epsilon=0.30000000000000016...
played 30000 games, using epsilon=0.20000000000000015...
played 35000 games, using epsilon=0.10000000000000014...
played 40000 games, using epsilon=1.3877787807814457e-16...
played 45000 games, using epsilon=0...
played 50000 games, using epsilon=0...
Training qtable O vs. random/minimax random...
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
x wins: 99.50%
o wins: 0.00%
draw  : 0.50%

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
x wins: 0.20%
o wins: 92.40%
draw  : 7.40%

Playing minimax random vs qtable:
---------------------------------
x wins: 1.60%
o wins: 0.00%
draw  : 98.40%

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
