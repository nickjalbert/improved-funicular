# improved-funicular

Python 3 implementation of the [2048 game](https://play2048.co/).

## Setup

Developed under Python 3.7 on OS X Catalina with pip and virtualenv installed
on the system.  Look [here](https://stackoverflow.com/a/23842752) to setup
Python 3.7.

* `git clone https://github.com/nickjalbert/improved-funicular.git`
* `cd improved-funicular`
* `virtualenv improved-funicular`
* `source improved-funicular/bin/activate`
* `pip install -r requirements.txt`
* `python play_in_console.py nick` or `python play_in_console.py andy`

## Dev and Testing

To automatically format your code, run:

* `black [filename.py]`

To lint, run:

* `./scripts/lint`

To test, run:

* `pytest`

## Example strategy trials

Run `python do_stats.py` to try several different strategies.

```
$ python do_stats.py

Running 1000 trials with Nick impl to test each strategy

Strategy only moves right (0.27 sec total, 0.00027 sec per trial):
	Max Tile: 16
	Max Score: 200
	Mean Score: 13.732
	Median Score: 4.0
	Standard Dev: 19.435686813743473
	Min Score: 0

Random strategy (5.56 sec total, 0.00556 sec per trial):
	Max Tile: 256
	Max Score: 3144
	Mean Score: 1078.524
	Median Score: 1030.0
	Standard Dev: 532.6747110308239
	Min Score: 176

Down Left strategy (48.54 sec total, 0.04854 sec per trial):
	Max Tile: 512
	Max Score: 7584
	Mean Score: 2294.576
	Median Score: 2122.0
	Standard Dev: 1175.9710132686298
	Min Score: 204

Greedy strategy (58.58 sec total, 0.05858 sec per trial):
	Max Tile: 1024
	Max Score: 12504
	Mean Score: 3091.016
	Median Score: 2970.0
	Standard Dev: 1531.7982522609268
	Min Score: 296

Down left greedy strategy (58.16 sec total, 0.05816 sec per trial):
	Max Tile: 1024
	Max Score: 11688
	Mean Score: 3163.592
	Median Score: 2940.0
	Standard Dev: 1543.4928479613075
	Min Score: 396


Running 1000 trials with Andy impl to test each strategy

Strategy only moves right (2.79 sec total, 0.00279 sec per trial):
	Max Tile: 16
	Max Score: 136.0
	Mean Score: 12.948
	Median Score: 4.0
	Standard Dev: 18.905732535998776
	Min Score: 0.0

Random strategy (54.43 sec total, 0.05443 sec per trial):
	Max Tile: 256
	Max Score: 3268.0
	Mean Score: 1081.156
	Median Score: 1060.0
	Standard Dev: 525.726404050544
	Min Score: 192.0

Down Left strategy (632.05 sec total, 0.63205 sec per trial):
	Max Tile: 512
	Max Score: 7108.0
	Mean Score: 2162.64
	Median Score: 1896.0
	Standard Dev: 1074.8192083930817
	Min Score: 280.0

Greedy strategy (789.69 sec total, 0.78969 sec per trial):
	Max Tile: 1024
	Max Score: 11072.0
	Mean Score: 2980.308
	Median Score: 2778.0
	Standard Dev: 1544.1913353001355
	Min Score: 348.0

Down left greedy strategy (812.7 sec total, 0.8127 sec per trial):
	Max Tile: 1024
	Max Score: 12068.0
	Mean Score: 3061.82
	Median Score: 2882.0
	Standard Dev: 1479.4329026060334
	Min Score: 576.0
```

## Example gameplay output

```
    2     ·     ·     ·
    ·     ·     ·     ·
    2     ·     ·     ·
    4     4     ·     ·


Score: 8
Move (w=UP, d=RIGHT, s=DOWN, a=LEFT, or arrows)?
Move: Left

    2     ·     ·     ·
    2     ·     ·     ·
    2     ·     ·     ·
    8     ·     ·     ·


Score: 16
Move (w=UP, d=RIGHT, s=DOWN, a=LEFT, or arrows)?
Move: Down

    ·     2     ·     ·
    2     ·     ·     ·
    4     ·     ·     ·
    8     ·     ·     ·


Score: 20
Move (w=UP, d=RIGHT, s=DOWN, a=LEFT, or arrows)?
Move: Left

    2     ·     ·     ·
    2     ·     ·     ·
    4     ·     ·     ·
    8     2     ·     ·


Score: 20
Move (w=UP, d=RIGHT, s=DOWN, a=LEFT, or arrows)?
Move: Down

    ·     ·     ·     ·
    4     ·     ·     ·
    4     ·     ·     ·
    8     2     4     ·
```

# Using MLflow
Some of the algorithms (e.g. reinforce.py) use [MLflow](https://mlflow.org/)
to track experiment runs. By default, MLflow writes experiment metadata to the
local filesystem. To visualize the results in the MLflow UI, just
run `mlflow ui` in the terminal and then visit localhost:5000 in your browser.

There you can see the history of runs of the algorithms, including parameters
and metrics associated with each run.
