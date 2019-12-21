# improved-funicular

[![Actions Status](https://github.com/nickjalbert/improved-funicular/workflows/python_lint_and_test/badge.svg)](https://github.com/nickjalbert/improved-funicular/actions)

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

## Tests

To run Andy's tests:

* `pytest andy_2048.py`

## Example strategy trials

Run `python do_stats.py` to try several different strategies.

```
$ python do_stats.py

Running 1000 trials with Nick impl to test each strategy

Strategy only moves right:
	Max Tile: 16
	Max Score: 184
	Mean Score: 12.896
	Median Score: 4.0
	Standard Dev: 19.793457630362738
	Min Score: 0

Random strategy:
	Max Tile: 256
	Max Score: 3024
	Mean Score: 1081.916
	Median Score: 1052.0
	Standard Dev: 510.36060672126024
	Min Score: 208

Down Left strategy:
	Max Tile: 512
	Max Score: 7636
	Mean Score: 2232.48
	Median Score: 1966.0
	Standard Dev: 1178.3934405370308
	Min Score: 268


Running 1000 trials with Andy impl to test each strategy

Strategy only moves right:
	Max Tile: 16
	Max Score: 172.0
	Mean Score: 13.724
	Median Score: 4.0
	Standard Dev: 21.576407610289653
	Min Score: 0.0

Random strategy:
	Max Tile: 256
	Max Score: 3268.0
	Mean Score: 1094.132
	Median Score: 1062.0
	Standard Dev: 541.5182883639598
	Min Score: 120.0

Down Left strategy:
	Max Tile: 512
	Max Score: 7496.0
	Mean Score: 2245.812
	Median Score: 1942.0
	Standard Dev: 1184.6117497533332
	Min Score: 188.0
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
