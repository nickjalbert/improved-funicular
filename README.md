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

## Tests

To run Andy's tests:

* `pytest andy_2048.py`

## Example strategy trials

Run `python do_stats.py` to try several different strategies.

```
$ python do_stats.py

Running 1000 trials to test each strategy

Strategy only moves right:
	Max Score: 172
	Mean Score: 14.028
	Median Score: 4.0
	Standard Dev: 20.92780899160706
	Min Score: 0

Random strategy:
	Max Score: 3088
	Mean Score: 1059.188
	Median Score: 1012.0
	Standard Dev: 519.918555121819
	Min Score: 148

Down Left strategy:
	Max Score: 7912
	Mean Score: 2101.34
	Median Score: 1782.0
	Standard Dev: 1100.5140651566464
	Min Score: 240
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
