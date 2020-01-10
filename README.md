# improved-funicular

[![Actions Status](https://github.com/nickjalbert/improved-funicular/workflows/Python%20Lint%20and%20Test/badge.svg)](https://github.com/nickjalbert/improved-funicular/actions)

Python 3 implementation of the [2048 game](https://play2048.co/).

## Performance

Here's how various 2048 strategies perform
(based on playing 100 games using Nick's
implementation with the named strategy,
tested on a 2012 macbook air with 1.8GHz i5 and 4GB of 1600MHz DDR3 RAM):

| Strategy           | Max observed tile | Mean Score    | Score Standard Dev | Mean steps per game | Mean sec per game  |
| -----------------  | ----------------: | ------------: | -----------------: |-------------------: | -----------------: |
| Only Go Right      | 16                | 13            | 18                 | 6                   |  .00005 sec        |
| Random             | 256               | 1058          | 504                | 138                 |  .006 sec          |
| Down Left          | 512               | 2332          | 1131               | 207                 |  .05 sec           |
| Fixed order        | 512               | 2492          | 1345               | 221                 |  .05 sec           |
| Greedy             | 512               | 2122          | 930                | 188                 |  .05 sec           |
| Greedy Fixed Order | 512               | 3028          | 1580               | 256                 |  .06 sec           |
| Down Left Greedy   | 512               | 2107          | 1076               | 192                 |  .05 sec           |
| Max Space Greedy   | 512               | 3157          | 1477               | 266                 |  .06 sec           |
| Lookahead 1        | 1024              | 3008          | 1864               | 252                 |  .1 sec            |
| Lookahead 2        | 1024              | 7446          | 3157               | 491                 |  .7 sec            |
| Lookahead 3        | 1024              | 11659         | 3958               | 710                 | 5.1 sec            |


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

Use `do_stats.py` to try different 2048 game strategies.  Run:

```python do_stats.py```

for detailed usage.  An example running 100 trials with Nick's implementation
of a random strategy:

```python do_stats.py nick 100 random```


## Manually play 2048

Use `python play_in_console.py nick` to manually play 2048 in the console.

Use `python play_in_console.py lookahead` to play with suggested moves from
the lookahead strategies.

## Using MLflow

Some of the algorithms (e.g. reinforce.py) use [MLflow](https://mlflow.org/)
to track experiment runs. By default, MLflow writes experiment metadata to the
local filesystem. To visualize the results in the MLflow UI, just
run `mlflow ui` in the terminal and then visit localhost:5000 in your browser.

There you can see the history of runs of the algorithms, including parameters
and metrics associated with each run.
