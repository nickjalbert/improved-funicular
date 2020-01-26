# improved-funicular

[![Actions Status](https://github.com/nickjalbert/improved-funicular/workflows/Python%20Lint%20and%20Test/badge.svg)](https://github.com/nickjalbert/improved-funicular/actions)

Python 3 implementation of the [2048 game](https://play2048.co/).

## Performance

Here's how various 2048 strategies perform
(tested on a 2012 macbook air with a 1.8GHz i5 and 4GB of 1600MHz DDR3 RAM
and run using `python do_stats.py nick 100 [strategy_name]`
):

| Strategy Name            | Max tile | Max Score | Mean Score    | Score Standard Dev | Mean steps per game | Mean sec per game    | Git Hash   |
| ------------------------ | -------: | --------: | ------------: | -----------------: |-------------------: | -------------------: | --------:  |
| `only_go_right`          |   16     |   140     |    13         |     18             |    6                |     .00005 sec       | b51ed7d    |
| `random`                 |  256     |  3064     |  1058         |    504             |  138                |     .006   sec       | b51ed7d    |
| `down_left`              |  512     |  7228     |  2332         |   1131             |  207                |     .05    sec       | b51ed7d    |
| `fixed_action_order`     |  512     |  4864     |  2492         |   1345             |  221                |     .05    sec       | b51ed7d    |
| `greedy`                 |  512     |  5404     |  2122         |    930             |  188                |     .05    sec       | b51ed7d    |
| `greedy_fixed_order`     | 1024     | 11376     |  3028         |   1580             |  256                |     .06    sec       | b51ed7d    |
| `down_left_greedy`       |  512     |  7084     |  2107         |   1076             |  192                |     .05    sec       | b51ed7d    |
| `max_space_then_greedy`  | 1024     | 12320     |  3157         |   1477             |  266                |     .06    sec       | b51ed7d    |
| `lookahead_1`            | 1024     | 11940     |  3008         |   1864             |  252                |     .1     sec       | b51ed7d    |
| `lookahead_2`            | 1024     | 15428     |  7446         |   3157             |  491                |     .7     sec       | b51ed7d    |
| `lookahead_3`            | 2048     | 25868     | 11659         |   3958             |  724                |    5.1     sec       | b51ed7d    |
| `lookahead_4`            | 2048     | 35500     | 16757         |   7443             |  945                |   33.1     sec       | 02b412d    |
| `lookahead_with_rollout`<sup>1</sup> | 4096 | 75832 | 36670     |  14124             | 1853                |  758       sec       | c3bb225    |

1. Lookahead 3 that switches to 150 random rollouts per move when board has <= 7 empty spaces

## Setup

Developed under Python 3.7 on OS X Catalina with pip and virtualenv installed
on the system.  Look [here](https://stackoverflow.com/a/23842752) to setup
Python 3.7.

* `git clone https://github.com/nickjalbert/improved-funicular.git`
* `cd improved-funicular`
* `virtualenv improved-funicular`
* `echo \`pwd\`/  > improved-funicular/lib/python3.7/site-packages/curr_dir.pth`
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
