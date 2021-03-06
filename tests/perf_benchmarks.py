# requires PYTHONPATH to contain the top-level directory
# i.e. the improved-funicular you checked out from github or run:
# $ PYTHONPATH=. python tests/perf_benchmarks.py
#
# Alternatively, add a pth file to site-packages, for example
# $ echo \`pwd\`/  > improved-funicular/lib/python3.7/site-packages/curr_dir.pth

import time
from envs.nick_2048 import Nick2048
from strategies.random import try_random


board = [2, 0, 8, 16, 2, 4, 8, 4, 2, 0, 2, 2, 4, 0, 0, 0]
game = Nick2048()

start = time.time()

# Initial implementation: .39sec
# With squash lookup table: .22sec
for i in range(10000):
    game.set_board(board)
    game.step(game.UP)

end = time.time()

print(f"Time to set board and step: {end-start}")

start = time.time()
rollouts = 100
# Initial (with squash table): .35sec
try_random(Nick2048, rollouts)
end = time.time()
print(f"Time for {rollouts} random rollouts: {end-start}")
print()
