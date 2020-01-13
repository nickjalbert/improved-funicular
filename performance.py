import time
from nick_2048 import Nick2048


board = [2, 0, 8, 16, 2, 4, 8, 4, 2, 0, 2, 2, 4, 0, 0, 0]
game = Nick2048()

start = time.time()

# Initial implementation: .39sec
# With squash lookup table: .22sec
for i in range(10000):
    game.set_board(board)
    game.step(game.UP)

end = time.time()

print(f"Time: {end-start}")
