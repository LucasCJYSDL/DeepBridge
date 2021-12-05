from environment.bridge_env import BridgeEnv
from environment.bridge_utils import *
import random
from copy import deepcopy
from environment.deal import Deal
# random.seed(0)

env = BridgeEnv(debug=True, score_mode='SCORE')
# for _ in range(8):
#     env.reset()
#     print("obs ", env.get_obs())
#     print("state", env.get_state())
env.reset()
cards = deepcopy(FULL_DECK)
random.shuffle(cards)
big_cards, small_cards = [], []
for i in range(len(cards)):
    card = cards[i]
    if card//13 == 0:
        big_cards.append(card)
    else:
        small_cards.append(card)

print(big_cards)
print(small_cards)

predeal = {}
ind = 0
for seat in [0]:
    predeal[seat] = big_cards[ind: ind+len(Rank)]
    ind += len(Rank) # shift the index
ind = 0
for seat in [1,2,3]:
    predeal[seat] = small_cards[ind: ind + len(Rank)]
    ind += len(Rank)  # shift the index
env.deal = Deal.prepare(predeal)
convert_hands2string(env.deal)
env.step(35)
env.step(35)
env.step(0)
env.step(10)
env.step(35)
env.step(35)
env.step(36)
env.step(35)
env.step(35)
env.step(37)
env.step(35)
env.step(35)
env.step(34)
env.step(36)
env.step(37)
env.step(35)
# print("obs ", env.get_obs()[:52])
env.step(35)
env.step(35)
# print("state", env.get_state()[:104])
print("action history: ", env.get_action_history())
print("score: ", env.get_board_reward())

