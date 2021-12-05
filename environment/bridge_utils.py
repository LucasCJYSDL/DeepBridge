from collections import defaultdict
from environment.config import *
import numpy as np


def convert_id2suit(card_id):
    return card_id//13


def convert_id2rank(card_id):
    return card_id%13+2


def convert_hands2string(deal):
    for seat, hand in deal.items():
        # assert seat in Seat, seat
        holding = defaultdict(list)
        for card in hand:
            suit = Suit2str[convert_id2suit(card)]
            rank = Rank2str[convert_id2rank(card)]
            holding[suit].append(rank)

        print("Seat ", Seat2str[seat],":")
        for suit in Suit2str.values():
            print(suit, " ".join(holding[suit]))


def one_hot_holding(holding): ##check
    one_hot_res = np.zeros(len(FULL_DECK), dtype=np.float32)
    one_hot_res[holding] = 1
    return one_hot_res

# 0: "S", 1: "H", 2: "D", 3: "C", 4: "N"
# Bid C D H S

def convert_action2strain(action):
    bid_strain = action%(len(Strain))
    return {3:0, 2:1, 1: 2, 0: 3, 4: 4}[bid_strain]


def convert_action2level(action):
    return action//len(Strain) + 1


def log_state(done, info):

    print("Finished: ", done)
    print("Whose Turn: ", info["turn"],'\n')





