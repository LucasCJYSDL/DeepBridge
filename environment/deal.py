from environment.dds import solve_all
from collections import defaultdict
from copy import deepcopy
from environment.config import *
from environment.scoring import SCORE_TABLE

class Deal(object):

    @classmethod
    def prepare(cls, predeal=None): ##check
        predeal = {} if predeal is None else predeal.copy()
        dealer = defaultdict(list)
        for seat in Seat:
            pre = predeal.pop(seat)
            if isinstance(pre, list):
                dealer[seat] = pre
            else:
                raise Exception("Wrong format of predeal")

        if predeal:
            raise Exception("Unused predeal entries: {}".format(predeal))

        predealt = [card for hand_cards in dealer.values()
                    for card in hand_cards]
        predealt_set = set(predealt)
        if len(predealt_set) < len(predealt):
            raise Exception("Same card dealt twice.")
        # dealer["_remaining"] = [card for card in FULL_DECK
        #                         if card not in predealt_set]
        return dealer  # a dictionary

    @classmethod
    def score(cls, dealer, level, strain, declarer, is_double, vulner, tries, mode=None):
        # target = level + 6

        dealers = []
        declarers = [declarer] * tries
        strains = [strain] * tries

        for _ in range(tries):
            tmp_dealer = deepcopy(dealer)
            # cards = dealer["_remaining"]
            # random.shuffle(cards)
            # for seat in Seat:
            #     to_deal = len(Rank) - len(tmp_dealer[seat])
            #     tmp_dealer[seat] += cards[:to_deal]
            #     cards = cards[to_deal:]
            dealers.append(tmp_dealer)
        max_tricks = solve_all(dealers, strains, declarers) ##check
        print("max tricks of the declarer group: ", max_tricks)
        if mode == "SCORE":
            # print("level: {}; starin: {}; trick: {}; vulner: {}; double: {}".format(level, strain, max_tricks[0], vulner, is_double))
            scores = [SCORE_TABLE[(level, strain, trick, vulner, is_double)] for trick in max_tricks]
        else:
            raise NotImplementedError
        return np.mean(scores)



