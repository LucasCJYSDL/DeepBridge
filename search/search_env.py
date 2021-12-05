from environment.bridge_env import *


class SearchEnv(BridgeEnv):

    def __init__(self):
        super().__init__()
        self.cards = None
        self.first_bidder = None

    def reset_from(self, deal_dict, vulner_dict, action_history, init):

        predeal = deal_dict.copy()
        self.one_hot_deal = np.zeros((len(Seat), len(FULL_DECK)), dtype=np.uint8)
        for seat in self.bidding_seats:
            self.one_hot_deal[seat] = one_hot_holding(predeal[seat])
        self.deal = Deal.prepare(predeal)

        self.turn = list(action_history[0].keys())[0]
        self.max_bid = -1
        self.n_pass = 0
        self.n_double = 0
        self.n_redouble = 0
        self.done = False
        self.score = None
        self.strain_declarer = {0: {}, 1: {}}
        self.group_declarer = -1

        self.bid_history = {}
        for seat in self.bidding_seats:
            self.bid_history[seat] = np.zeros(35, dtype=np.uint8)
        self.double_sign = np.zeros(35, dtype=np.uint8)
        self.vulner_sign = vulner_dict.copy()
        self.avai_action = self.update_avai_action()
        self.action_history = []

        ## reset from the action history
        if not init:
            for action_pair in action_history:
                turn = list(action_pair.keys())[0]
                action = list(action_pair.values())[0]
                assert turn == self.turn
                self.step(action)

    def sample_particle(self, current_player, known_deal):
        temp_known_deal = known_deal.copy()
        particle = {}
        known_cards = []
        assert current_player in temp_known_deal.keys()
        for player, card in temp_known_deal.items():
            known_cards.extend(card)
            particle[player] = card
        # print("known cards", known_cards)
        cards = deepcopy(FULL_DECK)
        left_cards = list(set(cards) - set(known_cards))
        random.shuffle(left_cards)
        ind = 0
        for seat in self.bidding_seats:
            if seat in temp_known_deal.keys():
                continue
            particle[seat] = left_cards[ind: ind+len(Rank)]
            ind += len(Rank) # shift the index
        return particle


if __name__ == '__main__':

    test_env = SearchEnv()
    own_deal = [0, 1, 2, 3, 5, 6, 7, 10, 9, 40, 25, 8, 16]
    partner_deal = [39, 41, 42, 43, 44, 45, 46, 50, 51, 29, 30, 31, 32]
    known_deal = {0: own_deal, 2: partner_deal}
    for i in range(10):
        particle = test_env.sample_particle(0, known_deal)
        print(particle)