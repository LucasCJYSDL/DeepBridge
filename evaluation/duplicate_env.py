'''
compete with Wbridge5 in a duplicate bridge game,
the rules for reference: https://www.acbl.org/learn_page/how-to-play-bridge/how-to-keep-score/duplicate/
'''

from environment.config import *
from copy import deepcopy
import random
import pickle
from environment.bridge_env import BridgeEnv
from environment.bridge_utils import *
from environment.deal import Deal
from environment.scoring import convert2IMP
import numpy as np

def make_dst(board_num=1600):
    assert board_num%16==0
    cards = deepcopy(FULL_DECK)
    seats = [0, 1, 2, 3]
    board_info = []
    for i in range(board_num):
        random.shuffle(cards)
        pre_deal = {}
        ind = 0
        for seat in seats:
            pre_deal[seat] = cards[ind:(ind+len(Rank))]
            ind += len(Rank)
        ##http: // web2.acbl.org / documentlibrary / play / Laws - of - Duplicate - Bridge.pdf: see LAW 2 in P 7
        dealer = i % 4 # who will bid first
        temp_idx = i % 16
        if temp_idx in [0, 7, 10, 13]:
            ns_vul = 0
            ew_vul = 0
        elif temp_idx in [1, 4, 11, 14]:
            ns_vul = 1
            ew_vul = 0
        elif temp_idx in [2, 5, 8, 15]:
            ns_vul = 0
            ew_vul = 1
        else:
            assert temp_idx in [3, 6, 9, 12]
            ns_vul = 1
            ew_vul = 1
        board_info.append({"board_id": i, "predeal": pre_deal, "dealer": dealer, "ns_vul": ns_vul, "ew_vul": ew_vul})

    with open('test_boards.pkl', 'wb') as f:
        pickle.dump(board_info, f)

class DuplicateEnv(BridgeEnv):

    def __init__(self, board_num):
        #assert board_num%16 == 0, "The board number should be the multiple of 16, in order to be fair!"
        super().__init__(score_mode='SCORE')

        self.cards = None
        self.first_bidder = None

        self.board_num = board_num
        with open('test_boards.pkl', 'rb') as f:
            self.board_info = pickle.load(f)
        assert board_num <= len(self.board_info), "The board number should be fewer than {}!".format(len(self.board_info))
        start_idxes = len(self.board_info) // board_num
        self.start_idx = np.random.randint(0, start_idxes)
        self.board_info = self.board_info[self.start_idx*board_num: self.start_idx*board_num+board_num]
        self.board_id = -1
        self.table_id = None ## table 0 or table 1

    def reset(self):
        self.board_id += 1
        self.table_id = self.board_id // self.board_num
        print("##########################Test Board {} at Table {}##########################".format(self.board_id, self.table_id))
        current_board = self.board_info[self.board_id%self.board_num]
        # self.current_board = current_board
        predeal = current_board['predeal']
        self.one_hot_deal = np.zeros((len(Seat), len(FULL_DECK)), dtype=np.uint8)
        for seat in self.bidding_seats:
            self.one_hot_deal[seat] = one_hot_holding(predeal[seat]) # one hot cards
        self.deal = Deal.prepare(predeal)

        self.first_bidder = current_board['dealer']
        self.turn = self.first_bidder

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
        ns_vul = current_board['ns_vul']
        ew_vul = current_board['ew_vul']
        self.vulner_sign = {0: ns_vul, 1: ew_vul}
        self.avai_action = self.update_avai_action()
        self.action_history = []

    def get_board_score(self):
        ## Table 0: N-S: NN, E-W: Wbridge5; Table 1: N-S: Wbridge5, E-W: NN.
        assert self.done == True
        if self.table_id == self.group_declarer:
            return self.score
        return -self.score

    def convert_to_IMP(self, score_list_0, score_list_1):
        # delta_IMP = 0
        IMP_list = []
        board_num = len(score_list_0)
        assert len(score_list_1) == board_num
        #assert board_num%16 == 0, "The board number should be the multiple of 16, in order to be fair!"
        for i in range(board_num):
            total_score = score_list_0[i] + score_list_1[i]
            IMP_list.append(convert2IMP(total_score))
            print("board:", " ", (i+1), " ", score_list_0[i], " ", score_list_1[i], " ", convert2IMP(total_score))
        #     delta_IMP += convert2IMP(total_score)
        # average_IMP = delta_IMP/float(board_num)
        IMP_list = np.array(IMP_list)
        average_IMP = IMP_list.mean()
        std_IMP = IMP_list.std()
        return average_IMP, std_IMP

    def get_board_id(self):

        return self.board_id

    def get_dealer(self):

        return self.first_bidder

    def get_vulner(self):
        ## 0: neither; 1: N-S; 2: E-W; 3: both;
        if self.vulner_sign[0] == 0:
            if self.vulner_sign[1] == 0:
                return 0
            else:
                return 2
        else:
            if self.vulner_sign[1] == 0:
                return 1
            else:
                return 3

    def is_terminal(self):

        return self.done

if __name__ == '__main__':

    # make_dst()
    # with open('test_boards.pkl', 'rb') as f:
    #     board_info = pickle.load(f)
    # for i in range(160, 176):
    #     print(i, "\n", board_info[i])
    # print(board_info[1558])
    for i in range(1):
        test = DuplicateEnv(160)
        print(test.start_idx)
        print(test.board_info)
        print(len(test.board_info))
        for j in range(320):
            test.reset()
            print(test.current_board)
