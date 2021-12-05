from environment.deal import Deal
from copy import deepcopy
from environment.config import *
import random
from environment.bridge_utils import *


class BridgeEnv(object):
    def __init__(self, score_mode="SCORE", bidding_seats=[0,1,2,3], debug=False):
        self.deal = None
        self.one_hot_deal = None
        self.cards = deepcopy(FULL_DECK)
        ### state
        self.bid_history = None
        self.double_sign = None
        self.vulner_sign = None
        self.avai_action = None

        self.n_pass = 0
        self.n_double = 0
        self.n_redouble = 0

        self.max_bid = -1
        self.done = False
        self.debug = debug
        self.score_mode = score_mode

        self.strain_declarer = None
        self.group_declarer = -1
        self.bidding_seats = sorted(list(set(bidding_seats)))
        self.num_player = len(self.bidding_seats)
        for seat in self.bidding_seats:
            if seat not in Seat:
                raise Exception("illegal seats")
        self.first_bidder = self.bidding_seats[0]
        self.turn = None
        self.score = None
        self.action_history = None

    def reset(self):  # North and South
        ### generate the hands
        random.shuffle(self.cards)
        ind = 0
        predeal = {}
        self.one_hot_deal = np.zeros((len(Seat), len(FULL_DECK)), dtype=np.uint8)
        for seat in self.bidding_seats:
            predeal[seat] = self.cards[ind: ind+len(Rank)]
            self.one_hot_deal[seat] = one_hot_holding(predeal[seat]) # one hot cards
            ind += len(Rank) # shift the index
        self.deal = Deal.prepare(predeal)

        self.turn = self.first_bidder
        self.first_bidder = (self.first_bidder+1)%self.num_player

        self.max_bid = -1
        self.n_pass = 0
        self.n_double = 0
        self.n_redouble = 0
        self.done = False
        self.score = None
        self.strain_declarer = {0: {}, 1: {}}
        self.group_declarer = -1

        ### state
        self.bid_history = {}
        for seat in self.bidding_seats:
            self.bid_history[seat] = np.zeros(35, dtype=np.uint8)
        self.double_sign = np.zeros(35, dtype=np.uint8)
        ns_vul = np.random.randint(0,2)
        ew_vul = np.random.randint(0,2)
        self.vulner_sign = {0: ns_vul, 1: ew_vul}
        self.avai_action = self.update_avai_action()
        self.action_history = []

        if self.debug:
            print("1 ", self.one_hot_deal)
            # print("2 ", self.deal)
            convert_hands2string(self.deal)
            # print("3 ", self.turn)
            print("4 ", self.vulner_sign)
            # print("5 ", self.avai_action)

    def step(self, action):
        """
        :param action: bid action
        :return: state, reward, done
        """
        if self.done:
            raise Exception("No more actions can be taken")

        if (action < 0) or (action > 37) or (self.avai_action[action] != 0):
            raise Exception("illegal action")

        if action <= 34:
            assert self.bid_history[self.turn][action] == 0
            self.bid_history[self.turn][action] = 1
            self.n_pass = 0
            self.n_double = 0
            self.n_redouble = 0
            assert action > self.max_bid
            self.max_bid = action
            strain = convert_action2strain(action)
            group = Seat2Group[self.turn]
            if self.strain_declarer[group].get(strain, '') == '': ##check
                self.strain_declarer[group][strain] = self.turn # which one
            self.group_declarer = group # which group

        elif action == 35: # PASS
            self.n_pass += 1

        elif action == 36:
            self.n_pass = 0
            assert (Seat2Group[self.turn] != self.group_declarer) and (self.double_sign[self.max_bid] == 0)
            self.n_double += 1
            self.double_sign[self.max_bid] = 1

        else:
            self.n_pass = 0
            assert (self.n_double == 1) and (self.double_sign[self.max_bid] == 1) and (Seat2Group[self.turn] == self.group_declarer)
            self.n_redouble += 1
            self.double_sign[self.max_bid] = 2

        self.action_history.append({self.turn: action})

        self.turn = (self.turn+1) % self.num_player  # loop
        self.avai_action = self.update_avai_action()
        # while True:  # move to the participant
        #     if self.turn not in self.bidding_seats:
        #         self.turn = (self.turn+1) % len(Seat)
        #         self.n_pass += 1
        #     else:
        #         break
        # state is the next bidding player's state
        if (self.n_pass == 3) and (self.max_bid >= 0):
            # extract the declarer, strain , level
            strain = convert_action2strain(self.max_bid)
            level = convert_action2level(self.max_bid)
            # parallel threads
            # np.mean is moved to score
            declarer = self.strain_declarer[self.group_declarer][strain] # this group's first declarer of the certain strain
            assert declarer in self.bidding_seats
            is_double = self.double_sign[self.max_bid]
            vulnerability = self.vulner_sign[self.group_declarer]
            print("The deal is: ", self.deal)
            self.score = Deal.score(dealer=self.deal, level=level, strain=strain, declarer=declarer, is_double=is_double,\
                                vulner=vulnerability, tries=1, mode=self.score_mode)
            self.done = True

        elif self.n_pass == 4:
            assert self.max_bid < 0
            self.score = 0.0
            self.done = True

        info = {"turn": self.turn, "max_bid": self.max_bid}
        if self.debug:
            # print("5 ", self.turn)
            print("6 ", self.strain_declarer)
            # print("7 ", self.group_declarer)
            # print("8 ", self.max_bid)
            # print("9 ", self.n_pass)
            # print("10 ", self.n_double)
            # print("11 ", self.n_redouble)
            # print("12 ", self.double_sign)
            # print("13 ", self.avai_action)
            # print("14 ", self.bid_history)
            log_state(self.done, info)
            if self.done:
                print("15 ", strain, " ", level, " ", declarer)
        return self.done, info


    def update_avai_action(self):
        avai_action = np.zeros(38, dtype=np.uint8) #0~34 normal bid, 35 pass, 36 double, 37 redouble; 0 allowed, 1 not allowed;
        for i in range(35):
            if i <= self.max_bid:
                avai_action[i] = 1
        # if (self.max_bid < 0) and (self.n_pass == 2):
        #     avai_action[35] = 1
        cur_group = Seat2Group[self.turn]
        if (self.max_bid < 0) or (self.n_double != 0) or (cur_group == self.group_declarer): ##check
            avai_action[36] = 1
        if (self.n_double == 0) or (self.n_redouble != 0) or (cur_group != self.group_declarer): ##check
            avai_action[37] = 1
        return avai_action


    def get_obs(self):
        own_card = self.one_hot_deal[self.turn]
        vulnerability = np.zeros(2, dtype=np.uint8)
        group = Seat2Group[self.turn]
        vulnerability[0] = self.vulner_sign[group]
        vulnerability[1] = self.vulner_sign[1-group]

        own_bid_hist = self.bid_history[self.turn]
        part_bid_hist = self.bid_history[(self.turn+2)%self.num_player]
        left_bid_hist = self.bid_history[(self.turn+1)%self.num_player]
        right_bid_hist = self.bid_history[(self.turn+3)%self.num_player]
        obs = np.concatenate((own_card, own_bid_hist, part_bid_hist, left_bid_hist, right_bid_hist, self.double_sign, \
                              vulnerability, self.avai_action))
        # obs = np.concatenate((own_card, self.bid_history[0], self.bid_history[1], self.bid_history[2], self.bid_history[3],\
        #                       self.double_sign, vulnerability, self.avai_action))
        assert obs.shape[0] == 267, obs.shape[0]
        return obs


    def get_state(self):
        cur_group = Seat2Group[self.turn]
        if cur_group==0:
            card_1 = self.one_hot_deal[0]
            card_2 = self.one_hot_deal[2]
        else:
            card_1 = self.one_hot_deal[1]
            card_2 = self.one_hot_deal[3]
        vulnerability = np.zeros(2, dtype=np.uint8)
        vulnerability[0] = self.vulner_sign[0]
        vulnerability[1] = self.vulner_sign[1]
        state = np.concatenate((card_1, card_2, self.bid_history[0], self.bid_history[1], self.bid_history[2], \
                                self.bid_history[3], self.double_sign, vulnerability))
        assert state.shape[0] == 281, state.shape[0]
        return state


    def get_avai_action(self):

        return self.avai_action

    def get_avai_action_num(self):
        avai_action_num = int(self.avai_action.size - self.avai_action.sum())
        # print("1: ", self.avai_action, " ", avai_action_num)
        return avai_action_num

    def get_action_history(self):

        return self.action_history


    def get_board_reward(self):
        assert self.done == True
        if self.group_declarer == 0:
            reward = self.score
        else:
            reward = -self.score
        return reward/NORM

    def get_least_reward(self):
        # according to the scoring sheet: http://web2.acbl.org/documentLibrary/play/InstantScorer.pdf
        return -7600/NORM


    def get_env_info(self):
        env_info = {}
        env_info['n_actions'] = 38
        env_info['n_agents'] = 2
        env_info['n_cards'] = 52
        env_info['state_shape'] = 281
        env_info['obs_shape'] = 267
        env_info['pre_obs_shape'] = 229
        env_info['episode_limit'] = 15 ###danger; round number; round: N-S-E-W
        env_info['state_hands_shape'] = 2*52 + 2
        return env_info


    def get_turn(self):

        return self.turn

    def get_current_deal(self):

        return {self.turn: self.deal[self.turn]}

    def get_current_team_deal(self):
        team_deal = {}
        if (self.turn == 0) or (self.turn == 2):
            team_deal[0] = self.deal[0]
            team_deal[2] = self.deal[2]
        else:
            team_deal[1] = self.deal[1]
            team_deal[3] = self.deal[3]
        # print("team_deal: ", team_deal)
        return team_deal

    def set_turn(self, agent_id):

        self.turn = Agent2Seat[agent_id]

    def get_player_num(self):

        return self.num_player

    def get_vulner_sign(self):

        return self.vulner_sign

    def is_target_group(self, turn=None):
        if turn:
            assert turn in self.bidding_seats
            temp_group = Seat2Group[turn]
        else:
            temp_group = Seat2Group[self.turn]
        return (temp_group == 0)

    def get_agent_id(self):

        return Seat2Agent[self.turn]

    def get_make_up_state(self):
        # self.turn = Agent2Seat[agent_id]
        make_up_state = self.get_state()
        return make_up_state

    def get_make_up_obs(self):
        # self.turn = Agent2Seat[agent_id]
        make_up_obs = self.get_obs()
        return make_up_obs

    def get_make_up_action(self):

        return 35 ## pass

    def is_terminal(self):

        return self.done

    def get_state_hands(self):
        vulnerability = np.zeros(2, dtype=np.uint8)
        vulnerability[0] = self.vulner_sign[0]
        vulnerability[1] = self.vulner_sign[1]

        target_state_hands = np.concatenate((self.one_hot_deal[0], self.one_hot_deal[2], vulnerability))
        oppo_state_hands = np.concatenate((self.one_hot_deal[1], self.one_hot_deal[3], vulnerability))

        return target_state_hands, oppo_state_hands


