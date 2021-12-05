import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    # input_shape = obs_shape + n_actions + n_agents
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.name = 'rnn'
        ## prediction part
        self.pre_input = nn.Linear(args.pre_obs_shape, args.pre_hidden_dim)
        self.pre_fc1 = nn.Linear(args.pre_hidden_dim, args.pre_hidden_dim)
        self.pre_fc2 = nn.Linear(args.pre_hidden_dim, args.pre_hidden_dim)
        self.pre_fc3 = nn.Linear(args.pre_hidden_dim, args.pre_hidden_dim)
        self.pre_fc4 = nn.Linear(args.pre_hidden_dim, args.pre_hidden_dim)
        self.pre_output = nn.Linear(args.pre_hidden_dim, args.n_cards)
        ## main part
        self.main_input = nn.Linear(args.n_cards + input_shape, args.rnn_hidden_dim)
        # self.main_input = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.main_fc_1 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.main_fc_2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.main_rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.main_fc_3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.main_fc_4 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.main_output = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on the same device and with the same type
        return self.main_fc_1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        # prediction part
        pre_obs = obs[:, :self.args.pre_obs_shape]

        pre_hidden = self.pre_input(pre_obs)
        pre_skip_1 = pre_hidden
        pre_out_1 = F.relu(self.pre_fc1(pre_hidden))
        pre_out_1 = self.pre_fc2(pre_out_1)
        pre_out_1 += pre_skip_1
        pre_out_1 = F.relu(pre_out_1)

        pre_skip_2 = pre_out_1
        # print("1: ", pre_skip_2[0])
        pre_out_2 = F.relu(self.pre_fc3(pre_out_1))
        pre_out_2 = self.pre_fc4(pre_out_2)
        # print("2: ", pre_skip_2[0])
        # print("3: ", pre_out_2[0])
        pre_out_2 += pre_skip_2
        # print("4: ", pre_out_2[0])
        pre_out_2 = F.relu(pre_out_2)
        # print("5: ", pre_out_2[0])
        pre_out = self.pre_output(pre_out_2) ## check: activation
        ## main part

        tot_obs = torch.cat((pre_out, obs), 1) ## check
        # print("6: ", tot_obs.size())
        main_hidden = self.main_input(tot_obs)
        # main_hidden = self.main_input(obs)
        main_skip_1 = main_hidden
        main_out_1 = F.relu(self.main_fc_1(main_hidden))
        main_out_1 = self.main_fc_2(main_out_1)
        main_out_1 += main_skip_1
        main_out_1 = F.relu(main_out_1)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # print("7: ", h_in.size())
        h = self.main_rnn(main_out_1, h_in)

        main_skip_2 = h
        main_out_2 = F.relu(self.main_fc_3(h))
        main_out_2 = self.main_fc_4(main_out_2)
        main_out_2 += main_skip_2
        main_out_2 = F.relu(main_out_2)
        q = self.main_output(main_out_2)

        return q, h
