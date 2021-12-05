import torch as th
import torch.nn as nn
import torch.nn.functional as F

class NoiseRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NoiseRNNAgent, self).__init__()
        self.args = args
        self.name = 'noise_rnn'

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

        self.noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.noise_embedding_dim)
        self.noise_fc2 = nn.Linear(args.noise_embedding_dim, args.noise_embedding_dim)
        self.noise_fc3 = nn.Linear(args.noise_embedding_dim, args.n_actions)

        self.hyper = args.hyper
        self.hyper_noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.rnn_hidden_dim * args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        # return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return self.main_fc_1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, noise_inputs):
        bs = inputs.shape[0]
        # prediction part
        pre_obs = inputs[:, :self.args.pre_obs_shape]
        pre_hidden = self.pre_input(pre_obs)
        pre_skip_1 = pre_hidden
        pre_out_1 = F.relu(self.pre_fc1(pre_hidden))
        pre_out_1 = self.pre_fc2(pre_out_1)
        pre_out_1 += pre_skip_1
        pre_out_1 = F.relu(pre_out_1)

        pre_skip_2 = pre_out_1
        pre_out_2 = F.relu(self.pre_fc3(pre_out_1))
        pre_out_2 = self.pre_fc4(pre_out_2)
        pre_out_2 += pre_skip_2
        pre_out_2 = F.relu(pre_out_2)
        pre_out = self.pre_output(pre_out_2)  ## check: activation

        tot_obs = th.cat((pre_out, inputs), 1)  ## check
        main_hidden = self.main_input(tot_obs)
        main_skip_1 = main_hidden
        main_out_1 = F.relu(self.main_fc_1(main_hidden))
        main_out_1 = self.main_fc_2(main_out_1)
        main_out_1 += main_skip_1
        main_out_1 = F.relu(main_out_1)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.main_rnn(main_out_1, h_in)

        main_skip_2 = h
        main_out_2 = F.relu(self.main_fc_3(h))
        main_out_2 = self.main_fc_4(main_out_2)
        main_out_2 += main_skip_2
        main_out_2 = F.relu(main_out_2)
        q = self.main_output(main_out_2)

        # agent_ids = th.eye(self.args.n_agents).repeat(noise.shape[0], 1)
        # noise_repeated = noise.repeat(1, self.args.n_agents).reshape(agent_ids.shape[0], -1)
        # noise_input = th.cat([noise_repeated, agent_ids], dim=-1)

        if self.hyper:
            W = self.hyper_noise_fc1(noise_inputs).reshape(-1, self.args.n_actions, self.args.rnn_hidden_dim)
            # wq = th.bmm(W, h.unsqueeze(2))
            wq = th.bmm(W, main_out_2.unsqueeze(2))
            wq = wq.view((bs, self.args.n_actions))
        else:
            z = F.tanh(self.noise_fc1(noise_inputs))
            z = F.tanh(self.noise_fc2(z))
            wz = self.noise_fc3(z)

            wq = q * wz

        return wq, h
