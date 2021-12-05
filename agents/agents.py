# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
from network.actors import REGISTRY as actor_REGISTRY
from torch.distributions import Categorical

class Agents:
    def __init__(self, args, is_target):
        self.args = args
        self.n_agents = args.n_agents
        self.target = is_target

        input_shape = self.args.obs_shape
        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.args.n_agents
        self.actor = actor_REGISTRY[self.args.actor](input_shape, self.args)
        if args.cuda:
            self.actor.cuda()
        self.hidden_states = None

    def is_target(self):
        return self.target

    def get_actor_name(self):
        return self.actor.name

    def init_hidden(self, batch_size):
        self.hidden_states = self.actor.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def save_models(self, path):
        torch.save(self.actor.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        # self.actor.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load("{}/agent.th".format(path)))

    def _get_output(self, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False, noise=None):
        # agent index
        onehot_agent_idx = np.zeros(self.n_agents)
        onehot_agent_idx[agent_idx] = 1.
        if self.args.last_action:
            obs = np.hstack((obs, last_action))
        if self.args.reuse_network:
            obs = np.hstack((obs, onehot_agent_idx))
        hidden_state = self.hidden_states[:, agent_idx, :]
        obs = torch.Tensor(obs).unsqueeze(0)  ## check
        avail_actions_mask = torch.Tensor(avail_actions_mask).unsqueeze(0)  ## check

        if noise is not None:
            assert self.args.alg == 'maven'
            noise = np.hstack((noise, onehot_agent_idx))
            noise = torch.Tensor(noise).unsqueeze(0)

        if self.args.cuda:
            obs = obs.cuda()
            hidden_state = hidden_state.cuda() ## check: may be not required
            if noise is not None:
                noise = noise.cuda()

        if noise is not None:
            actor_output, self.hidden_states[:, agent_idx, :] = self.actor(obs, hidden_state, noise)
        else:
            actor_output, self.hidden_states[:, agent_idx, :] = self.actor(obs, hidden_state)

        if self.args.agent_output_type == "q_value":
            # mask out
            # qsa[avail_actions_mask == 1.0] = -float("inf")  ##check
            actor_output[avail_actions_mask == 1.0] = -99999999.0
            qsa_array = actor_output.clone().detach().cpu().numpy()

            ## convert the q_value to probability
            if evaluate:
                temperature = self.args.boltzmann_coe * self.args.min_epsilon
            else:
                temperature = self.args.boltzmann_coe * epsilon
            boltzmann = np.exp(qsa_array / temperature)
            prob = boltzmann / np.expand_dims(boltzmann.sum(axis=1), axis=1)

        elif self.args.agent_output_type == "pi_logits":
            actor_output = torch.nn.functional.softmax(actor_output, dim=-1) ## check
            if not evaluate: ## fine tune: with or without the noise
                epsilon_action_num = actor_output.size(-1)
                actor_output = ((1 - epsilon) * actor_output + torch.ones_like(actor_output) * epsilon / epsilon_action_num)
            actor_output[avail_actions_mask == 1.0] = 0.0
            prob = actor_output.clone().detach().cpu().numpy() ## check

        else:
            raise NotImplementedError

        return actor_output, prob

    def choose_action(self, obs, last_action, agent_idx, avail_actions_mask, epsilon, replay_buffer, evaluate=False, noise=None):

        # available actions
        avail_actions = np.nonzero(1 - avail_actions_mask)[0]
        actor_output, prob = self._get_output(obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate, noise)

        if evaluate:
            return np.argmax(actor_output.clone().detach().cpu().numpy())

        if self.args.exploration == 'epsilon':
            assert self.args.agent_output_type == "q_value"
            if np.random.uniform() < epsilon:
                return np.random.choice(avail_actions)
            else:
                return np.argmax(actor_output.clone().detach().cpu().numpy())

        elif self.args.exploration == 'ucb1':
            assert self.args.agent_output_type == "q_value"
            temp_qsa = [actor_output[0][i] + self.args.ucb_coe * math.sqrt(2*math.log(replay_buffer.get_T())/replay_buffer.get_TA(i)) for i in range(self.args.n_actions)]
            return np.argmax(temp_qsa)

        elif self.args.exploration == 'boltzmann':
            assert self.args.agent_output_type == "q_value"
            cumProb_boltzmann = np.cumsum(prob, axis=1)
            cb = cumProb_boltzmann[0]
            try:
                act = np.where(cb > np.random.rand(1))[0][0]
            except Exception:
                print("Index error occurs!!!") ## check
                print("prob is {}, and cb is {}.".format(prob, cb))
                act = 35
            return act

        elif self.args.exploration == 'multinomial':
            assert self.args.agent_output_type == "pi_logits"
            action = Categorical(actor_output).sample()
            return action.clone().detach().cpu().numpy()[0]

        else:
            raise NotImplementedError

    def get_action_prob(self, action, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False, noise=None):

        avail_actions = np.nonzero(1 - avail_actions_mask)[0]
        _, prob = self._get_output(obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate, noise)
        assert action in avail_actions

        return prob[0][action]

    def get_top_k_actions(self, k_actions, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False, noise=None):

        avail_actions = np.nonzero(1 - avail_actions_mask)[0]
        avail_actions = list(avail_actions)
        if k_actions > len(avail_actions):
            k_actions = len(avail_actions)
        _, prob = self._get_output(obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate, noise)
        avail_actions.sort(key=lambda x: -prob[0][x])
        # print("3: ", _, "\n", prob, "\n", avail_actions, "\n", avail_actions[:k_actions])

        return avail_actions[:k_actions]


class Opponent_agents(Agents):

    def __init__(self, args):
        super().__init__(args, False)

    def update_model(self, target_agent: Agents):
        self.actor.load_state_dict(target_agent.actor.state_dict())


class Search_agents(Agents):

    def __init__(self, args, other_agent: Agents):
        super().__init__(args, False)
        # print("4: ", self.policy.eval_rnn.state_dict())
        self.actor.load_state_dict(other_agent.actor.state_dict()) ## can't influence the other's policy
        # print("4: ", other_agent.policy.eval_rnn.state_dict(), "\n", self.policy.eval_rnn.state_dict())
    def get_eval_hidden(self):
        ## shallow copy
        return self.hidden_states.clone().detach()

    def set_eval_hidden(self, eval_hidden):
        ## shallow copy
        self.hidden_states = eval_hidden.clone().detach()

