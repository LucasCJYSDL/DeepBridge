# -*- coding: utf-8 -*-

import numpy as np
from common.replay_buffer import ReplayBuffer
from agents.agents import Opponent_agents, Search_agents
from agents.target_agents import Target_agents
from search.borel_search import get_search_result
from evaluation.duplicate_env import DuplicateEnv
from evaluation.online_evaluation import evaluate
from learners import REGISTRY as learner_REGISTRY
from network.bandits.uniform import Uniform
from network.bandits.hierarchial import EZ_agent

class Runner:
    def __init__(self, env, args):
        self.args = args
        self.env = env

        self.target_agents = Target_agents(args)
        self.opponent_agents = Opponent_agents(args)
        self.replay_buffer = ReplayBuffer(self.args)
        if args.load_replay_buffer:
            self.replay_buffer.restore(args.replay_buffer_path)
            print("Loading buffers from {}".format(args.replay_buffer_path))
        self.learner = learner_REGISTRY[self.args.learner](self.target_agents, self.args)

        self.noise_generator = None
        if args.alg == 'maven':
            assert self.target_agents.get_actor_name() == 'noise_rnn'
            if args.noise_bandit:
                self.noise_generator = EZ_agent(args)
                self.oppo_noise_generator = EZ_agent(args)
            else:
                self.noise_generator = Uniform(args)
                self.oppo_noise_generator = Uniform(args)

        self.start_itr = 0
        if args.load_model:
            assert len(args.load_model_path) > 0
            print("Loading from model: {}!!!".format(args.load_model_path))
            self.learner.load_models(args.load_model_path)
            self.opponent_agents.load_models(args.load_model_path)
            if args.alg == 'maven':
                self.noise_generator.load_models(args.load_model_path)
                self.oppo_noise_generator.load_models(args.load_model_path)
            ## load_model_path: 'ckpt + '/' + args.alg + '/' + timestamp + '/' +idx
            alg = args.load_model_path.split('/')[-3]
            assert alg == args.alg, "Wrong model to load!"
            idx = int(args.load_model_path.split('/')[-1])
            self.start_itr = idx * args.save_model_period // args.train_steps
            self.args.epsilon = self.args.epsilon - self.start_itr * self.args.epsilon_decay
            if self.args.epsilon < self.args.min_epsilon:
                self.args.epsilon = self.args.min_epsilon
        self.start_train_steps = self.start_itr * args.train_steps

    def generate_episode(self, episode_num, evaluate=False):

        epsilon = 0 if evaluate else self.args.epsilon
        if self.args.epsilon_anneal_scale == 'episode' or (self.args.epsilon_anneal_scale == 'itr' and episode_num == 0):
            epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon
        if not evaluate:
            self.args.epsilon = epsilon

        episode_buffer = None
        if not evaluate:
            episode_buffer = {'o':            np.zeros([self.args.episode_limit + 1, self.args.n_agents, self.args.obs_shape]),
                              's':            np.zeros([self.args.episode_limit + 1, self.args.state_shape]),
                              'a':            np.zeros([self.args.episode_limit, self.args.n_agents, 1]),
                              'onehot_a':     np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'avail_a':      np.zeros([self.args.episode_limit + 1, self.args.n_agents, self.args.n_actions]),
                              'r':            np.zeros([self.args.episode_limit, 1]),
                              'done':         np.ones([self.args.episode_limit, 1]),
                              'padded':       np.ones([self.args.episode_limit, 1]),
                              'gamma':        np.zeros([self.args.episode_limit, 1]),
                              'next_idx':     np.zeros([self.args.episode_limit, 1])}
        # roll out
        self.target_agents.init_hidden(1)
        self.opponent_agents.init_hidden(1)
        target_last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        opponent_last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        temp_list = []
        self.env.reset()
        target_noise = None
        oppo_noise = None
        if self.args.alg == 'maven':
            target_state_hands, oppo_state_hands = self.env.get_state_hands()
            target_noise = self.noise_generator.sample(target_state_hands, test_mode=False)
            oppo_noise = self.oppo_noise_generator.sample(oppo_state_hands, test_mode=False)
        for episode in range(self.args.episode_limit*2):
            temp_obs = self.env.get_obs()
            is_target = self.env.is_target_group()
            agent_id = self.env.get_agent_id()
            avai_action = self.env.get_avai_action()

            if is_target:
                if (self.env.get_avai_action_num() > 1) and (np.random.rand() <= self.args.search_prob):
                    print("SEARCH......")
                    top_k_actions = self.target_agents.get_top_k_actions(self.args.k_actions, temp_obs, target_last_action[agent_id],\
                                                                         agent_id, avai_action, epsilon, evaluate, noise=target_noise)
                    # print("obs_before: ", temp_obs)
                    # print("la_before: ", target_last_action, "\n", opponent_last_action)
                    # print("hidden_before: ", self.target_agents.policy.eval_hidden[:, agent_id, :50])
                    action = get_search_result(top_k_actions, self.args, self.env.get_turn(),self.env.get_current_team_deal(),\
                                               self.env.get_vulner_sign(), self.env.get_action_history(), epsilon, evaluate,\
                                               Search_agents(self.args, self.target_agents), Search_agents(self.args, self.opponent_agents),
                                               target_noise=target_noise, oppo_noise=oppo_noise)
                else:
                    action = self.target_agents.choose_action(temp_obs, target_last_action[agent_id], agent_id,
                                                   avai_action, epsilon, self.replay_buffer, evaluate, noise=target_noise)
                onehot_action = np.zeros(self.args.n_actions)
                onehot_action[action] = 1
                target_last_action[agent_id] = onehot_action
                temp_state = self.env.get_state()
                temp_dict = {'agent_id': agent_id, 'o': temp_obs, 's':temp_state, 'a': action, 'onehot_a': onehot_action, 'avail_a': avai_action}
                temp_list.append(temp_dict)
            else:
                action = self.opponent_agents.choose_action(temp_obs, opponent_last_action[agent_id], agent_id,
                                                   avai_action, epsilon, self.replay_buffer, evaluate, noise=oppo_noise)
                onehot_action = np.zeros(self.args.n_actions)
                onehot_action[action] = 1
                opponent_last_action[agent_id] = onehot_action
            # print(action)
            done, _ = self.env.step(action)
            if done:
                break

        round_num = len(temp_list) - 1
        for idx in range(round_num):
            target_group = {}
            assert temp_list[idx]['agent_id'] != temp_list[idx+1]['agent_id']
            target_group[temp_list[idx]['agent_id']] = temp_list[idx]
            target_group[temp_list[idx+1]['agent_id']] = temp_list[idx+1]
            obses, avail_actions, actions, onehot_actions = [], [], [], []
            for agent_id in range(self.args.n_agents):
                obses.append(target_group[agent_id]['o'])
                avail_actions.append(target_group[agent_id]['avail_a'])
                actions.append(target_group[agent_id]['a'])
                onehot_actions.append(target_group[agent_id]['onehot_a'])
            target_state = temp_list[idx+1]['s']
            # if not evaluate:
            episode_buffer['o'][idx] = obses
            episode_buffer['s'][idx] = target_state
            episode_buffer['a'][idx] = np.reshape(actions, [self.args.n_agents, 1])
            episode_buffer['onehot_a'][idx] = onehot_actions
            episode_buffer['avail_a'][idx] = avail_actions
            episode_buffer['r'][idx] = [0.]  ## need further update
            episode_buffer['done'][idx] = [False]  ## need further update
            episode_buffer['padded'][idx] = [0.]
            # episode_buffer['gamma'][idx] = [self.args.gamma]

        print("The total round number is {}.".format(round_num))
        episode_buffer['done'][round_num-1] = [True]
        episode_buffer['o'][round_num] = episode_buffer['o'][round_num-1].copy()
        episode_buffer['s'][round_num] = episode_buffer['s'][round_num-1].copy()
        episode_buffer['avail_a'][round_num] = episode_buffer['avail_a'][round_num-1].copy()

        if done:
            episode_buffer['r'][round_num-1] = self.env.get_board_reward()
        else:
            assert episode == self.args.episode_limit*2-1
            print("The biding is not completed within the episode limit: {}!".format(self.args.episode_limit*2))
            # episode_buffer['done'][round_num - 1] = [True] ## danger
            episode_buffer['r'][round_num - 1] = self.env.get_least_reward() ## danger

        print("The board score for the target group is {}!".format(episode_buffer['r'][round_num - 1][0]))

        if self.args.alg == 'maven':
            self.noise_generator.update_returns(target_state_hands, target_noise, episode_buffer['r'][round_num - 1][0])
            episode_buffer['noise'] = np.array(target_noise)

        episode_buffer = self.multi_step_TD(episode_buffer, round_num)

        return episode_buffer, episode_buffer['r'][round_num - 1]


    def multi_step_TD(self, episode_buffer, round_num):

        n = self.args.step_num
        gamma = self.args.gamma
        for e in range(round_num):
            if (e + n) < round_num:
                episode_buffer['gamma'][e] = [gamma**n]
                temp_rwd = 0.
                for idx in range(e, e+n):
                    factor = gamma**(idx-e)
                    temp_rwd += factor * episode_buffer['r'][idx][0]
                episode_buffer['r'][e] = [temp_rwd]
                episode_buffer['next_idx'][e] = [n]
            else:
                episode_buffer['done'][e] = [True]
                episode_buffer['gamma'][e] = [gamma**(round_num-e)]
                temp_rwd = 0.
                for idx in range(e, round_num):
                    factor = gamma**(idx-e)
                    temp_rwd += factor * episode_buffer['r'][idx][0]
                episode_buffer['r'][e] = [temp_rwd]
                episode_buffer['next_idx'][e] = [round_num - 1 - e] ## check
            if episode_buffer['next_idx'][e][0] + e - 1 < 0:
                print("Bad index!!!")
                episode_buffer['next_idx'][e][0] = 1 - e
        return episode_buffer


    def run(self):
        train_steps = self.start_train_steps
        for itr in range(self.start_itr, self.args.n_itr):
            print("##########################{}##########################".format(itr))
            if (train_steps != 0) and ((train_steps % self.args.opponent_update_period) == 0):
                print("Loading model for the opponent group ......")
                self.opponent_agents.update_model(self.target_agents)
                if self.args == 'maven':
                    self.oppo_noise_generator.update_model(self.noise_generator)
            scores = []
            episode_batch, score = self.generate_episode(0)
            scores.append(score[0])
            for key in episode_batch.keys():
                episode_batch[key] = np.array([episode_batch[key]])
            if self.args.alg == 'coma':
                assert (self.args.n_episodes > 1) and (self.args.n_episodes == self.args.batch_size == self.args.buffer_size) \
                and self.args.train_steps == 1, "COMA should be online learning!!!"
            for e in range(1, self.args.n_episodes):
                episode, score = self.generate_episode(e)
                scores.append(score[0])
                for key in episode.keys():
                    episode[key] = np.array([episode[key]])
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            self.replay_buffer.store(episode_batch)
            if not self.replay_buffer.can_sample(self.args.batch_size):
                print("No enough episodes!!!")
                continue

            delta_IMP = None
            if (train_steps % self.args.evaluation_period == 0):
                print("Online Evaluation Starts!")
                duplicate_game = DuplicateEnv(self.args.board_num)
                delta_IMP = evaluate(duplicate_game, self.args, self.target_agents)

            log_dict = self.learner.get_log_dict()
            for _ in range(self.args.train_steps):
                batch = self.replay_buffer.sample(self.args.batch_size)
                max_episode_len = self.target_agents.get_max_episode_len(batch)
                for key in batch.keys():
                    if key == 'noise':
                        continue
                    if key in ['o', 's', 'avail_a']:
                        batch[key] = batch[key][:, :max_episode_len+1]
                    else:
                        batch[key] = batch[key][:, :max_episode_len]
                log_info = self.learner.train(batch, max_episode_len, train_steps)
                for key in log_dict.keys():
                    assert key in log_info, key
                    log_dict[key].append(log_info[key])
                if train_steps > 0 and train_steps % self.args.save_model_period == 0:
                    print("Saving the models!")
                    self.learner.save_models(train_steps)
                    if self.args.alg == 'maven':
                        save_dir = self.learner.get_save_dir()
                        self.noise_generator.save_models(save_dir, train_steps)
                train_steps += 1

            print("Log to the tensorboard!")
            self.learner.log_info(delta_IMP, scores, log_dict, itr)

            # if (itr > 0) and (itr%self.args.save_buffer_period == 0):
            #     print("Saving the replay buffer!")
            #     self.replay_buffer.save(self.args.replay_buffer_path)
