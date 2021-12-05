from search.search_env import SearchEnv
from agents.agents import Search_agents
import numpy as np

def _check_node(env, args, particle, vulner_dict, action_history, epsilon, evaluate, target_agent, oppo_agent, target_noise=None, oppo_noise=None):
    flag = False
    target_last_action = np.zeros((args.n_agents, args.n_actions))
    opponent_last_action = np.zeros((args.n_agents, args.n_actions))
    target_agent.init_hidden(1)
    oppo_agent.init_hidden(1)
    if len(action_history) == 0:
        return flag, target_last_action, opponent_last_action

    env.reset_from(particle, vulner_dict, [action_history[0]], init=True)
    # for player, action in action_history.items():
    for action_pair in action_history:
        player = list(action_pair.keys())[0]
        action = list(action_pair.values())[0]
        assert player == env.get_turn(), player
        is_target = env.is_target_group()
        temp_obs = env.get_obs()
        agent_id = env.get_agent_id()
        avai_action = env.get_avai_action()
        onehot_action = np.zeros(args.n_actions)
        onehot_action[action] = 1

        if is_target:
            prob = target_agent.get_action_prob(action, temp_obs, target_last_action[agent_id], agent_id,\
                                                avai_action, epsilon, evaluate, noise=target_noise)
            target_last_action[agent_id] = onehot_action
        else:
            prob = oppo_agent.get_action_prob(action, temp_obs, opponent_last_action[agent_id], agent_id,\
                                              avai_action, epsilon, evaluate, noise=oppo_noise)
            opponent_last_action[agent_id] = onehot_action

        # if np.random.rand() > prob:
        #     flag = True
        #     break

        env.step(action)

    return flag, target_last_action, opponent_last_action

def _rollout_value(env, args, current_player, target_last_action, opponent_last_action, target_eval_hidden, \
                   oppo_eval_hidden, action_history, action, particle, vulner_dict, epsilon, evaluate, target_agent, oppo_agent,\
                   target_noise=None, oppo_noise=None):

    temp_target_last_action = target_last_action.copy()
    temp_oppo_last_action = opponent_last_action.copy()
    # print("7: ", target_agent.policy.eval_hidden[:, :, :50], "\n", oppo_agent.policy.eval_hidden[:, :, :50])
    target_agent.set_eval_hidden(target_eval_hidden)
    oppo_agent.set_eval_hidden(oppo_eval_hidden)
    # print("8: ", target_agent.policy.eval_hidden[:, :, :50], "\n", oppo_agent.policy.eval_hidden[:, :, :50])
    if len(action_history) == 0:
        env.reset_from(particle, vulner_dict, [{current_player: action}], init=True)
    else:
        env.reset_from(particle, vulner_dict, action_history, init=False)

    round_num = 0
    while not env.is_terminal():
        round_num += 1
        is_target = env.is_target_group()
        temp_obs = env.get_obs()
        agent_id = env.get_agent_id()
        avai_action = env.get_avai_action()
        onehot_action = np.zeros(args.n_actions)
        if round_num == 1:
            assert current_player == env.get_turn() and is_target
            top_k_actions = target_agent.get_top_k_actions(args.k_actions, temp_obs, temp_target_last_action[agent_id],\
                                                           agent_id, avai_action, epsilon, evaluate, noise=target_noise)
            # print("obs_after: ", temp_obs)
            # print("la_after: ", temp_target_last_action, "\n", temp_oppo_last_action)
            # print("hidden_after: ", target_agent.policy.eval_hidden[:, agent_id, :50])
            ## since the last action. hidden tensor and temp_obs should be the same
            assert action in top_k_actions
            temp_action = action
            onehot_action[temp_action] = 1
            temp_target_last_action[agent_id] = onehot_action

        else:
            if is_target:
                temp_action = target_agent.choose_action(temp_obs, temp_target_last_action[agent_id], agent_id, \
                                                         avai_action, epsilon, replay_buffer=None, evaluate=evaluate, noise=target_noise)
                onehot_action[temp_action] = 1
                temp_target_last_action[agent_id] = onehot_action
            else:
                temp_action = oppo_agent.choose_action(temp_obs, temp_oppo_last_action[agent_id], agent_id, avai_action,\
                                                       epsilon, replay_buffer=None, evaluate=evaluate, noise=oppo_noise)
                onehot_action[temp_action] = 1
                temp_oppo_last_action[agent_id] = onehot_action

        env.step(temp_action)

    return env.get_board_reward() ## always for the target group

def get_search_result(top_k_actions, args, current_player, known_deal, vulner_dict, \
                      action_history, epsilon, evaluate, target_agent: Search_agents, oppo_agent: Search_agents, \
                      target_noise=None, oppo_noise=None):
    assert len(top_k_actions) > 1
    V = {}
    for action in top_k_actions:
        V[action] = 0
    R = 0
    P = 0
    env = SearchEnv()
    if len(action_history) > 0:
        assert current_player == (list(action_history[-1].keys())[0] + 1) % (env.num_player)
    print("Searching......")
    while (P < args.p_max) and (R < args.r_max):
        particle = env.sample_particle(current_player, known_deal)
        P = P + 1
        flag, target_last_action, opponent_last_action = _check_node(env, args, particle, vulner_dict, action_history, \
                                                                    epsilon, evaluate, target_agent, oppo_agent, target_noise, oppo_noise)
        if flag:
            continue

        ## TODO: check the init_hidden of the target agent
        target_eval_hidden = target_agent.get_eval_hidden()
        oppo_eval_hidden = oppo_agent.get_eval_hidden()
        # print("6: ", target_eval_hidden[:, :, :50], "\n", oppo_eval_hidden[:, :, :50])
        for action in top_k_actions:
            # print("9: ", target_last_action, "\n", opponent_last_action)
            temp_value = _rollout_value(env, args, current_player, target_last_action, opponent_last_action, \
                                        target_eval_hidden, oppo_eval_hidden, action_history, action, particle, \
                                        vulner_dict, epsilon, evaluate, target_agent, oppo_agent, target_noise, oppo_noise)
            V[action] += temp_value
        R = R + 1

    if R > args.r_min:
        print("The Search is Successful!")
        top_k_actions.sort(key=lambda x: -V[x])
    else:
        print("The Search is Failed!")
    print("P is {}, and R is {}".format(P, R))
    print("Search Results: ", V, " ", top_k_actions)
    return top_k_actions[0] ## the highest q value or highest estimated value