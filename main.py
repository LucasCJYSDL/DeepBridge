# -*- coding: utf-8 -*-

from environment.bridge_env import BridgeEnv
from common.arguments import get_common_args, get_q_decom_args, get_search_args
from runner import Runner
import os


def main(env, arg):
    runner = Runner(env, arg)
    runner.run()


if __name__ == '__main__':

    arguments = get_q_decom_args(get_common_args())
    arguments = get_search_args(arguments, evaluation=False)

    if arguments.gpu is not None:
        arguments.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        arguments.cuda = False

    environment = BridgeEnv(debug=arguments.debug, score_mode=arguments.score_mode)
    env_info = environment.get_env_info()
    arguments.n_actions = env_info['n_actions']
    arguments.n_agents = env_info['n_agents']
    arguments.n_cards = env_info['n_cards']
    arguments.state_shape = env_info['state_shape']
    arguments.state_hands_shape = env_info['state_hands_shape']
    arguments.obs_shape = env_info['obs_shape']
    arguments.pre_obs_shape = env_info['pre_obs_shape'] ## observation for predicting the partner's cards
    arguments.episode_limit = env_info['episode_limit']

    main(environment, arguments)

