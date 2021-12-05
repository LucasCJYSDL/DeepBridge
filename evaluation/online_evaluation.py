import numpy as np

from evaluation.external_bot import BlueChipBridgeBot
from evaluation.duplicate_env import DuplicateEnv
from evaluation.external_client import WBridge5Client

from common.arguments import get_common_args, get_q_decom_args
from agents.agents import Agents
from network.bandits.uniform import Uniform



def _run_once(game, bots, our_agent, args, noise_generator=None):
    """Plays bots with each other, and returns terminal utility."""
    our_agent.init_hidden(1)
    last_action = np.zeros((args.n_agents, args.n_actions))
    game.reset()
    noise = None
    if args.alg == 'maven':
        noise = noise_generator.sample(state=None, test_mode=True)
    for bot in bots.values():
        bot.restart()
    while not game.is_terminal():
        current_player = game.get_turn()
        if current_player in bots.keys():
            action = bots[current_player].step()
            game.step(action)
        else:
            # game.step(35) ## TODO
            temp_obs = game.get_obs()
            agent_id = game.get_agent_id()
            avai_action = game.get_avai_action()
            action = our_agent.choose_action(temp_obs, last_action[agent_id], agent_id, avai_action,
                                             epsilon=None, replay_buffer=None, evaluate=True, noise=noise)
            onehot_action = np.zeros(args.n_actions)
            onehot_action[action] = 1
            last_action[agent_id] = onehot_action
            game.step(action)

    score = game.get_board_score()
    return score


def evaluate(game, arguments, our_agent):
    noise_generator = None
    if arguments.alg == 'maven':
        noise_generator = Uniform(arguments)

    bots_0 = {
        1: BlueChipBridgeBot(game, 1, WBridge5Client('wine ./Wbridge5.exe Autoconnect {port}')), ## EAST
        3: BlueChipBridgeBot(game, 3, WBridge5Client('wine ./Wbridge5.exe Autoconnect {port}'))  ## WEST
    }
    score_list_0 = [] ## score list of table 0
    for i_deal in range(arguments.board_num):
        contract_score = _run_once(game, bots_0, our_agent, arguments, noise_generator=noise_generator)
        print("Deal #{}; Final score: {}".format(i_deal, contract_score))
        score_list_0.append(contract_score)

    bots_1 = {
        0: BlueChipBridgeBot(game, 0, WBridge5Client('wine ./Wbridge5.exe Autoconnect {port}')),  ## NORTH
        2: BlueChipBridgeBot(game, 2, WBridge5Client('wine ./Wbridge5.exe Autoconnect {port}'))   ## SOUTH
    }
    score_list_1 = [] ## score list of table 1
    for i_deal in range(arguments.board_num, 2*arguments.board_num):
        contract_score = _run_once(game, bots_1, our_agent, arguments, noise_generator=noise_generator)
        print("Deal #{}; Final score: {}".format(i_deal, contract_score))
        score_list_1.append(contract_score)

    average_IMP, std_IMP = game.convert_to_IMP(score_list_0, score_list_1)
    print("Number of the duplicate boards: {}; IMP difference: Mean: {}, Standard Error: {}".format(16, \
                                                                                                    average_IMP, std_IMP))

    return average_IMP


if __name__ == "__main__":
    game = DuplicateEnv(16)
    arguments = get_q_decom_args(get_common_args())
    env_info = game.get_env_info()
    arguments.n_actions = env_info['n_actions']
    arguments.n_agents = env_info['n_agents']
    arguments.n_cards = env_info['n_cards']
    arguments.obs_shape = env_info['obs_shape']
    arguments.pre_obs_shape = env_info['pre_obs_shape']  ## observation for predicting the partner's cards
    arguments.load_model = True
    arguments.load_model_path = '../ckpt/qmix/2020_11_11_19_11_32/qmix_24'
    our_agent = Agents(arguments, is_target=False)
    evaluate(game, arguments, our_agent)
    print("End of Online Evaluation!")
