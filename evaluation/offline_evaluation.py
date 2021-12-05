"""
The bot_cmd FLAG should contain a command-line to launch an external bot, e.g.
`Wbridge5 Autoconnect {port}`.
1. Install wine;
2. Run Wbridge5_setup.exe (under wine), which will create WBridge5.exe somewhere;
3. python3 bridge_uncontested_bidding_bluechip.py --num_deals=2 --bot_cmd 'wine /your_path/Wbridge5.exe Autoconnect {port}'
"""
# pylint: enable=line-too-long


from absl import app
from absl import flags
import numpy as np

from evaluation.external_bot import BlueChipBridgeBot
from evaluation.duplicate_env import DuplicateEnv
from evaluation.external_client import WBridge5Client

from common.arguments import get_common_args, get_q_decom_args
from agents.agents import Agents
from network.bandits.uniform import Uniform

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_deals", 16, "How many deals to play")
flags.DEFINE_string("bot_cmd", 'wine ./Wbridge5.exe Autoconnect {port}',\
                    "Command to launch the external bot; must include {port} which will be replaced by the port number to attach to.")
flags.DEFINE_string("model_file", '../ckpt/qmix/2020_11_11_19_11_32/qmix_24', "NN model for our agent")


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


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    game = DuplicateEnv(FLAGS.num_deals)
    arguments = get_q_decom_args(get_common_args())
    env_info = game.get_env_info()
    arguments.n_actions = env_info['n_actions']
    arguments.n_agents = env_info['n_agents']
    arguments.n_cards = env_info['n_cards']
    arguments.obs_shape = env_info['obs_shape']
    arguments.pre_obs_shape = env_info['pre_obs_shape'] ## observation for predicting the partner's cards
    arguments.load_model = True
    arguments.load_model_path = FLAGS.model_file
    our_agent = Agents(arguments, is_target=False)

    noise_generator = None
    if arguments.alg == 'maven':
        noise_generator = Uniform(arguments)

    bots_0 = {
        1: BlueChipBridgeBot(game, 1, WBridge5Client(FLAGS.bot_cmd)), ## EAST
        3: BlueChipBridgeBot(game, 3, WBridge5Client(FLAGS.bot_cmd))  ## WEST
    }
    score_list_0 = [] ## score list of table 0
    for i_deal in range(FLAGS.num_deals):
        contract_score = _run_once(game, bots_0, our_agent, arguments, noise_generator=noise_generator)
        print("Deal #{}; Final score: {}".format(i_deal, contract_score))
        score_list_0.append(contract_score)

    bots_1 = {
        0: BlueChipBridgeBot(game, 0, WBridge5Client(FLAGS.bot_cmd)),  ## NORTH
        2: BlueChipBridgeBot(game, 2, WBridge5Client(FLAGS.bot_cmd))   ## SOUTH
    }
    score_list_1 = [] ## score list of table 1
    for i_deal in range(FLAGS.num_deals, 2*FLAGS.num_deals):
        contract_score = _run_once(game, bots_1, our_agent, arguments, noise_generator=noise_generator)
        print("Deal #{}; Final score: {}".format(i_deal, contract_score))
        score_list_1.append(contract_score)

    average_IMP, std_IMP = game.convert_to_IMP(score_list_0, score_list_1)
    print("Number of the duplicate boards: {}; IMP difference: Mean: {}, Standard Error: {}".format(FLAGS.num_deals, \
                                                                                                    average_IMP, std_IMP))


if __name__ == "__main__":
    app.run(main)
    print("End of Offline Evaluation!")
