"""Wraps third-party bridge bots

Take advantage of implement the BlueChip bridge protocol.
This is widely used, e.g. in the World computer bridge championships.
For a rough outline of the protocol, see: http://www.bluechipbridge.co.uk/protocol.htm

This implementation has been verified to work correctly with WBridge5.

This bot controls a single player in the game of bridge bidding.
It chooses its actions by invoking an external bot which plays the full game of bridge.
This means that each time the bot is asked for an action, it sends up to
three actions (forced passes from both opponents, plus partner's most recent
action) to the external bridge bot, and obtains an action in return.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re

# Example session:
#
# Recv: Connecting "WBridge5" as ANYPL using protocol version 18
# Send: WEST ("WBridge5") seated
# Recv: WEST ready for teams
# Send: Teams: N/S "silent" E/W "bidders"
# Recv: WEST ready to start
# Send: Start of board
# Recv: WEST ready for deal
# Send: Board number 8. Dealer WEST. Neither vulnerable.
# Recv: WEST ready for cards
# Send: WEST's cards: S A T 9 5. H K 6 5. D Q J 8 7 6. C 7.
# Recv: WEST PASSES
# Recv: WEST ready for NORTH's bid
# Send: NORTH PASSES
# Recv: WEST ready for EAST's bid
# Send: EAST bids 1C
# Recv: WEST ready for SOUTH's bid

# Template regular expressions for messages we receive
_CONNECT = 'Connecting "(?P<client_name>.*)" as ANYPL using protocol version 18'
_SELF_BID_OR_PASS_OR_DBL = "{seat} ((?P<pass>PASSES)|(?P<dbl>DOUBLES)|(?P<rdbl>REDOUBLES)|bids (?P<bid>[^ ]*))( Alert.)?"

# Templates for fixed messages we receive
_READY_FOR_TEAMS = "{seat} ready for teams"
_READY_TO_START = "{seat} ready to start"
_READY_FOR_DEAL = "{seat} ready for deal"
_READY_FOR_CARDS = "{seat} ready for cards"
_READY_FOR_BID = "{seat} ready for {other}'s bid"

# Templates for messages we send
_SEATED = '{seat} ("{client_name}") seated'
_TEAMS = 'Teams: N/S "north-south" E/W "east-west"'  ## 'Teams: N/S "opponents" E/W "bidders"'
_START_BOARD = "start of board"
# The board number is arbitrary, but "8" is consistent with the dealer and
# vulnerability we want (in the standard numbering). See Law 2:
# http://web2.acbl.org/documentlibrary/play/Laws-of-Duplicate-Bridge.pdf
_DEAL = "Board number {num}. Dealer {seat}. {vulner}."  ## "Board number 8. Dealer WEST. Neither vulnerable."
_CARDS = "{seat}'s cards: {hand}"
_OTHER_PLAYER_PASS = "{player} PASSES"
_OTHER_PLAYER_DBL = "{player} DOUBLES"
_OTHER_PLAYER_RDBL = "{player} REDOUBLES"
_OTHER_PLAYER_BID = "{player} bids {bid}"

# BlueChip bridge protocol message constants
_SEATS = ["NORTH", "EAST", "SOUTH", "WEST"]
_VULNERS = ["Neither vulnerable", "N/S vulnerable", "E/W vulnerable", "Both vulnerable"]
_TRUMP_SUIT = ["C", "D", "H", "S", "NT"]
_NUMBER_TRUMP_SUITS = len(_TRUMP_SUIT)
_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

_PASS_ACTION = 35
_DBL_ACTION = 36
_RDBL_ACTION = 37


def _string_to_action(call_str):
    """
    Args:
      call_str: string representing a bid in the BlueChip format, i.e. "[level]
        (as a digit) + [trump suit (S, H, D, C or NT)]", e.g. "1C".
    Returns:
      An integer action id: 0~34
    """
    level = int(call_str[0])
    trumps = _TRUMP_SUIT.index(call_str[1:])
    return (level - 1) * _NUMBER_TRUMP_SUITS + trumps


def _action_to_string(action):
    """   Inverse of `_string_to_action.
    Args:
      action: an integer action id corresponding to a bid.
    Returns:
      A string in BlueChip format.
    """
    assert action >= 0 and action <= 34
    level = str(action // _NUMBER_TRUMP_SUITS + 1)
    trumps = _TRUMP_SUIT[action % _NUMBER_TRUMP_SUITS]
    return level + trumps


def _expect_regex(client, regex):
    """Reads a line from the client, parses it using the regular expression."""
    line = client.read_line()
    # print("receive: ", line)
    match = re.match(regex, line)
    if not match:
        raise ValueError("Received '{}' which does not match regex '{}'".format(
            line, regex))
    return match.groupdict()


def _expect(client, expected):
    """Reads a line from the client, checks it matches expected line exactly."""
    line = client.read_line()
    # print("receive: ", line)
    if expected != line:
        raise ValueError("Received '{}' but expected '{}'".format(line, expected))


def _hand_string(obs_vec):  ## check
    """Returns the hand of the to-play player in the state in BlueChip format."""
    # The first 52 bits of the obs_vec is the own cards, ordered suit-by-suit, in ascending order of rank.

    suits = []
    for suit in reversed(range(4)):
        cards = []
        for rank in reversed(range(13)):
            idx = rank + (3 - suit) * 13
            assert idx < 52
            if obs_vec[idx]:
                cards.append(_RANKS[rank])
        suits.append(_TRUMP_SUIT[suit] + " " + (" ".join(cards) if cards else "-") + ".")
    return " ".join(suits)


def _connect(client, seat, board_id, dealer, vulner, obs_vec):
    """Performs the initial handshake with a BlueChip bot."""
    client.start()
    client_name = _expect_regex(client, _CONNECT)["client_name"]
    client.send_line(_SEATED.format(seat=seat, client_name=client_name))
    _expect(client, _READY_FOR_TEAMS.format(seat=seat))
    client.send_line(_TEAMS)
    _expect(client, _READY_TO_START.format(seat=seat))
    client.send_line(_START_BOARD)
    _expect(client, _READY_FOR_DEAL.format(seat=seat))
    client.send_line(_DEAL.format(num=board_id, seat=dealer, vulner=vulner))
    _expect(client, _READY_FOR_CARDS.format(seat=seat))
    client.send_line(_CARDS.format(seat=seat, hand=_hand_string(obs_vec)))


class BlueChipBridgeBot(object):

    def __init__(self, game, player_id, client):
        '''
          Args:
          game: The duplicate game object;
          player_id: The id of the player the bot will act as, 0: N, 1: E, 2: S, 3: W;
          client: The BlueChip bot; must support methods `start`, `read_line`, and `send_line`.
        '''
        self._game = game
        self._player_id = player_id
        self._client = client
        self._seat = _SEATS[player_id]
        self._partner = _SEATS[(player_id + 2) % 4]
        self._left_hand_opponent = _SEATS[(player_id + 1) % 4]
        self._right_hand_opponent = _SEATS[(player_id + 3) % 4]
        self._other_players = {-1: self._right_hand_opponent, -2: self._partner, -3: self._left_hand_opponent}
        self._connected = False

    def player_id(self):
        return self._player_id

    def restart(self):
        self._connected = False

    def step(self):
        ## Get the observation for the bot first.
        assert self._player_id == self._game.get_turn()  ## It's this player's turn
        obs_vec = self._game.get_obs()

        # Connect if necessary.
        if not self._connected:
            board_id = self._game.get_board_id() + 1
            dealer = _SEATS[self._game.get_dealer()]
            vulner = _VULNERS[self._game.get_vulner()]
            _connect(self._client, self._seat, board_id, dealer, vulner, obs_vec)
            self._connected = True

        # Get the actions in the game so far.
        actions = self._game.get_action_history()
        len_actions = len(actions)

        # communicate with other player bots
        for idx in reversed(range(1, 4)):
            if len_actions >= idx:
                temp_player = self._other_players[-idx]
                assert (self._player_id+4-idx)%4 in actions[-idx].keys()
                temp_action = actions[-idx][(self._player_id+4-idx)%4]
                _expect(self._client, _READY_FOR_BID.format(seat=self._seat, other=temp_player))
                if temp_action == _PASS_ACTION:
                    self._client.send_line(_OTHER_PLAYER_PASS.format(player=temp_player))
                elif temp_action == _DBL_ACTION:
                    self._client.send_line(_OTHER_PLAYER_DBL.format(player=temp_player))
                elif temp_action == _RDBL_ACTION:
                    self._client.send_line(_OTHER_PLAYER_RDBL.format(player=temp_player))
                else:
                    self._client.send_line(
                        _OTHER_PLAYER_BID.format(player=temp_player, bid=_action_to_string(temp_action)))

        own_action = _expect_regex(self._client, _SELF_BID_OR_PASS_OR_DBL.format(seat=self._seat))
        if own_action["pass"]:
            action = _PASS_ACTION
        elif own_action['dbl']:
            action = _DBL_ACTION
        elif own_action['rdbl']:
            action = _RDBL_ACTION
        else:
            action = _string_to_action(own_action['bid'])

        return action
