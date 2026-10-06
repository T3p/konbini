from gymnasium.envs.registration import register
from .full_bandit import FullBandit
from .instances import (
    bernoulli,
    correlated,
    corrupted_stationary_good_arms,
    geometric_blocks,
    stationary_good_arms,
)
from .semi_bandit import SemiBandit
from .multiplayer_mab import MultiplayerMab
from .win_win import WinWin

__all__ = [
    "FullBandit",
    "bernoulli",
    "correlated",
    "corrupted_stationary_good_arms",
    "geometric_blocks",
    "MultiplayerMab",
    "SemiBandit",
    "WinWin",
    "stationary_good_arms",
]

register(
    id='FullBandit-v0',
    entry_point='konbini.envs.full_bandit:FullBandit'
)

register(
    id='SemiBandit-v0',
    entry_point='konbini.envs.semi_bandit:SemiBandit'
)

register(
    id='MultiplayerMab-v0',
    entry_point='konbini.envs.multiplayer_mab:MultiplayerMab'
)

register(
    id='WinWin-v0',
    entry_point='konbini.envs.win_win:WinWin'
)
