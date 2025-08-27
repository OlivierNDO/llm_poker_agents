"""
poker_types.py

Poker type classes broken out to avoid circular imports
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class ActionType(Enum):
    FOLD = 'fold'
    CHECK = 'check'
    CALL = 'call'
    BET = 'bet'
    RAISE = 'raise'
    ALL_IN = 'all_in'

@dataclass
class Action:
    action_type: ActionType
    amount: int = 0
    reasons: Optional[List[str]] = None

@dataclass
class ActionRecord:
    player_name: str
    action: Action
    betting_round: int
    board_at_time: list  # list of Card
    pot_before: int
    current_bet_before: int
    position: int
