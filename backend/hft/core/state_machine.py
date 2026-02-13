from enum import Enum, auto
from typing import Set, Dict

class TradeState(Enum):
    INIT = "INIT"
    SIGNALLED = "SIGNALLED"
    SUBMITTED = "SUBMITTED"
    FILLED_PARTIAL = "FILLED_PARTIAL"
    FILLED_FULL = "FILLED_FULL"
    EXITED = "EXITED"
    LOGGED = "LOGGED"
    REJECTED = "REJECTED"

class StateTransitionError(Exception):
    pass

class TradeStateMachine:
    """
    Enforces valid trade state transitions.
    """
    def __init__(self):
        self._allowed_transitions: Dict[TradeState, Set[TradeState]] = {
            TradeState.INIT: {TradeState.SIGNALLED, TradeState.REJECTED},
            TradeState.SIGNALLED: {TradeState.SUBMITTED, TradeState.REJECTED},
            TradeState.SUBMITTED: {TradeState.FILLED_PARTIAL, TradeState.FILLED_FULL, TradeState.REJECTED},
            TradeState.FILLED_PARTIAL: {TradeState.FILLED_PARTIAL, TradeState.FILLED_FULL, TradeState.EXITED}, # Can exit partial
            TradeState.FILLED_FULL: {TradeState.EXITED},
            TradeState.EXITED: {TradeState.LOGGED},
            TradeState.LOGGED: set(), # Terminal
            TradeState.REJECTED: {TradeState.LOGGED},
        }
        self.current_state = TradeState.INIT

    def transition(self, new_state: TradeState) -> None:
        """
        Transitions to new_state if valid.
        Raises StateTransitionError if invalid.
        """
        if new_state not in self._allowed_transitions[self.current_state]:
            raise StateTransitionError(
                f"Invalid Transition: {self.current_state} -> {new_state}"
            )
        self.current_state = new_state
        
    def check_state(self, required_state: TradeState) -> bool:
        return self.current_state == required_state
