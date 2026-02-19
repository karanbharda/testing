from enum import Enum
from typing import Set, Dict, List, Tuple


class TradeState(Enum):
    INIT          = "INIT"
    SIGNALLED     = "SIGNALLED"
    SUBMITTED     = "SUBMITTED"
    FILLED_PARTIAL = "FILLED_PARTIAL"
    FILLED_FULL   = "FILLED_FULL"
    EXITED        = "EXITED"
    LOGGED        = "LOGGED"
    REJECTED      = "REJECTED"


class StateTransitionError(Exception):
    pass


class TradeStateMachine:
    """
    Deterministic Trade State Machine.
    Enforces valid transitions and records full history for audit compliance.
    History is append-only — no mutation allowed after recording.
    """

    _ALLOWED: Dict[TradeState, Set[TradeState]] = {
        TradeState.INIT:          {TradeState.SIGNALLED, TradeState.REJECTED},
        TradeState.SIGNALLED:     {TradeState.SUBMITTED, TradeState.REJECTED},
        TradeState.SUBMITTED:     {TradeState.FILLED_PARTIAL, TradeState.FILLED_FULL, TradeState.REJECTED},
        TradeState.FILLED_PARTIAL:{TradeState.FILLED_PARTIAL, TradeState.FILLED_FULL, TradeState.EXITED},
        TradeState.FILLED_FULL:   {TradeState.EXITED},
        TradeState.EXITED:        {TradeState.LOGGED},
        TradeState.LOGGED:        set(),          # Terminal — no further transitions
        TradeState.REJECTED:      {TradeState.LOGGED},
    }

    def __init__(self):
        self.current_state: TradeState = TradeState.INIT
        self._history: List[TradeState] = [TradeState.INIT]

    def transition(self, new_state: TradeState) -> None:
        """
        Transitions to new_state if valid.
        Raises StateTransitionError if the transition is illegal.
        Records every valid transition in the immutable history.
        """
        if new_state not in self._ALLOWED[self.current_state]:
            raise StateTransitionError(
                f"Illegal transition: {self.current_state.value} → {new_state.value}. "
                f"Allowed: {[s.value for s in self._ALLOWED[self.current_state]]}"
            )
        self.current_state = new_state
        self._history.append(new_state)

    def check_state(self, required_state: TradeState) -> bool:
        """Returns True if current state matches required_state."""
        return self.current_state == required_state

    def get_state_history(self) -> Tuple[str, ...]:
        """
        Returns an immutable tuple of all states visited, in order.
        Suitable for embedding directly into TradeLifecycleArtifact.state_history.
        """
        return tuple(s.value for s in self._history)

    def validate_lifecycle_completeness(self) -> bool:
        """
        Returns True if the state machine reached a valid terminal state.
        Valid terminals: LOGGED (after EXITED or REJECTED path).
        """
        return self.current_state == TradeState.LOGGED
