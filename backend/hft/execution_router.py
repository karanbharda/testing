from dataclasses import dataclass
from enum import Enum
from typing import Optional
from .shadow_execution.simulator import ShadowSimulator, ShadowOrder

class ExecutionMode(Enum):
    SHADOW_ONLY = "SHADOW_ONLY"
    # LIVE = "LIVE"  # Commented out to strictly enforce OFF switch at code level

@dataclass(frozen=True)
class ExecutionConfig:
    mode: ExecutionMode = ExecutionMode.SHADOW_ONLY
    allow_live_broker: bool = False

class ExecutionRouter:
    """
    The 'OFF Switch' for live execution.
    Guarantees that orders are routed ONLY to the Shadow Simulator.
    """
    def __init__(self, simulator: ShadowSimulator):
        self._simulator = simulator
        self._config = ExecutionConfig(mode=ExecutionMode.SHADOW_ONLY, allow_live_broker=False)

    def route_order(self, order: ShadowOrder, market_price: float, spread_bps: float = 0.0) -> None:
        """
        Routes an order to the appropriate destination.
        
        CRITICAL SAFETY CHECK:
        If logic somehow requests LIVE execution, this router MUST block it 
        or raise an exception.
        """
        if self._config.mode == ExecutionMode.SHADOW_ONLY:
            self._simulator.place_order(order, market_price, spread_bps)
            return

        # Explicit safety catch
        raise RuntimeError("Live execution is STRICTLY PROHIBITED in current configuration.")

    def set_mode(self, mode: ExecutionMode) -> None:
        """
        Runtime mode switcher. 
        Currently only accepts SHADOW_ONLY.
        """
        if mode != ExecutionMode.SHADOW_ONLY:
             raise ValueError("Only SHADOW_ONLY mode is supported in this build.")
        # self._config = ... (Immutable in this implementation for safety)
