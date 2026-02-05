"""Compatibility helper for market regime detection.
Provides `safe_detect_market_regime` which calls available detectors
and normalizes the returned regime string.
"""
from typing import Any, Optional

def _normalize_regime(raw: Any) -> str:
    try:
        # dict with 'regime' key
        if isinstance(raw, dict) and 'regime' in raw:
            return str(raw['regime']).lower()
        # Enum-like with .value
        if hasattr(raw, 'value'):
            return str(raw.value).lower()
        # plain string
        if isinstance(raw, str):
            return raw.lower()
        # fallback
        return str(raw).lower()
    except Exception:
        return 'unknown'


def safe_detect_market_regime(detector_or_obj: Any, price_history: Optional[Any] = None) -> str:
    """Safely detect market regime.

    - If `detector_or_obj` has `detect_market_regime`, call it.
    - Else, try `detect_regime`.
    - Else, fall back to the canonical `get_regime_detector()` implementation.

    Returns a lowercase regime string (e.g., 'bull', 'bear', 'volatile', 'sideways', 'unknown').
    """
    # Local import to avoid circular imports
    try:
        from backend.utils.market_regime_detector import get_regime_detector
    except Exception:
        try:
            # try alternative import path
            from utils.market_regime_detector import get_regime_detector
        except Exception:
            get_regime_detector = None

    # 1) If object has detect_market_regime
    try:
        if hasattr(detector_or_obj, 'detect_market_regime'):
            try:
                raw = detector_or_obj.detect_market_regime(price_history)
                return _normalize_regime(raw)
            except TypeError:
                # maybe method expects no args
                raw = detector_or_obj.detect_market_regime()
                return _normalize_regime(raw)
    except Exception:
        pass

    # 2) If object has detect_regime
    try:
        if hasattr(detector_or_obj, 'detect_regime'):
            raw = detector_or_obj.detect_regime(price_history)
            return _normalize_regime(raw)
    except Exception:
        pass

    # 3) Try canonical global detector
    try:
        if get_regime_detector is not None:
            detector = get_regime_detector()
            raw = detector.detect_regime(price_history)
            return _normalize_regime(raw)
    except Exception:
        pass

    return 'unknown'
