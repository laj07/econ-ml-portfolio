"""Experiment dispatcher — routes a config to the correct training function."""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[[dict], None]] = {}


def register(name: str) -> Callable:
    """Decorator: register an experiment trainer function by name."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def build_and_train(cfg: dict) -> None:
    exp_type = cfg.get("experiment_type")
    if not exp_type:
        raise ValueError("Config must contain 'experiment_type'.")

    # Import all experiment modules so they call @register at import time
    import src.experiments.economic_complexity  # noqa: F401
    import src.experiments.gdp_nowcasting       # noqa: F401
    import src.experiments.central_bank_nlp     # noqa: F401
    import src.experiments.informal_economy     # noqa: F401
    import src.experiments.labor_market         # noqa: F401

    if exp_type not in _REGISTRY:
        raise ValueError(
            f"Unknown experiment_type '{exp_type}'. "
            f"Registered: {sorted(_REGISTRY)}"
        )

    log.info("Dispatching to trainer: %s", exp_type)
    _REGISTRY[exp_type](cfg)
