"""
Entry point for all experiments.

Usage
-----
  python src/main.py --config experiments/central-bank-nlp/config.yaml
  python src/main.py --config experiments/economic-complexity/config.yaml --smoke
  python src/main.py --config experiments/gdp-nowcasting/config.yaml --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("main")


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving optional `include:` parent overrides."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(p) as f:
        cfg: dict = yaml.safe_load(f)

    # Resolve parent config (like thavlik's include: directive)
    if "include" in cfg:
        parent_path = p.parent / cfg.pop("include")
        with open(parent_path) as f:
            parent: dict = yaml.safe_load(f)
        parent.update(cfg)  # child keys override parent
        cfg = parent

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Economics ML Portfolio — experiment runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to experiment YAML config (e.g. experiments/central-bank-nlp/config.yaml)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: synthetic data, 1 epoch, no downloads required",
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda", "mps"],
        help="Override device. Defaults to CUDA if available, else CPU.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke:
        cfg.setdefault("exp_params", {})
        cfg["exp_params"]["smoke"] = True
        cfg["exp_params"]["max_epochs"] = 1
        log.info("Smoke-test mode: 1 epoch, synthetic data")

    if args.device:
        cfg.setdefault("exp_params", {})["device"] = args.device

    log.info("Experiment : %s", cfg.get("name", args.config))
    log.info("Type       : %s", cfg.get("experiment_type", "unknown"))

    # Late import so the module itself is not required at CLI parse time
    from src.trainer import build_and_train
    build_and_train(cfg)


if __name__ == "__main__":
    main()
