# ============================================================
# cli/main.py
# ============================================================
import argparse
from core.config import load_config
from core.engine import EvaluationEngine
from core.logger import setup_logger, get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    import logging
    log_level = getattr(logging, args.log_level)
    setup_logger(level=log_level)
    logger = get_logger()

    config = load_config(args.config)
    engine = EvaluationEngine(config)

    logger.info("Starting evaluation")
    result = engine.run()
    logger.info(f"Final Result: {result}")

if __name__ == "__main__":
    main()