# ============================================================
# cli/local_main.py
# ============================================================
from core.config import load_config
from core.engine import EvaluationEngine
from core.logger import setup_logger, get_logger


def main():

    setup_logger()
    logger = get_logger()

    config = load_config("E:\LLM\model_eval\configs/local_test.yaml")
    engine = EvaluationEngine(config)

    logger.info("Starting evaluation")
    result = engine.run()
    logger.info(f"Final Result: {result}")

if __name__ == "__main__":
    main()
