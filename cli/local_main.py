# ============================================================
# cli/main.py
# ============================================================
import argparse
from core.config import load_config
from core.engine import EvaluationEngine


def main():


    config = load_config("E:\LLM\model_eval\configs/local_test.yaml")
    engine = EvaluationEngine(config)

    result = engine.run()
    print("Final Result:", result)

if __name__ == "__main__":
    main()