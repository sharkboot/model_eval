# ============================================================
# cli/main.py
# ============================================================
import argparse
from core.config import load_config
from core.engine import EvaluationEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = EvaluationEngine(config)

    result = engine.run()
    print("Final Result:", result)

if __name__ == "__main__":
    main()