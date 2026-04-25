# core/leaderboard.py

from core.logger import get_logger

logger = get_logger()


class Leaderboard:

    def __init__(self):
        self.results = {}

    def add(self, task_name, metrics: dict):
        self.results[task_name] = metrics

    def summary(self):
        return self.results

    def pretty_print(self):
        logger.info("===== Leaderboard =====")
        for task, metrics in self.results.items():
            logger.info(f"[{task}]")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}")