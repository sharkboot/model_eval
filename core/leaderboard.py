# core/leaderboard.py

class Leaderboard:

    def __init__(self):
        self.results = {}

    def add(self, task_name, metrics: dict):
        self.results[task_name] = metrics

    def summary(self):
        return self.results

    def pretty_print(self):
        print("\n===== Leaderboard =====")
        for task, metrics in self.results.items():
            print(f"\n[{task}]")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")