# core/engine.py
from core.auto_import import auto_import
from tasks.multitask_runner import MultiTaskRunner
from core.leaderboard import Leaderboard


class EvaluationEngine:

    def __init__(self, config):
        self.config = config

        auto_import("datasets")
        auto_import("models")
        auto_import("evaluators")
        auto_import("tasks")
        auto_import("prompt_builder")

    def run(self):

        # 判断是否多任务
        if "tasks" in self.config:
            runner = MultiTaskRunner(self.config)
            results = runner.run()
        else:
            # 单任务兼容
            from tasks.standard_runner import StandardTaskRunner
            runner = StandardTaskRunner(self.config)
            results = {"single_task": runner.run()}

        # 汇总 leaderboard
        leaderboard = Leaderboard()

        for task_name, metrics in results.items():
            leaderboard.add(task_name, metrics)

        leaderboard.pretty_print()

        return leaderboard.summary()
