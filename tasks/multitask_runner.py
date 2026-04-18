
from core.registry import Registry
from tasks.standard_runner import StandardTaskRunner


class MultiTaskRunner:

    def __init__(self, config):
        self.config = config
        self.tasks = config.get("tasks", [])

    def _build_runner(self, task_cfg):
        task_type = task_cfg.get("type", "standard")

        if task_type == "standard":
            return StandardTaskRunner(task_cfg)

        else:
            return Registry.create(
                task_type,
                "tasks",
                **task_cfg.get("params", {})
            )

    def run(self):
        all_results = {}

        for task_cfg in self.tasks:
            task_name = task_cfg.get("name", "unknown")

            print(f"\n=== Running Task: {task_name} ===")

            runner = self._build_runner(task_cfg)

            result = runner.run()

            all_results[task_name] = result

            for k, v in result.items():
                print(f"{k}: {v:.4f}")

        return all_results