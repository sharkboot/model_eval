
from core.registry import Registry
from tasks.standard_runner import StandardTaskRunner


from core.logger import get_logger

logger = get_logger()


class MultiTaskRunner:

    def __init__(self, config):
        self.config = config
        self.tasks = config.get("tasks", [])
        self.visualizer = None

    def set_visualizer(self, visualizer):
        """设置可视化器"""
        self.visualizer = visualizer

    def _build_runner(self, task_cfg):
        task_type = task_cfg.get("type", "standard")

        if task_type == "standard":
            runner = StandardTaskRunner(task_cfg)
            if self.visualizer:
                runner.set_visualizer(self.visualizer)
            return runner

        else:
            runner = Registry.create(
                task_type,
                "tasks",
                **task_cfg.get("params", {})
            )
            if self.visualizer and hasattr(runner, 'set_visualizer'):
                runner.set_visualizer(self.visualizer)
            return runner

    def run(self):
        all_results = {}

        for task_cfg in self.tasks:
            task_name = task_cfg.get("name", "unknown")

            logger.info(f"Running Task: {task_name}")

            runner = self._build_runner(task_cfg)

            result = runner.run()

            all_results[task_name] = result

            for k, v in result.items():
                logger.info(f"{k}: {v:.4f}")

        return all_results