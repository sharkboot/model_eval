# core/engine.py
import threading
from core.auto_import import auto_import
from tasks.multitask_runner import MultiTaskRunner
from core.leaderboard import Leaderboard


class EvaluationEngine:

    def __init__(self, config):
        self.config = config
        self.visualizer = None

        auto_import("datasets")
        auto_import("models")
        auto_import("tasks")
        auto_import("prompt_builder")
        auto_import("adapter")  # 包含 adapter/evaluators
        auto_import("reports")  # 报告格式

    def _setup_visualizer(self):
        """设置可视化监控"""
        monitor_config = self.config.get("monitor")
        if not monitor_config:
            return

        monitor_type = monitor_config if isinstance(monitor_config, str) else monitor_config.get("type", "gradio")

        if monitor_type == "gradio":
            try:
                from visualization.platforms import GradioVisualizer
                self.visualizer = GradioVisualizer(monitor_config if isinstance(monitor_config, dict) else {})
                self.visualizer.setup()
            except Exception as e:
                print(f"Failed to setup Gradio visualizer: {e}")

    def run(self):

        # 设置可视化
        self._setup_visualizer()

        # 启动可视化服务器（不阻塞）
        if self.visualizer:
            print("Starting visualization server in background...")
            viz_thread = threading.Thread(target=self.visualizer.visualize, daemon=True)
            viz_thread.start()

        # 判断是否多任务
        if "tasks" in self.config:
            runner = MultiTaskRunner(self.config)
            if self.visualizer:
                runner.set_visualizer(self.visualizer)
            results = runner.run()
        else:
            # 单任务兼容
            from tasks.standard_runner import StandardTaskRunner
            runner = StandardTaskRunner(self.config)
            if self.visualizer:
                runner.set_visualizer(self.visualizer)
            results = {"single_task": runner.run()}

        # 汇总 leaderboard
        leaderboard = Leaderboard()

        for task_name, metrics in results.items():
            leaderboard.add(task_name, metrics)

        leaderboard.pretty_print()

        # 保持可视化运行（仅当不是 daemon 线程时）
        if self.visualizer:
            input("Press Enter to stop visualization...")

        return leaderboard.summary()
