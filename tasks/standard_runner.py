from tasks.task_runner import BaseTaskRunner
from core.registry import Registry


class StandardTaskRunner(BaseTaskRunner):

    def __init__(self, config):
        self.config = config

        # Dataset
        self.dataset = Registry.create(
            config["dataset"]["name"],
            "dataset",
            **config["dataset"].get("params", {})
        )

        # Model
        self.model = Registry.create(
            config["model"]["name"],
            "model",
            **config["model"].get("params", {})
        )

        # Evaluators
        self.evaluators = [
            Registry.create(e["name"], "evaluator", **e.get("params", {}))
            for e in config["evaluators"]
        ]

        # Prompt

    def run(self):
        data = self.dataset.load()
        results = {}

        for item in data:
            prompt = item.prompt
            output = self.model.generate(prompt)

            for evaluator in self.evaluators:
                metrics = evaluator.evaluate(output.text, item)

        return {k: sum(v) / len(v) for k, v in results.items()}
