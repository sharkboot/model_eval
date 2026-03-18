from .base import BaseBackend

class ThirdPartyBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.framework = config.get('framework')
    
    def evaluate(self, model, dataset):
        raise NotImplementedError("Subclass must implement evaluate method")
    
    def get_backend_info(self):
        return {
            'backend_type': 'third_party',
            'framework': self.framework
        }

class OpenCompassBackend(ThirdPartyBackend):
    def __init__(self, config):
        config['framework'] = 'opencompass'
        super().__init__(config)
    
    def evaluate(self, model, dataset):
        try:
            from opencompass.runner import Runner
            runner = Runner(self.config)
            return runner.run(model, dataset)
        except ImportError:
            raise ImportError("Please install opencompass: pip install opencompass")

class MTEBBackend(ThirdPartyBackend):
    def __init__(self, config):
        config['framework'] = 'mteb'
        super().__init__(config)
    
    def evaluate(self, model, dataset):
        try:
            from mteb import MTEB
            mteb = MTEB(tasks=[self.config.get('task', 'STS12')])
            return mteb.run(model, output_folder=self.config.get('output_folder', './results'))
        except ImportError:
            raise ImportError("Please install mteb: pip install mteb")

class VLMEvalKitBackend(ThirdPartyBackend):
    def __init__(self, config):
        config['framework'] = 'vlm_eval_kit'
        super().__init__(config)
    
    def evaluate(self, model, dataset):
        try:
            from vlm_eval_kit.run import run_evaluation
            return run_evaluation(model, dataset, self.config)
        except ImportError:
            raise ImportError("Please install vlm_eval_kit")

class RAGASBackend(ThirdPartyBackend):
    def __init__(self, config):
        config['framework'] = 'ragas'
        super().__init__(config)
    
    def evaluate(self, model, dataset):
        try:
            from ragas import evaluate
            return evaluate(dataset, model, **self.config)
        except ImportError:
            raise ImportError("Please install ragas: pip install ragas")
