from .base import BaseTool

class ToolBench(BaseTool):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = config.get('task_type', 'general')
    
    def setup(self):
        try:
            import toolbench
            return True
        except ImportError:
            raise ImportError("Please install toolbench")
    
    def run(self, model):
        # Simulate ToolBench evaluation
        results = []
        for i in range(10):  # Run 10 sample tasks
            try:
                prompt = f"ToolBench task {i}: Find the weather in New York"
                response = model.generate(prompt)
                results.append({
                    'task_id': i,
                    'prompt': prompt,
                    'response': response,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'task_id': i,
                    'prompt': prompt,
                    'response': None,
                    'error': str(e),
                    'success': False
                })
        return results
    
    def get_results(self):
        # Return sample results
        return {
            'tool': 'ToolBench',
            'task_type': self.task_type,
            'metrics': {
                'success_rate': 0.8,
                'average_score': 0.75
            }
        }

class NeedleInHaystack(BaseTool):
    def __init__(self, config):
        super().__init__(config)
        self.document_size = config.get('document_size', 10000)
        self.needle = config.get('needle', 'The quick brown fox jumps over the lazy dog')
    
    def setup(self):
        # Generate sample document with needle
        self.document = ' '.join(['Lorem ipsum dolor sit amet'] * (self.document_size // 20))
        # Insert needle at random position
        import random
        pos = random.randint(0, len(self.document) - len(self.needle))
        self.document = self.document[:pos] + self.needle + self.document[pos:]
        return True
    
    def run(self, model):
        prompt = f"Find the following sentence in the document: '{self.needle}'\n\nDocument: {self.document}"
        response = model.generate(prompt)
        return response
    
    def get_results(self):
        # Return sample results
        return {
            'tool': 'Needle-in-a-Haystack',
            'document_size': self.document_size,
            'metrics': {
                'success': True,
                'retrieval_accuracy': 0.9
            }
        }

class BFCL(BaseTool):
    def __init__(self, config):
        super().__init__(config)
        self.version = config.get('version', 'v3')
    
    def setup(self):
        try:
            import bfcl
            return True
        except ImportError:
            raise ImportError("Please install bfcl")
    
    def run(self, model):
        # Simulate BFCL evaluation
        results = []
        for i in range(5):  # Run 5 sample tasks
            try:
                prompt = f"BFCL task {i}: Solve the math problem: 2 + 2 * 3"
                response = model.generate(prompt)
                results.append({
                    'task_id': i,
                    'prompt': prompt,
                    'response': response,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'task_id': i,
                    'prompt': prompt,
                    'response': None,
                    'error': str(e),
                    'success': False
                })
        return results
    
    def get_results(self):
        # Return sample results
        return {
            'tool': f'BFCL-{self.version}',
            'metrics': {
                'success_rate': 0.9,
                'average_score': 0.85
            }
        }
