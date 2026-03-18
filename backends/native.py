from .base import BaseBackend
import time

class NativeBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = config.get('task_type', 'llm')
    
    def evaluate(self, model, dataset):
        results = []
        dataset.load()
        data = dataset.get_data()
        
        for i, item in enumerate(data):
            start_time = time.time()
            try:
                if self.task_type == 'llm':
                    prompt = item.get('prompt', item.get('question', ''))
                    response = model.generate(prompt)
                elif self.task_type == 'vlm':
                    prompt = item.get('prompt', '')
                    image = item.get('image', '')
                    response = model.generate(prompt, image=image)
                elif self.task_type == 'embedding':
                    text = item.get('text', '')
                    response = model.generate(text)
                elif self.task_type == 'reranker':
                    query = item.get('query', '')
                    documents = item.get('documents', [])
                    response = model.generate(query, documents=documents)
                elif self.task_type == 'aigc':
                    prompt = item.get('prompt', '')
                    response = model.generate(prompt)
                else:
                    raise ValueError(f"Unsupported task type: {self.task_type}")
                
                end_time = time.time()
                results.append({
                    'id': i,
                    'input': item,
                    'output': response,
                    'latency': end_time - start_time
                })
            except Exception as e:
                end_time = time.time()
                results.append({
                    'id': i,
                    'input': item,
                    'output': None,
                    'error': str(e),
                    'latency': end_time - start_time
                })
        
        return results
    
    def get_backend_info(self):
        return {
            'backend_type': 'native',
            'task_type': self.task_type
        }
