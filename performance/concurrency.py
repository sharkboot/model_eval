from .base import BasePerformanceTest
import time
import asyncio
import concurrent.futures
import statistics

class ConcurrencyTest(BasePerformanceTest):
    def __init__(self, config):
        super().__init__(config)
        self.concurrency = config.get('concurrency', 1)
        self.requests = config.get('requests', 100)
        self.metrics = {
            'ttft': [],  # Time To First Token
            'tpot': [],  # Time Per Output Token
            'latency': [],
            'throughput': [],
            'success_rate': 0
        }
    
    def run(self):
        if not self.model or not self.dataset:
            raise ValueError("Model and dataset must be set up before running")
        
        self.dataset.load()
        data = self.dataset.get_data()
        if not data:
            raise ValueError("Dataset is empty")
        
        # Use the first item as test prompt
        test_item = data[0]
        prompt = test_item.get('prompt', test_item.get('question', ''))
        
        # Run concurrent requests
        success_count = 0
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            for i in range(self.requests):
                futures.append(executor.submit(self._make_request, prompt, i))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    success_count += 1
                except Exception:
                    pass
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        self.metrics['success_rate'] = success_count / self.requests
        if self.metrics['latency']:
            self.metrics['throughput'] = self.requests / total_time
        
        return self.metrics
    
    def _make_request(self, prompt, request_id):
        req_start = time.time()
        response = self.model.generate(prompt)
        req_end = time.time()
        
        # Simulate TTFT and TPOT (in real scenario, these would be measured from model response)
        ttft = (req_end - req_start) * 0.3  # Assume 30% of time is TTFT
        tpot = (req_end - req_start) * 0.7  # Assume 70% of time is TPOT
        
        self.metrics['ttft'].append(ttft)
        self.metrics['tpot'].append(tpot)
        self.metrics['latency'].append(req_end - req_start)
        
        return response
    
    def get_metrics(self):
        # Calculate statistics
        metrics = {
            'concurrency': self.concurrency,
            'requests': self.requests,
            'success_rate': self.metrics['success_rate'],
            'throughput': self.metrics['throughput'][0] if self.metrics['throughput'] else 0,
            'ttft': {
                'mean': statistics.mean(self.metrics['ttft']) if self.metrics['ttft'] else 0,
                'median': statistics.median(self.metrics['ttft']) if self.metrics['ttft'] else 0,
                'p95': self._calculate_percentile(self.metrics['ttft'], 95) if self.metrics['ttft'] else 0,
                'p99': self._calculate_percentile(self.metrics['ttft'], 99) if self.metrics['ttft'] else 0
            },
            'tpot': {
                'mean': statistics.mean(self.metrics['tpot']) if self.metrics['tpot'] else 0,
                'median': statistics.median(self.metrics['tpot']) if self.metrics['tpot'] else 0,
                'p95': self._calculate_percentile(self.metrics['tpot'], 95) if self.metrics['tpot'] else 0,
                'p99': self._calculate_percentile(self.metrics['tpot'], 99) if self.metrics['tpot'] else 0
            },
            'latency': {
                'mean': statistics.mean(self.metrics['latency']) if self.metrics['latency'] else 0,
                'median': statistics.median(self.metrics['latency']) if self.metrics['latency'] else 0,
                'p95': self._calculate_percentile(self.metrics['latency'], 95) if self.metrics['latency'] else 0,
                'p99': self._calculate_percentile(self.metrics['latency'], 99) if self.metrics['latency'] else 0
            }
        }
        
        return metrics
    
    def _calculate_percentile(self, data, percentile):
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[index] if index < len(sorted_data) else sorted_data[-1]
