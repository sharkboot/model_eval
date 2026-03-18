from .base import BaseReport
import json
import csv
import logging
from datetime import datetime

class JSONReport(BaseReport):
    def generate(self):
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': self.results
        }
        return report
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.generate(), f, ensure_ascii=False, indent=2)

class TableReport(BaseReport):
    def generate(self):
        if not self.results:
            return []
        
        # Extract headers from the first result
        headers = list(self.results[0].keys())
        rows = []
        
        for result in self.results:
            row = [result.get(key, '') for key in headers]
            rows.append(row)
        
        return {'headers': headers, 'rows': rows}
    
    def save(self, path):
        report = self.generate()
        if not report:
            return
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(report['headers'])
            writer.writerows(report['rows'])

class LogReport(BaseReport):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger('eval_report')
        if not self.logger.handlers:
            handler = logging.FileHandler('eval.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def generate(self):
        log_entries = []
        for result in self.results:
            log_entry = f"Result: {json.dumps(result, ensure_ascii=False)}"
            log_entries.append(log_entry)
        return log_entries
    
    def save(self, path):
        # Log to file
        for entry in self.generate():
            self.logger.info(entry)
        
        # Also save to specified path
        with open(path, 'w', encoding='utf-8') as f:
            for entry in self.generate():
                f.write(entry + '\n')
