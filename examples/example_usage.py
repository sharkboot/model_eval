import json
import sys
import os
import argparse

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import OpenAIModel, LocalModel
from datasets import MMLUDataset, MCQDataset
from backends import NativeBackend, OpenCompassBackend
from performance import ConcurrencyTest
from reports import JSONReport, TableReport
from visualization import WandbVisualizer, GradioVisualizer

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Model Evaluation Framework')
    parser.add_argument('--config', type=str, default='configs/example_config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Initialize model
    model_config = config['model']['config']
    if config['model']['name'] == 'OpenAIModel':
        model = OpenAIModel(model_config)
    elif config['model']['name'] == 'LocalModel':
        model = LocalModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {config['model']['name']}")
    
    # Initialize dataset
    dataset_config = config['dataset']['config']
    if config['dataset']['name'] == 'MMLUDataset':
        dataset = MMLUDataset(dataset_config)
    elif config['dataset']['name'] == 'MCQDataset':
        dataset = MCQDataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset type: {config['dataset']['name']}")
    
    # Initialize backend
    backend_config = config['backend']['config']
    if config['backend']['name'] == 'NativeBackend':
        backend = NativeBackend(backend_config)
    elif config['backend']['name'] == 'OpenCompassBackend':
        backend = OpenCompassBackend(backend_config)
    else:
        raise ValueError(f"Unknown backend type: {config['backend']['name']}")
    
    # Run evaluation
    print("Running evaluation...")
    results = backend.evaluate(model, dataset)
    print(f"Evaluation completed. Results: {len(results)} items")
    
    # Generate report
    report_config = config['report']['config']
    if config['report']['name'] == 'JSONReport':
        report = JSONReport(report_config)
    elif config['report']['name'] == 'TableReport':
        report = TableReport(report_config)
    else:
        raise ValueError(f"Unknown report type: {config['report']['name']}")
    
    for result in results:
        report.add_result(result)
    
    report.save(report_config['output_path'])
    print(f"Report saved to: {report_config['output_path']}")
    
    # Run performance test
    print("Running performance test...")
    perf_config = config['performance']['config']
    perf_test = ConcurrencyTest(perf_config)
    perf_test.setup(model, dataset)
    perf_metrics = perf_test.run()
    print(f"Performance test completed. Metrics: {perf_metrics}")
    
    # Save performance metrics
    with open('results/performance_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(perf_metrics, f, ensure_ascii=False, indent=2)
    print("Performance metrics saved to: results/performance_metrics.json")
    
    # Visualize results
    viz_config = config['visualization']['config']
    if config['visualization']['name'] == 'WandbVisualizer':
        viz = WandbVisualizer(viz_config)
        viz.visualize(perf_metrics)
        viz.save()
        print("Results visualized with Wandb")
    elif config['visualization']['name'] == 'GradioVisualizer':
        viz = GradioVisualizer(viz_config)
        viz.visualize(results)
        print("Results visualized with Gradio")
    
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
