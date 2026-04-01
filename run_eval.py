import json
import sys
import os
import argparse

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models import OpenAIModel, LocalModel, GenericAPIModel
from datasets import MMLUDataset, MCQDataset, StandardDataset, CustomDataset, ChineseSimpleQADataset
from backends import NativeBackend, ChineseSimpleQAEvaluator
from performance import ConcurrencyTest
from reports import JSONReport, TableReport
from visualization import GradioVisualizer

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model(config):
    # 支持两种配置格式
    if 'model' in config and isinstance(config['model'], dict):
        if 'config' in config['model']:
            # 旧格式：model.config
            model_config = config['model']['config']
            model_name = config['model']['name']
        else:
            # 新格式：直接使用model字段
            model_config = config['model']
            model_name = 'OpenAIModel'  # 默认使用OpenAIModel
    else:
        # 裁判模型的配置格式
        model_config = config.get('config', config)
        model_name = config.get('name', 'OpenAIModel')
    
    if model_name == 'OpenAIModel':
        return OpenAIModel(model_config)
    elif model_name == 'LocalModel':
        return LocalModel(model_config)
    elif model_name == 'GenericAPIModel':
        return GenericAPIModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def get_dataset(config):
    # 支持两种配置格式
    if 'dataset' in config and 'config' in config['dataset']:
        dataset_config = config['dataset']['config']
        dataset_name = config['dataset']['name']
    else:
        # 直接使用dataset作为配置
        dataset_config = config['dataset']
        dataset_name = dataset_config['type']
    
    if dataset_name == 'MMLUDataset':
        return MMLUDataset(dataset_config)
    elif dataset_name == 'MCQDataset':
        return MCQDataset(dataset_config)
    elif dataset_name == 'StandardDataset':
        return StandardDataset(dataset_config)
    elif dataset_name == 'CustomDataset':
        return CustomDataset(dataset_config)
    elif dataset_name == 'chinese_simpleqa':
        return ChineseSimpleQADataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")

def get_backend(config):
    # 支持两种配置格式
    if 'backend' in config and isinstance(config['backend'], dict):
        if 'config' in config['backend']:
            # 旧格式：backend.config
            backend_config = config['backend']['config']
            backend_name = config['backend']['name']
        else:
            # 新格式：直接使用backend字段
            backend_config = config['backend']
            backend_name = backend_config.get('type', 'NativeBackend')  # 默认使用NativeBackend
    else:
        raise ValueError("Backend configuration not found")
    
    # 检查是否需要裁判模型
    if backend_config.get('eval_type') == 'model' and 'judge_model' in config:
        judge_model_config = config['judge_model']
        print(f"Initializing judge model: {judge_model_config}")
        judge_model = get_model({'model': judge_model_config})
        backend_config['judge_model'] = judge_model
        print(f"Judge model initialized: {judge_model.get_model_info()}")
    
    if backend_name == 'NativeBackend':
        return NativeBackend(backend_config)
    elif backend_name == 'ChineseSimpleQAEvaluator':
        return ChineseSimpleQAEvaluator(backend_config)
    else:
        raise ValueError(f"Unknown backend type: {backend_name}")

def get_report(config):
    report_config = config['report']['config']
    report_name = config['report']['name']
    
    if report_name == 'JSONReport':
        return JSONReport(report_config)
    elif report_name == 'TableReport':
        return TableReport(report_config)
    else:
        raise ValueError(f"Unknown report type: {report_name}")

def get_visualizer(config):
    viz_config = config['visualization']['config']
    viz_name = config['visualization']['name']
    
    if viz_name == 'GradioVisualizer':
        try:
            from visualization import GradioVisualizer
            return GradioVisualizer(viz_config)
        except ImportError:
            print("Gradio is not installed. Visualization will be skipped.")
            return None
    else:
        raise ValueError(f"Unknown visualizer type: {viz_name}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Model Evaluation Framework')
    parser.add_argument('--config', type=str, default='configs/example_config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    model = get_model(config)
    dataset = get_dataset(config)
    backend = get_backend(config)
    
    # Run evaluation
    print("\nRunning evaluation...")
    max_samples = config.get('evaluation', {}).get('max_samples', None)
    num_runs = config.get('evaluation', {}).get('num_runs', 1)
    scoring_strategy = config.get('evaluation', {}).get('scoring_strategy', 'highest')
    results = backend.evaluate(model, dataset, max_samples, num_runs, scoring_strategy)
    print(f"Evaluation completed. Results: {len(results)} items")
    
    # Generate report
    print("\nGenerating report...")
    if 'evaluation' in config:
        output_path = config['evaluation'].get('output_path', 'results/eval_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Report saved to: {output_path}")
        
        # 计算Chinese SimpleQA特定指标
        if config['dataset']['type'] == 'chinese_simpleqa':
            calculate_chinese_simpleqa_metrics(results, output_path)
    else:
        # 兼容旧格式
        report = get_report(config)
        for result in results:
            report.add_result(result)
        report.save(config['report']['config']['output_path'])
        print(f"Report saved to: {config['report']['config']['output_path']}")
    
    # Run performance test (optional)
    if 'performance' in config:
        print("\nRunning performance test...")
        perf_config = config['performance']['config']
        perf_test = ConcurrencyTest(perf_config)
        perf_test.setup(model, dataset)
        perf_metrics = perf_test.run()
        print(f"Performance test completed.")
        
        # Save performance metrics
        with open('results/performance_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(perf_metrics, f, ensure_ascii=False, indent=2)
        print("Performance metrics saved to: results/performance_metrics.json")
    
    print("\nAll tasks completed successfully!")

def calculate_chinese_simpleqa_metrics(results, output_path):
    """计算Chinese SimpleQA的特定指标"""
    total = len(results)
    correct = 0
    incorrect = 0
    not_attempted = 0
    
    for result in results:
        score = result.get('final_score', result.get('score'))
        if score == 'A':
            correct += 1
        elif score == 'B':
            incorrect += 1
        elif score == 'C':
            not_attempted += 1
    
    # 计算指标
    CO = correct / total if total > 0 else 0
    NA = not_attempted / total if total > 0 else 0
    IN = incorrect / total if total > 0 else 0
    CGA = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
    F_score = 2 * CGA * CO / (CGA + CO) if (CGA + CO) > 0 else 0
    
    metrics = {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'not_attempted': not_attempted,
        'CO': CO,
        'NA': NA,
        'IN': IN,
        'CGA': CGA,
        'F_score': F_score
    }
    
    # 保存指标
    metrics_path = output_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("\nChinese SimpleQA Metrics:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Not Attempted: {not_attempted}")
    print(f"CO (Correct): {CO:.4f}")
    print(f"NA (Not Attempted): {NA:.4f}")
    print(f"IN (Incorrect): {IN:.4f}")
    print(f"CGA (Correct given Attempted): {CGA:.4f}")
    print(f"F-score: {F_score:.4f}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
