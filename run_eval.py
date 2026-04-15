import json
import sys
import os
import argparse

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models import OpenAIModel, LocalModel, GenericAPIModel
from datasets import StandardDataset, CustomDataset, ChineseSimpleQADataset, WritingBenchDataset, CEvalDataset, AIMEDataset, HMMTDataset, AMODataset, IMODataset, SuperGPQADataset, EQBenchDataset
from evaluators import NativeEvaluator, ChineseSimpleQAEvaluator, AgentEvaluator, MultimodalEvaluator, WritingBenchEvaluator, CEvalEvaluator, AIMEEvaluator, HMMTEvaluator, AMOEvaluator, IMOMEvaluator, SuperGPQAEvaluator, EQBenchEvaluator, ThirdPartyEvaluator
from performance import ConcurrencyTest

# Import new architecture components
from datasets.registry import DatasetRegistry
from models.registry import ModelRegistry
from evaluators.registry import EvaluatorRegistry
from core.config import RunConfig, DatasetConfig, BenchmarkConfig
from core.engine import EvaluationRunner

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
    
    if dataset_name == 'StandardDataset':
        return StandardDataset(dataset_config)
    elif dataset_name == 'CustomDataset':
        return CustomDataset(dataset_config)
    elif dataset_name == 'chinese_simpleqa':
        return ChineseSimpleQADataset(dataset_config)
    elif dataset_name == 'writing_bench':
        return WritingBenchDataset(dataset_config)
    elif dataset_name == 'ceval':
        return CEvalDataset(dataset_config)
    elif dataset_name == 'aime':
        return AIMEDataset(dataset_config)
    elif dataset_name == 'hmmt':
        return HMMTDataset(dataset_config)
    elif dataset_name == 'amo':
        return AMODataset(dataset_config)
    elif dataset_name == 'imo':
        return IMODataset(dataset_config)
    elif dataset_name == 'supergpqa':
        return SuperGPQADataset(dataset_config)
    elif dataset_name == 'eq_bench':
        return EQBenchDataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Model Evaluation Framework')
    parser.add_argument('--config', type=str, default='configs/example_config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Register components
    print("Registering components...")
    # Register datasets
    DatasetRegistry.register('StandardDataset', StandardDataset)
    DatasetRegistry.register('CustomDataset', CustomDataset)
    DatasetRegistry.register('chinese_simpleqa', ChineseSimpleQADataset)
    DatasetRegistry.register('writing_bench', WritingBenchDataset)
    DatasetRegistry.register('ceval', CEvalDataset)
    DatasetRegistry.register('aime', AIMEDataset)
    DatasetRegistry.register('hmmt', HMMTDataset)
    DatasetRegistry.register('amo', AMODataset)
    DatasetRegistry.register('imo', IMODataset)
    DatasetRegistry.register('supergpqa', SuperGPQADataset)
    DatasetRegistry.register('eq_bench', EQBenchDataset)
    
    # Register models
    ModelRegistry.register('OpenAIModel', OpenAIModel)
    ModelRegistry.register('LocalModel', LocalModel)
    ModelRegistry.register('GenericAPIModel', GenericAPIModel)
    
    # Register evaluators
    EvaluatorRegistry.register('NativeEvaluator', NativeEvaluator)
    EvaluatorRegistry.register('ChineseSimpleQAEvaluator', ChineseSimpleQAEvaluator)
    EvaluatorRegistry.register('AgentEvaluator', AgentEvaluator)
    EvaluatorRegistry.register('MultimodalEvaluator', MultimodalEvaluator)
    EvaluatorRegistry.register('WritingBenchEvaluator', WritingBenchEvaluator)
    EvaluatorRegistry.register('CEvalEvaluator', CEvalEvaluator)
    EvaluatorRegistry.register('AIMEEvaluator', AIMEEvaluator)
    EvaluatorRegistry.register('HMMTEvaluator', HMMTEvaluator)
    EvaluatorRegistry.register('AMOEvaluator', AMOEvaluator)
    EvaluatorRegistry.register('IMOMEvaluator', IMOMEvaluator)
    EvaluatorRegistry.register('SuperGPQAEvaluator', SuperGPQAEvaluator)
    EvaluatorRegistry.register('EQBenchEvaluator', EQBenchEvaluator)
    
    # Create run config
    print("Creating run configuration...")
    # 从配置文件中提取任务信息
    dataset_config = DatasetConfig(
        name=config['dataset']['type'],
        limit=config.get('evaluation', {}).get('max_samples', None)
    )
    
    run_config = RunConfig(
        tasks=[dataset_config],
        evaluator_configs=[{'name': config['backend'].get('type', 'NativeEvaluator')}],
        rounds=config.get('evaluation', {}).get('num_runs', 1),
        model_config=config['model']
    )
    
    # Run evaluation using new architecture
    print("\nRunning evaluation...")
    runner = EvaluationRunner(run_config)
    results = runner.run()
    
    print(f"Evaluation completed. Results: {len(results['details'])} items")
    
    # Generate report
    print("\nGenerating report...")
    output_path = config.get('evaluation', {}).get('output_path', 'results/eval_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Report saved to: {output_path}")
    
    # Run performance test (optional)
    if 'performance' in config:
        print("\nRunning performance test...")
        model = get_model(config)
        dataset = get_dataset(config)
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

if __name__ == "__main__":
    main()

