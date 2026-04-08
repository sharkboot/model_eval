import json
import sys
import os
import argparse
import time
import asyncio

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models import OpenAIModel, LocalModel, GenericAPIModel
from datasets import BaseDataset, StandardDataset, CustomDataset, ChineseSimpleQADataset, WritingBenchDataset, CEvalDataset, AIMEDataset, HMMTDataset
from backends import NativeBackend, ChineseSimpleQAEvaluator, AgentBackend, MultimodalBackend, WritingBenchEvaluator, CEvalEvaluator, AIMEEvaluator, HMMTEvaluator
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
    elif backend_name == 'AgentBackend':
        return AgentBackend(backend_config)
    elif backend_name == 'MultimodalBackend':
        return MultimodalBackend(backend_config)
    elif backend_name == 'WritingBenchEvaluator':
        return WritingBenchEvaluator(backend_config)
    elif backend_name == 'CEvalEvaluator':
        return CEvalEvaluator(backend_config)
    elif backend_name == 'AIMEEvaluator':
        return AIMEEvaluator(backend_config)
    elif backend_name == 'HMMTEvaluator':
        return HMMTEvaluator(backend_config)
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

def _score_to_value(score):
    """将评分转换为数值，支持多种类型"""
    # 处理字符串类型
    if isinstance(score, str):
        if score == 'A':
            return 2
        elif score == 'B':
            return 1
        elif score == 'C':
            return 0
        else:
            return 0
    # 处理数值类型
    elif isinstance(score, (int, float)):
        return float(score)
    # 处理字典类型
    elif isinstance(score, dict):
        return score.get('value', 0)
    # 其他类型
    else:
        return 0

def _value_to_score(value):
    """将数值转换为评分，支持多种类型"""
    if value >= 1.5:
        return 'A'
    elif value >= 0.5:
        return 'B'
    else:
        return 'C'

def _determine_final_score(runs, strategy):
    """根据策略确定最终得分"""
    scores = [run['score'] for run in runs]
    
    if strategy == 'highest':
        # 取最高分
        return max(scores, key=_score_to_value)
    elif strategy == 'lowest':
        # 取最低分
        return min(scores, key=_score_to_value)
    elif strategy == 'average':
        # 取平均值
        score_values = [_score_to_value(score) for score in scores]
        avg_value = sum(score_values) / len(score_values)
        return _value_to_score(avg_value)
    elif strategy == 'median':
        # 取中位数
        score_values = [_score_to_value(score) for score in scores]
        score_values.sort()
        mid = len(score_values) // 2
        median_value = score_values[mid] if len(score_values) % 2 == 1 else (score_values[mid-1] + score_values[mid]) / 2
        return _value_to_score(median_value)
    else:
        # 默认取最高分
        return max(scores, key=_score_to_value)

def evaluate_sync(model, dataset, backend, max_samples=None, num_runs=1, scoring_strategy='highest'):
    """同步评估"""
    results = []
    dataset.load()
    data = dataset.get_data()
    
    # 限制样本数量
    if max_samples is not None:
        data = data[:max_samples]
    
    for i, item in enumerate(data):
        start_time = time.time()
        try:
            # 转换为标准案例格式
            case = dataset.convert_to_case(item)
            
            # 执行多次评测
            runs = []
            for run in range(num_runs):
                # 生成模型响应
                response = model.generate(case['prompt'])
                # 执行评估
                score = backend.execute(model, case, response)
                runs.append({
                    'run_id': run,
                    'output': response,
                    'score': score
                })
            
            # 根据策略确定最终得分
            final_score = _determine_final_score(runs, scoring_strategy)
            
            end_time = time.time()
            results.append({
                'id': i,
                'input': item,
                'case': case,
                'runs': runs,  # 所有运行的结果
                'final_score': final_score,  # 最终得分
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

async def evaluate_async(model, dataset, backend, max_samples=None, num_runs=1, scoring_strategy='highest', concurrency_limit=5):
    """异步评估"""
    results = []
    dataset.load()
    data = dataset.get_data()
    
    # 限制样本数量
    if max_samples is not None:
        data = data[:max_samples]
    
    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    # 定义处理单个样本的异步函数
    async def process_item(i, item):
        async with semaphore:
            start_time = time.time()
            try:
                # 转换为标准案例格式
                case = dataset.convert_to_case(item)
                
                # 执行多次评测
                runs = []
                # 并发执行多次评测
                run_tasks = []
                for run in range(num_runs):
                    run_tasks.append(_async_run_evaluation(model, backend, case, run))
                runs = await asyncio.gather(*run_tasks)
                
                # 根据策略确定最终得分
                final_score = _determine_final_score(runs, scoring_strategy)
                
                end_time = time.time()
                return {
                    'id': i,
                    'input': item,
                    'case': case,
                    'runs': runs,  # 所有运行的结果
                    'final_score': final_score,  # 最终得分
                    'latency': end_time - start_time
                }
            except Exception as e:
                end_time = time.time()
                return {
                    'id': i,
                    'input': item,
                    'output': None,
                    'error': str(e),
                    'latency': end_time - start_time
                }
    
    # 并发处理所有样本
    tasks = []
    for i, item in enumerate(data):
        tasks.append(process_item(i, item))
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    return results

async def _async_run_evaluation(model, backend, case, run_id):
    """异步执行单次评测"""
    # 生成模型响应
    response = await model.async_generate(case['prompt'])
    # 执行评估
    score = await backend.async_execute(model, case, response)
    return {
        'run_id': run_id,
        'output': response,
        'score': score
    }

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
    use_async = config.get('evaluation', {}).get('async', False)
    concurrency_limit = config.get('evaluation', {}).get('concurrency_limit', 5)
    
    if use_async:
        # 异步评估
        print(f"Running async evaluation with concurrency limit: {concurrency_limit}")
        results = asyncio.run(evaluate_async(model, dataset, backend, max_samples, num_runs, scoring_strategy, concurrency_limit))
    else:
        # 同步评估
        print("Running sync evaluation")
        results = evaluate_sync(model, dataset, backend, max_samples, num_runs, scoring_strategy)
    
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
        # 计算WritingBench特定指标
        elif config['dataset']['type'] == 'writing_bench':
            calculate_writing_bench_metrics(results, output_path)
        # 计算C-Eval特定指标
        elif config['dataset']['type'] == 'ceval':
            calculate_ceval_metrics(results, output_path)
        # 计算AIME特定指标
        elif config['dataset']['type'] == 'aime':
            calculate_aime_metrics(results, output_path)
        # 计算HMMT特定指标
        elif config['dataset']['type'] == 'hmmt':
            calculate_hmmt_metrics(results, output_path)
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

def calculate_writing_bench_metrics(results, output_path):
    """计算WritingBench的特定指标"""
    total = len(results)
    total_score = 0.0
    domain_scores = {}
    
    for result in results:
        score = result.get('final_score', 0.0)
        total_score += score
        
        # 按领域统计
        domain = result.get('case', {}).get('metadata', {}).get('domain', 'Unknown')
        if domain not in domain_scores:
            domain_scores[domain] = {'total': 0, 'score': 0.0}
        domain_scores[domain]['total'] += 1
        domain_scores[domain]['score'] += score
    
    # 计算平均分数
    average_score = total_score / total if total > 0 else 0.0
    
    # 计算各领域的平均分数
    domain_averages = {}
    for domain, data in domain_scores.items():
        domain_averages[domain] = data['score'] / data['total'] if data['total'] > 0 else 0.0
    
    metrics = {
        'total': total,
        'average_score': average_score,
        'domain_scores': domain_averages
    }
    
    # 保存指标
    metrics_path = output_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("\nWritingBench Metrics:")
    print(f"Total: {total}")
    print(f"Average Score: {average_score:.4f}")
    print("Domain Scores:")
    for domain, score in domain_averages.items():
        print(f"  {domain}: {score:.4f}")
    print(f"Metrics saved to: {metrics_path}")

def calculate_ceval_metrics(results, output_path):
    """计算C-Eval的特定指标"""
    total = len(results)
    correct = 0
    incorrect = 0
    subject_scores = {}
    category_scores = {}
    
    for result in results:
        score = result.get('final_score', 0.0)
        if score == 1.0:
            correct += 1
        else:
            incorrect += 1
        
        # 按学科统计
        subject = result.get('case', {}).get('metadata', {}).get('subject', 'Unknown')
        if subject not in subject_scores:
            subject_scores[subject] = {'total': 0, 'correct': 0}
        subject_scores[subject]['total'] += 1
        if score == 1.0:
            subject_scores[subject]['correct'] += 1
        
        # 按分类统计
        category = result.get('case', {}).get('metadata', {}).get('category', 'Unknown')
        if category not in category_scores:
            category_scores[category] = {'total': 0, 'correct': 0}
        category_scores[category]['total'] += 1
        if score == 1.0:
            category_scores[category]['correct'] += 1
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 计算各学科的准确率
    subject_accuracies = {}
    for subject, data in subject_scores.items():
        subject_accuracies[subject] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
    
    # 计算各类别的准确率
    category_accuracies = {}
    for category, data in category_scores.items():
        category_accuracies[category] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
    
    metrics = {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'subject_accuracies': subject_accuracies,
        'category_accuracies': category_accuracies
    }
    
    # 保存指标
    metrics_path = output_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("\nC-Eval Metrics:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Subject Accuracies:")
    for subject, acc in subject_accuracies.items():
        print(f"  {subject}: {acc:.4f}")
    print("Category Accuracies:")
    for category, acc in category_accuracies.items():
        print(f"  {category}: {acc:.4f}")
    print(f"Metrics saved to: {metrics_path}")

def calculate_aime_metrics(results, output_path):
    """计算AIME的特定指标"""
    total = len(results)
    correct = 0
    incorrect = 0
    year_scores = {}
    problem_scores = {}
    
    for result in results:
        score = result.get('final_score', 0.0)
        if score == 1.0:
            correct += 1
        else:
            incorrect += 1
        
        # 按年份统计
        year = result.get('case', {}).get('metadata', {}).get('year', 'Unknown')
        if year not in year_scores:
            year_scores[year] = {'total': 0, 'correct': 0}
        year_scores[year]['total'] += 1
        if score == 1.0:
            year_scores[year]['correct'] += 1
        
        # 按问题编号统计
        problem_number = result.get('case', {}).get('metadata', {}).get('problem_number', 'Unknown')
        if problem_number not in problem_scores:
            problem_scores[problem_number] = {'total': 0, 'correct': 0}
        problem_scores[problem_number]['total'] += 1
        if score == 1.0:
            problem_scores[problem_number]['correct'] += 1
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 计算各年份的准确率
    year_accuracies = {}
    for year, data in year_scores.items():
        year_accuracies[year] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
    
    # 计算各问题编号的准确率
    problem_accuracies = {}
    for problem_number, data in problem_scores.items():
        problem_accuracies[problem_number] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
    
    metrics = {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'year_accuracies': year_accuracies,
        'problem_accuracies': problem_accuracies
    }
    
    # 保存指标
    metrics_path = output_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("\nAIME Metrics:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Year Accuracies:")
    for year, acc in year_accuracies.items():
        print(f"  {year}: {acc:.4f}")
    print("Problem Accuracies:")
    for problem_number, acc in problem_accuracies.items():
        print(f"  Problem {problem_number}: {acc:.4f}")
    print(f"Metrics saved to: {metrics_path}")

def calculate_hmmt_metrics(results, output_path):
    """计算HMMT的特定指标"""
    total = len(results)
    correct = 0
    incorrect = 0
    category_scores = {}
    round_scores = {}
    year_scores = {}
    
    for result in results:
        score = result.get('final_score', 0.0)
        if score == 1.0:
            correct += 1
        else:
            incorrect += 1
        
        # 按类别统计
        category = result.get('case', {}).get('metadata', {}).get('category', 'Unknown')
        if category not in category_scores:
            category_scores[category] = {'total': 0, 'correct': 0}
        category_scores[category]['total'] += 1
        if score == 1.0:
            category_scores[category]['correct'] += 1
        
        # 按轮次类型统计
        round_type = result.get('case', {}).get('metadata', {}).get('round_type', 'Unknown')
        if round_type not in round_scores:
            round_scores[round_type] = {'total': 0, 'correct': 0}
        round_scores[round_type]['total'] += 1
        if score == 1.0:
            round_scores[round_type]['correct'] += 1
        
        # 按年份统计
        year = result.get('case', {}).get('metadata', {}).get('year', 'Unknown')
        if year not in year_scores:
            year_scores[year] = {'total': 0, 'correct': 0}
        year_scores[year]['total'] += 1
        if score == 1.0:
            year_scores[year]['correct'] += 1
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 计算各类别的准确率
    category_accuracies = {}
    for category, data in category_scores.items():
        category_accuracies[category] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
    
    # 计算各轮次类型的准确率
    round_accuracies = {}
    for round_type, data in round_scores.items():
        round_accuracies[round_type] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
    
    # 计算各年份的准确率
    year_accuracies = {}
    for year, data in year_scores.items():
        year_accuracies[year] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
    
    metrics = {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'category_accuracies': category_accuracies,
        'round_accuracies': round_accuracies,
        'year_accuracies': year_accuracies
    }
    
    # 保存指标
    metrics_path = output_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("\nHMMT Metrics:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Category Accuracies:")
    for category, acc in category_accuracies.items():
        print(f"  {category}: {acc:.4f}")
    print("Round Type Accuracies:")
    for round_type, acc in round_accuracies.items():
        print(f"  {round_type}: {acc:.4f}")
    print("Year Accuracies:")
    for year, acc in year_accuracies.items():
        print(f"  {year}: {acc:.4f}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
