from typing import List, Dict, Any
from core.config import RunConfig, DatasetConfig, BenchmarkConfig
from datasets.registry import DatasetRegistry
from models.registry import ModelRegistry
from evaluators.registry import EvaluatorRegistry
from core.base import DataItem, ModelInput

class EvaluationRunner:
    """
    评测运行类：支持混合任务调度
    """
    
    def __init__(self, config: RunConfig):
        self.config = config
        
    def _prepare_dataset(self, ds_config: DatasetConfig) -> List[DataItem]:
        """
        加载单个数据集，并应用 limit 和 filter
        """
        # 从配置中获取数据集配置
        dataset_config = {}
        # 这里简化处理，直接使用默认配置
        if ds_config.name == 'chinese_simpleqa':
            dataset_config = {'data_path': 'data/chinese_simpleqa.json'}
        elif ds_config.name == 'writing_bench':
            dataset_config = {'data_path': 'data/writing_bench.json'}
        elif ds_config.name == 'ceval':
            dataset_config = {'data_path': 'data/ceval.json'}
        # 添加其他数据集的默认配置
        
        dataset = DatasetRegistry.get(ds_config.name)(dataset_config)
        raw_data = dataset.load()
        
        if ds_config.filter:
            raw_data = ds_config.filter.apply(raw_data)
        if ds_config.limit is not None:
            raw_data = raw_data[:ds_config.limit]
            
        return raw_data

    def _run_single_dataset(self, ds_config: DatasetConfig) -> Dict[str, Any]:
        """
        执行单个数据集的评测
        """
        data_items = self._prepare_dataset(ds_config)
        if not data_items: return {"score": 0, "report": "No data"}

        # 暂时跳过引擎设置和运行步骤，直接返回模拟结果
        # engine, collector = self._setup_engine(data_items, ds_config.name)
        # for _ in range(self.config.rounds): engine.run()
        # 
        # results = collector.get_all_results()
        # analyzer = DatasetAnalyzer(dataset=None, results=results)
        # report = analyzer.generate_report()
        
        # 模拟报告
        report = {"accuracy": 0.8, "total_items": len(data_items)}
        
        return {
            "name": ds_config.name,
            "type": "dataset",
            "score": report.get("accuracy", 0.0), 
            "report": report
        }

    def _run_single_benchmark(self, bench_config: BenchmarkConfig) -> Dict[str, Any]:
        """
        执行单个 Benchmark 的评测与分数聚合
        """
        print(f"\n 开始评测基准: {bench_config.name}")
        benchmark_details = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for ds_config in bench_config.datasets:
            result = self._run_single_dataset(ds_config)
            benchmark_details[ds_config.name] = result
            
            score = result["score"]
            weighted_score = score * ds_config.weight
            total_weighted_score += weighted_score
            total_weight += ds_config.weight
            
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        return {
            "name": bench_config.name,
            "type": "benchmark",
            "final_score": final_score,
            "details": benchmark_details
        }

    def run(self) -> Dict[str, Any]:
        """
        执行所有任务
        """
        all_results = []
        
        for task in self.config.tasks:
            if isinstance(task, BenchmarkConfig):
                result = self._run_single_benchmark(task)
            elif isinstance(task, DatasetConfig):
                result = self._run_single_dataset(task)
            else:
                raise ValueError("Invalid task type in RunConfig")
                
            all_results.append(result)
            
        return {
            "summary": {r["name"]: r.get("final_score", r["score"]) for r in all_results},
            "details": all_results
        }
        
    def _setup_engine(self, data_items, dataset_name):
        # 构建 EvaluationEngine 的标准逻辑
        # 这里需要根据实际情况实现
        pass

class DatasetAnalyzer:
    def __init__(self, dataset, results):
        self.dataset = dataset
        self.results = results
    
    def generate_report(self):
        # 生成报告的逻辑
        # 这里需要根据实际情况实现
        return {"accuracy": 0.0}
