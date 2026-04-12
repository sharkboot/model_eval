

以下是包含完整基类定义的 **评测框架 API 接口与执行规范 (v3.0)** 完整文档。

---

### 评测框架 API 接口与执行规范 (v3.0)

**最后更新时间：** 2026年4月12日
**版本：** 3.0 (完整版)

---

#### 1. 概述
本规范定义了评测框架的完整架构。v3.0 版本整合了数据定义、核心基类、注册机制、通用筛选器以及支持混合任务调度的运行配置。所有实现必须遵循此规范以保证框架的通用性与扩展性。

---

#### 2. 数据类规范
所有数据在框架中流转必须遵循标准数据单元定义。

**2.1 DataItem**
标准数据单元，所有数据集加载后必须转换为 `List[DataItem]` 格式。

```python
@dataclass
class DataItem:
    id: str # 唯一标识符
    prompt: str # 模型输入提示
    reference: Any # 参考答案
    metadata: Dict[str, Any] = field(default_factory=dict) # 元数据字典
    category: List[str] = field(default_factory=list) # 分类标签列表 (支持多标签)
    difficulty: Optional[str] = None # 难度级别
    extra: Dict[str, Any] = field(default_factory=dict) # 扩展字段
```

| 字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| **id** | str | 是 | 唯一标识符 |
| **prompt** | str | 是 | 模型的输入提示 |
| **reference** | Any | 是 | 参考答案 |
| **category** | **List[str]** | 否 | **分类标签列表**，支持多标签分类 |
| **difficulty** | str | 否 | 难度级别 |

**2.2 ModelInput**
模型输入结构。

```python
@dataclass
class ModelInput:
    prompt: str # 必填，提示词
    system_prompt: Optional[str] = None # 可选，系统提示词
    generation_config: Dict[str, Any] = field(default_factory=dict) # 可选，生成配置
```

**2.3 EvaluationResult**
评估结果容器。

```python
@dataclass
class EvaluationResult:
    data_id: str # 对应数据项ID
    evaluator_name: str # 评估器名称
    raw_output: Any # 模型原始输出
    metrics: Dict[str, Any] # 评估指标字典
    details: Dict[str, Any] = field(default_factory=dict) # 详细结果
```

---

#### 3. 基类与注册规范
框架采用基于接口的设计模式，所有组件必须继承相应的基类并注册到中心。

**3.1 BaseDataset (数据集基类)**
定义数据集加载的标准接口，**必须包含 `load` 和 `preprocess` 方法**。

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> List[DataItem]:
        """
        加载数据集并转换为标准格式

        Returns:
            List[DataItem]: 标准格式的数据项列表

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass

    @abstractmethod
    def preprocess(self, data_item: Dict[str, Any]) -> DataItem:
        """
        将原始数据预处理为标准 DataItem

        Args:
            data_item: 原始数据项字典

        Returns:
            DataItem: 标准格式数据项
        """
        pass
```

**3.2 BaseModel (模型基类)**
定义模型调用的标准接口。

```python
class BaseModel(ABC):
    @abstractmethod
    def generate(self, inputs: List[ModelInput]) -> List[str]:
        """接收提示词列表，返回生成文本列表"""
        pass
```

**3.3 BaseEvaluator (评估器基类)**
定义评估逻辑的标准接口。

```python
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, data_item: DataItem, model_output: str) -> EvaluationResult:
        """对比参考答案与模型输出，返回评估结果"""
        pass
```

**3.4 注册中心 (Registry)**
使用单例模式管理组件。

```python
class Registry:
    _store = {}
    
    @classmethod
    def register(cls, name: str, obj: Any):
        cls._store[name] = obj
        
    @classmethod
    def get(cls, name: str) -> Any:
        return cls._store.get(name)

# 具体的注册中心实例
DatasetRegistry = Registry()
ModelRegistry = Registry()
EvaluatorRegistry = Registry()
```

---

#### 4. 通用筛选与限制
为了支持灵活的数据测试，框架内置了通用的筛选器。

**4.1 DataFilter**
```python
from dataclasses import dataclass
from typing import List, Optional, Callable, Any

@dataclass
class DataFilter:
    categories_include: Optional[List[str]] = None # 白名单
    categories_exclude: Optional[List[str]] = None # 黑名单
    custom_filter: Optional[Callable[[Any], bool]] = None # 自定义函数

    def apply(self, data_items: List[Any]) -> List[Any]:
        result = data_items
        if self.categories_include:
            result = [item for item in result if any(cat in item.category for cat in self.categories_include)]
        if self.categories_exclude:
            result = [item for item in result if not any(cat in item.category for cat in self.categories_exclude)]
        if self.custom_filter:
            result = [item for item in result if self.custom_filter(item)]
        return result
```

---

#### 5. 配置体系
支持混合任务调度与加权聚合。

**5.1 DatasetConfig**
用于定义具体数据集的参数。

```python
@dataclass
class DatasetConfig:
    name: str                        # 数据集名称
    weight: float = 1.0              # 权重（仅在属于 Benchmark 时有效）
    limit: Optional[int] = None      # 数量限制
    filter: Optional[DataFilter] = None # 筛选规则
```

**5.2 BenchmarkConfig**
定义一个评测基准。

```python
@dataclass
class BenchmarkConfig:
    name: str                        # 基准名称
    datasets: List[DatasetConfig]    # 包含的子数据集列表
    aggregation_method: str = "weighted_average" # 聚合方式
```

**5.3 RunConfig**
支持传入混合任务列表。

```python
from typing import Union, List
from dataclasses import dataclass, field

@dataclass
class RunConfig:
    tasks: List[Union[DatasetConfig, BenchmarkConfig]] # 混合任务列表
    
    evaluator_configs: List[Dict[str, Any]]
    rounds: int = 1
    model_config: Dict[str, Any] = field(default_factory=dict)
    extra_args: Dict[str, Any] = field(default_factory=dict)
```

---

#### 6. 运行类与执行逻辑 (EvaluationRunner)

**6.1 核心运行类代码**

```python
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
        # 使用注册中心创建数据集实例
        dataset = DatasetRegistry.get(ds_config.name)() # 实例化
        raw_data = dataset.load() # 获取 DataItem 列表
        
        # 应用筛选
        if ds_config.filter:
            raw_data = ds_config.filter.apply(raw_data)
        # 应用限制
        if ds_config.limit is not None:
            raw_data = raw_data[:ds_config.limit]
            
        return raw_data

    def _run_single_dataset(self, ds_config: DatasetConfig) -> Dict[str, Any]:
        """
        执行单个数据集的评测
        """
        data_items = self._prepare_dataset(ds_config)
        if not data_items: return {"score": 0, "report": "No data"}

        # 构建并运行引擎
        engine, collector = self._setup_engine(data_items, ds_config.name)
        for _ in range(self.config.rounds): engine.run()
        
        # 分析结果
        results = collector.get_all_results()
        analyzer = DatasetAnalyzer(dataset=None, results=results)
        report = analyzer.generate_report()
        
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
            
            # 加权聚合
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
        # ... 构建 EvaluationEngine 的标准逻辑 ...
        pass
```

---

#### 7. 使用示例

```python
# 1. 定义任务
# 任务 A: 单个数据集
debug_task = DatasetConfig(name="gsm8k", limit=5)

# 任务 B: 基准
mmlu_stem = DatasetConfig(name="mmlu_stem", weight=0.6)
mmlu_humanities = DatasetConfig(name="mmlu_humanities", weight=0.4)
official_bench = BenchmarkConfig(
    name="Academic_Bench",
    datasets=[mmlu_stem, mmlu_humanities]
)

# 2. 配置运行
config = RunConfig(
    tasks=[debug_task, official_bench], # 混合传入
    evaluator_configs=[{"name": "ExactMatchEvaluator"}],
    rounds=1,
    model_config={"name": "qwen-max"}
)

# 3. 运行
runner = EvaluationRunner(config)
results = runner.run()
```