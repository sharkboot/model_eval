from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict, Any
from utils.data_filter import DataFilter

@dataclass
class DatasetConfig:
    name: str
    weight: float = 1.0
    limit: Optional[int] = None
    filter: Optional[DataFilter] = None

@dataclass
class BenchmarkConfig:
    name: str
    datasets: List[DatasetConfig]
    aggregation_method: str = "weighted_average"

@dataclass
class RunConfig:
    tasks: List[Union[DatasetConfig, BenchmarkConfig]]
    evaluator_configs: List[Dict[str, Any]]
    rounds: int = 1
    model_config: Dict[str, Any] = field(default_factory=dict)
    extra_args: Dict[str, Any] = field(default_factory=dict)
