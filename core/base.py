from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DataItem:
    id: str
    prompt: str
    reference: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: List[str] = field(default_factory=list)
    difficulty: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelInput:
    prompt: str
    system_prompt: Optional[str] = None
    generation_config: Dict[str, Any] = field(default_factory=dict)
    messages: Optional[List[Dict[str, str]]] = None  # 多轮对话历史，格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

@dataclass
class EvaluationResult:
    data_id: str
    evaluator_name: str
    raw_output: Any
    metrics: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    text: str