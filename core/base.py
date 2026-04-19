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
    type: str  # "text" | "chat"

    # text 模式
    prompt: Optional[str] = None

    # chat 模式
    messages: Optional[List[Dict[str, str]]] = None

    # 通用配置
    system_prompt: Optional[str] = None


@dataclass
class EvaluationResult:
    data_id: str
    evaluator_name: str
    raw_output: Any
    metrics: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    type: str  # "text" | "chat"

    text: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None

    raw: Any = None
    usage: Dict[str, Any] = field(default_factory=dict)

    def get_text(self) -> str:
        if self.type == "text":
            return self.text or ""
        elif self.type == "chat" and self.messages:
            return self.messages[-1].get("content", "")
        return ""

    def get_messages(self) -> List[Dict[str, str]]:
        if self.type == "chat" and self.messages:
            return self.messages
        elif self.type == "text" and self.text:
            return [{"role": "assistant", "content": self.text}]
        return []
