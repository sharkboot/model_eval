from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.base import DataItem


class BaseDataset(ABC):
    def __init__(self, config):
        self.config = config
        self.limits = config.get("limits", None)

        # 自动用类名作为 dataset_name（更优雅）
        self.dataset_name = self.__class__.__name__

    def load(self) -> List[DataItem]:
        raw_data = self.load_raw_data()

        # ✅ limits 统一处理
        if self.limits is not None:
            raw_data = raw_data[: self.limits]

        return [self.preprocess(item) for item in raw_data]

    @abstractmethod
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        子类只负责加载原始数据
        """
        pass

    @abstractmethod
    def preprocess(self, data_item: Dict[str, Any]) -> DataItem:
        pass

    # ✅ 统一 ID 生成逻辑
    def build_id(self, raw_id: Any) -> str:
        return f"{self.dataset_name}_{hash(str(raw_id))}"