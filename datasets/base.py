from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.base import DataItem

class BaseDataset(ABC):
    def __init__(self, config):
        self.config = config
    
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
