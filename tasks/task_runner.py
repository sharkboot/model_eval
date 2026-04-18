
from abc import ABC, abstractmethod
from typing import Dict


class BaseTaskRunner(ABC):

    @abstractmethod
    def run(self) -> Dict[str, float]:
        """
        Returns:
            Dict[str, float]: metric_name -> value
        """
        pass