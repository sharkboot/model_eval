from dataclasses import dataclass
from typing import List, Optional, Callable, Any

@dataclass
class DataFilter:
    categories_include: Optional[List[str]] = None
    categories_exclude: Optional[List[str]] = None
    custom_filter: Optional[Callable[[Any], bool]] = None

    def apply(self, data_items: List[Any]) -> List[Any]:
        result = data_items
        if self.categories_include:
            result = [item for item in result if any(cat in item.category for cat in self.categories_include)]
        if self.categories_exclude:
            result = [item for item in result if not any(cat in item.category for cat in self.categories_exclude)]
        if self.custom_filter:
            result = [item for item in result if self.custom_filter(item)]
        return result
