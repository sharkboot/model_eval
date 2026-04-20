from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class BaseReport(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._results: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def results(self) -> List[Dict[str, Any]]:
        return self._results

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def add_result(self, result: Dict[str, Any]) -> None:
        self._results.append(result)

    def add_results(self, results: List[Dict[str, Any]]) -> None:
        self._results.extend(results)

    def clear_results(self) -> None:
        self._results.clear()

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_summary(self) -> Dict[str, Any]:
        if not self._results:
            return {}

        numeric_fields = {}
        for key in self._results[0].keys():
            if all(isinstance(r.get(key), (int, float)) for r in self._results if key in r):
                values = [r[key] for r in self._results if key in r]
                numeric_fields[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        return numeric_fields

    @abstractmethod
    def generate(self) -> Any:
        pass

    def save(self, path: str, **kwargs) -> None:
        raise NotImplementedError("Subclass must implement save method")


class JSONReportMixin:
    def _generate_base(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.now().isoformat(),
            'config': self._config,
            'metadata': self._metadata,
            'summary': self.get_summary(),
            'results': self._results
        }


class CSVReportMixin:
    def _flatten_result(self, result: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        flat = {}
        for key, value in result.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_result(value, f"{new_key}."))
            else:
                flat[new_key] = value
        return flat

    def _get_all_keys(self) -> List[str]:
        if not self._results:
            return []
        flat_result = self._flatten_result(self._results[0])
        return list(flat_result.keys())
