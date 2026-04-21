from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional
import statistics


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

    @property
    def result_count(self) -> int:
        return len(self._results)

    def add_result(self, result: Dict[str, Any]) -> None:
        self._results.append(result)

    def add_results(self, results: List[Dict[str, Any]]) -> None:
        self._results.extend(results)

    def clear_results(self) -> None:
        self._results.clear()

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def filter_results(self, predicate: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        return [r for r in self._results if predicate(r)]

    def get_field_values(self, field: str) -> List[Any]:
        return [r.get(field) for r in self._results if field in r]

    def group_by(self, field: str) -> Dict[Any, List[Dict[str, Any]]]:
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for r in self._results:
            key = r.get(field)
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        return groups

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'sum': sum(values),
            'count': len(values)
        }

    def get_summary(self, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self._results:
            return {}

        if fields is None:
            fields = list(self._results[0].keys())

        summary = {}
        for field in fields:
            if all(isinstance(r.get(field), (int, float)) for r in self._results if field in r):
                values = [r[field] for r in self._results if field in r]
                summary[field] = self._compute_stats(values)

        return summary

    def get_field_summary(self, field: str) -> Dict[str, float]:
        values = [r.get(field) for r in self._results if field in r and isinstance(r.get(field), (int, float))]
        if not values:
            return {}
        return self._compute_stats(values)

    def iter_results(self) -> Iterator[Dict[str, Any]]:
        return iter(self._results)

    @abstractmethod
    def generate(self) -> Any:
        pass

    def save(self, path: str, **kwargs) -> None:
        raise NotImplementedError("Subclass must implement save method")


class JSONReportMixin:
    def _generate_base(self, include_summary: bool = True) -> Dict[str, Any]:
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self._config,
            'metadata': self._metadata,
            'result_count': len(self._results)
        }
        if include_summary:
            report['summary'] = self.get_summary()
        report['results'] = self._results
        return report


class CSVReportMixin:
    def _flatten_result(self, result: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        flat = {}
        for key, value in result.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_result(value, f"{new_key}."))
            elif isinstance(value, list):
                flat[new_key] = ','.join(str(v) for v in value)
            else:
                flat[new_key] = value
        return flat

    def _get_all_keys(self) -> List[str]:
        if not self._results:
            return []
        flat_result = self._flatten_result(self._results[0])
        return list(flat_result.keys())
