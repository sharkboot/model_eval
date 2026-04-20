from .base import BaseReport, JSONReportMixin, CSVReportMixin
import json
import csv
from typing import Any, Dict, List


class JSONReport(BaseReport, JSONReportMixin):
    def generate(self) -> Dict[str, Any]:
        return self._generate_base()

    def save(self, path: str, **kwargs) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.generate(), f, ensure_ascii=False, indent=2)


class TableReport(BaseReport, CSVReportMixin):
    def __init__(self, config: Dict[str, Any] = None, flatten: bool = True):
        super().__init__(config)
        self._flatten = flatten

    def generate(self) -> Dict[str, Any]:
        if not self._results:
            return {'headers': [], 'rows': []}

        if self._flatten:
            keys = self._get_all_keys()
            rows = [[self._flatten_result(r).get(k, '') for k in keys] for r in self._results]
        else:
            keys = list(self._results[0].keys()) if self._results else []
            rows = [[r.get(k, '') for k in keys] for r in self._results]

        return {'headers': keys, 'rows': rows}

    def save(self, path: str, **kwargs) -> None:
        report = self.generate()
        if not report['headers']:
            return

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(report['headers'])
            writer.writerows(report['rows'])


class LogReport(BaseReport):
    def generate(self) -> List[str]:
        return [f"Result: {json.dumps(r, ensure_ascii=False)}" for r in self._results]

    def save(self, path: str, **kwargs) -> None:
        level = kwargs.get('level', 'INFO')
        for entry in self.generate():
            print(f"[{level}] {entry}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.generate()))


class LinesReport(BaseReport):
    def generate(self) -> List[Dict[str, Any]]:
        return self._results

    def save(self, path: str, **kwargs) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for result in self._results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


class MarkdownReport(BaseReport, CSVReportMixin):
    def generate(self) -> str:
        if not self._results:
            return ""

        summary = self.get_summary()
        lines = ["# Evaluation Report\n"]

        if self._config:
            lines.append("## Configuration\n")
            for key, value in self._config.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if summary:
            lines.append("## Summary\n")
            lines.append("| Metric | Mean | Min | Max | Count |")
            lines.append("|--------|------|-----|-----|-------|")
            for metric, stats in summary.items():
                lines.append(f"| {metric} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |")
            lines.append("")

        keys = self._get_all_keys()
        lines.append("## Results\n")
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for result in self._results:
            flat = self._flatten_result(result)
            row = [str(flat.get(k, '')) for k in keys]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def save(self, path: str, **kwargs) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.generate())
