from .base import BaseReport, JSONReportMixin, CSVReportMixin
from core.logger import get_logger
from core.registry import Registry
import json
import csv
from typing import Any, Dict, List, Optional, Iterator

logger = get_logger()


@Registry.register("json", "report")
class JSONReport(BaseReport, JSONReportMixin):
    def generate(self) -> Dict[str, Any]:
        return self._generate_base()

    def save(self, path: str, **kwargs) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.generate(), f, ensure_ascii=False, indent=2)


@Registry.register("jsonl", "report")
class JSONLinesReport(BaseReport):
    def generate(self) -> List[Dict[str, Any]]:
        return self._results

    def save(self, path: str, **kwargs) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for result in self._results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def save_streaming(self, path: str) -> Iterator[None]:
        with open(path, 'w', encoding='utf-8') as f:
            for result in self._results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                yield


@Registry.register("table", "report")
class TableReport(BaseReport, CSVReportMixin):
    def __init__(self, config: Optional[Dict[str, Any]] = None, flatten: bool = True):
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

    def save_streaming(self, path: str) -> Iterator[None]:
        keys = self._get_all_keys() if self._flatten else (list(self._results[0].keys()) if self._results else [])
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for r in self._results:
                if self._flatten:
                    row = [self._flatten_result(r).get(k, '') for k in keys]
                else:
                    row = [r.get(k, '') for k in keys]
                writer.writerow(row)
                yield


class LogReport(BaseReport):
    def generate(self) -> List[str]:
        return [f"Result: {json.dumps(r, ensure_ascii=False)}" for r in self._results]

    def save(self, path: str, **kwargs) -> None:
        level = kwargs.get('level', 'INFO')
        log_method = getattr(logger, level.lower(), logger.info)
        for entry in self.generate():
            log_method(f"[{level}] {entry}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.generate()))


@Registry.register("markdown", "report")
class MarkdownReport(BaseReport, CSVReportMixin):
    def generate(self, fields: Optional[List[str]] = None) -> str:
        if not self._results:
            return ""

        summary = self.get_summary(fields)
        lines = ["# Evaluation Report\n"]

        if self._config:
            lines.append("## Configuration\n")
            for key, value in self._config.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if summary:
            lines.append("## Summary\n")
            lines.append("| Metric | Mean | Median | StdDev | Min | Max | Count |")
            lines.append("|--------|------|--------|--------|-----|-----|-------|")
            for metric, stats in summary.items():
                lines.append(f"| {metric} | {stats['mean']:.4f} | {stats['median']:.4f} | {stats['stdev']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |")
            lines.append("")

        keys = fields or self._get_all_keys()
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
            f.write(self.generate(kwargs.get('fields')))


@Registry.register("html", "report")
class HTMLReport(BaseReport, CSVReportMixin):
    def __init__(self, config: Optional[Dict[str, Any]] = None, title: str = "Evaluation Report"):
        super().__init__(config)
        self._title = title

    def generate(self, fields: Optional[List[str]] = None) -> str:
        if not self._results:
            return f"<html><head><title>{self._title}</title></head><body><h1>{self._title}</h1><p>No results.</p></body></html>"

        summary = self.get_summary(fields)
        keys = fields or self._get_all_keys()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self._title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        tr:nth-child(even) {{ background-color: #fafafa; }}
        .summary {{ background-color: #f0f8ff; }}
    </style>
</head>
<body>
    <h1>{self._title}</h1>
"""

        if self._config:
            html += "<h2>Configuration</h2><ul>"
            for key, value in self._config.items():
                html += f"<li><strong>{key}</strong>: {value}</li>"
            html += "</ul>"

        if summary:
            html += "<h2>Summary</h2><table><tr><th>Metric</th><th>Mean</th><th>Median</th><th>StdDev</th><th>Min</th><th>Max</th><th>Count</th></tr>"
            for metric, stats in summary.items():
                html += f"<tr class='summary'><td>{metric}</td><td>{stats['mean']:.4f}</td><td>{stats['median']:.4f}</td><td>{stats['stdev']:.4f}</td><td>{stats['min']:.4f}</td><td>{stats['max']:.4f}</td><td>{stats['count']}</td></tr>"
            html += "</table>"

        html += "<h2>Results</h2><table><tr>"
        for key in keys:
            html += f"<th>{key}</th>"
        html += "</tr>"

        for result in self._results:
            flat = self._flatten_result(result)
            html += "<tr>"
            for key in keys:
                html += f"<td>{flat.get(key, '')}</td>"
            html += "</tr>"

        html += "</table></body></html>"
        return html

    def save(self, path: str, **kwargs) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.generate(kwargs.get('fields')))


@Registry.register("chart", "report")
class ChartReport(BaseReport, CSVReportMixin):
    CHART_JS = """<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, title: str = "Evaluation Report"):
        super().__init__(config)
        self._title = title
        self._chart_type = 'bar'
        self._numeric_fields: List[str] = []

    def set_chart_type(self, chart_type: str) -> 'ChartReport':
        self._chart_type = chart_type
        return self

    def _prepare_chart_data(self) -> Dict[str, Any]:
        if not self._results:
            return {}

        self._numeric_fields = []
        for key in self._results[0].keys():
            if all(isinstance(r.get(key), (int, float)) for r in self._results if key in r):
                self._numeric_fields.append(key)

        labels = [str(r.get(self._numeric_fields[0], i)) if self._numeric_fields else str(i) for i, r in enumerate(self._results)]
        datasets = []
        colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']

        for idx, field in enumerate(self._numeric_fields):
            values = [r.get(field, 0) for r in self._results]
            datasets.append({
                'label': field,
                'data': values,
                'backgroundColor': colors[idx % len(colors)]
            })

        return {'labels': labels, 'datasets': datasets}

    def _generate_bar_chart(self, chart_data: Dict[str, Any]) -> str:
        canvas_id = "chart_" + str(hash(str(chart_data)) % 100000)
        chart_config = json.dumps({
            'type': self._chart_type,
            'data': chart_data,
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'position': 'top'},
                    'title': {'display': True, 'text': self._title}
                }
            }
        }, ensure_ascii=False)

        return f"""
<canvas id="{canvas_id}"></canvas>
<script>
var ctx = document.getElementById('{canvas_id}').getContext('2d');
new Chart(ctx, {chart_config});
</script>
"""

    def _generate_line_chart(self, chart_data: Dict[str, Any]) -> str:
        canvas_id = "chart_" + str(hash(str(chart_data)) % 100000)
        for ds in chart_data['datasets']:
            ds['fill'] = False
            ds['tension'] = 0.1

        chart_config = json.dumps({
            'type': 'line',
            'data': chart_data,
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'position': 'top'},
                    'title': {'display': True, 'text': self._title}
                }
            }
        }, ensure_ascii=False)

        return f"""
<canvas id="{canvas_id}"></canvas>
<script>
var ctx = document.getElementById('{canvas_id}').getContext('2d');
new Chart(ctx, {chart_config});
</script>
"""

    def _generate_pie_chart(self, chart_data: Dict[str, Any]) -> str:
        if len(self._numeric_fields) < 1:
            return "<p>No numeric fields available for pie chart.</p>"

        canvas_id = "chart_" + str(hash(str(chart_data)) % 100000)
        pie_data = {
            'labels': [str(r.get(self._numeric_fields[0], i)) for i, r in enumerate(self._results)],
            'datasets': [{
                'data': [r.get(self._numeric_fields[0], 0) for r in self._results],
                'backgroundColor': ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']
            }]
        }

        chart_config = json.dumps({
            'type': 'pie',
            'data': pie_data,
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'position': 'top'},
                    'title': {'display': True, 'text': self._title}
                }
            }
        }, ensure_ascii=False)

        return f"""
<canvas id="{canvas_id}"></canvas>
<script>
var ctx = document.getElementById('{canvas_id}').getContext('2d');
new Chart(ctx, {chart_config});
</script>
"""

    def generate(self, fields: Optional[List[str]] = None) -> str:
        if not self._results:
            return f"<html><head><title>{self._title}</title></head><body><h1>{self._title}</h1><p>No results.</p></body></html>"

        chart_data = self._prepare_chart_data()
        summary = self.get_summary(fields)

        chart_html = ""
        if chart_data.get('datasets'):
            if self._chart_type == 'bar':
                chart_html = self._generate_bar_chart(chart_data)
            elif self._chart_type == 'line':
                chart_html = self._generate_line_chart(chart_data)
            elif self._chart_type == 'pie':
                chart_html = self._generate_pie_chart(chart_data)

        keys = fields or self._get_all_keys()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self._title}</title>
    {self.CHART_JS}
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .chart-container {{ width: 80%; margin: 20px auto; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        tr:nth-child(even) {{ background-color: #fafafa; }}
        .summary {{ background-color: #f0f8ff; }}
    </style>
</head>
<body>
    <h1>{self._title}</h1>
"""

        if self._config:
            html += "<h2>Configuration</h2><ul>"
            for key, value in self._config.items():
                html += f"<li><strong>{key}</strong>: {value}</li>"
            html += "</ul>"

        if summary:
            html += "<h2>Summary</h2><table><tr><th>Metric</th><th>Mean</th><th>Median</th><th>StdDev</th><th>Min</th><th>Max</th><th>Count</th></tr>"
            for metric, stats in summary.items():
                html += f"<tr class='summary'><td>{metric}</td><td>{stats['mean']:.4f}</td><td>{stats['median']:.4f}</td><td>{stats['stdev']:.4f}</td><td>{stats['min']:.4f}</td><td>{stats['max']:.4f}</td><td>{stats['count']}</td></tr>"
            html += "</table>"

        if chart_html:
            html += f"<h2>Charts</h2><div class='chart-container'>{chart_html}</div>"

        html += "<h2>Results</h2><table><tr>"
        for key in keys:
            html += f"<th>{key}</th>"
        html += "</tr>"

        for result in self._results:
            flat = self._flatten_result(result)
            html += "<tr>"
            for key in keys:
                html += f"<td>{flat.get(key, '')}</td>"
            html += "</tr>"

        html += "</table></body></html>"
        return html

    def save(self, path: str, **kwargs) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.generate(kwargs.get('fields')))
