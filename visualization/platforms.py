import os
import json
import threading
import time
from datetime import datetime
from collections import defaultdict

from .base import BaseVisualizer
from core.logger import get_logger

logger = get_logger()

METRIC_COLORS = [
    '#4CAF50', '#2196F3', '#FF9800', '#E91E63',
    '#9C27B0', '#00BCD4', '#FF5722', '#607D8B'
]


class LiveMonitor:
    def __init__(self, result_file: str, summary_file: str, total: int = None):
        self.result_file = result_file
        self.summary_file = summary_file
        self.total = total
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._metric_history = defaultdict(list)
        self._current_stage = "idle"
        self._current_item_id = None
        self._current_item_detail = None
        self._logs = ["System initialized"]
        self._max_logs = 100

    def stop(self):
        self._stop.set()

    def is_stopped(self):
        return self._stop.is_set()

    def set_stage(self, stage: str, item_id: str = None, detail: str = None):
        with self._lock:
            self._current_stage = stage
            self._current_item_id = item_id
            self._current_item_detail = detail
            if stage != "idle":
                self._logs.append(f"[{stage.upper()}] {item_id or ''} - {detail or ''}")
                if len(self._logs) > self._max_logs:
                    self._logs = self._logs[-self._max_logs:]

    def add_log(self, message: str, level: str = "info"):
        with self._lock:
            self._logs.append(f"[{level.upper()}] {message}")
            if len(self._logs) > self._max_logs:
                self._logs = self._logs[-self._max_logs:]

    def read_results(self):
        results = []
        if os.path.exists(self.result_file):
            with open(self.result_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return results

    def get_progress(self):
        results = self.read_results()
        done = len(results)
        total = self.total or done

        metrics = {}
        if os.path.exists(self.summary_file):
            try:
                with open(self.summary_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                pass

        with self._lock:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._metric_history[key].append(value)
            current_stage = self._current_stage
            current_item_id = self._current_item_id
            current_item_detail = self._current_item_detail
            logs = list(self._logs)

        return {
            "done": done,
            "total": total,
            "progress": done / total if total > 0 else 0,
            "metrics": metrics,
            "metric_history": dict(self._metric_history),
            "results": results,
            "current_stage": current_stage,
            "current_item_id": current_item_id,
            "current_item_detail": current_item_detail,
            "logs": logs
        }


class GradioVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
        self.interface = None
        self.monitor = None
        self.monitor_thread = None
        self._start_time = None

    def setup(self):
        try:
            import gradio as gr
            self.interface = self._build_interface()
            return True
        except ImportError:
            return False

    def _build_interface(self):
        import gradio as gr

        self._css = """
        body { background: #0f0f1a !important; font-family: 'Segoe UI', sans-serif; }
        .gradio-container { background: #0f0f1a !important; }

        .header { text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px; }
        .header h1 { color: #ccd6f6; margin: 0; font-size: 28px; }
        .header p { color: #8892b0; margin: 8px 0 0; font-size: 14px; }

        .tab-content { padding: 10px 0; }

        /* Status Cards */
        .status-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
        .status-card { background: linear-gradient(145deg, #1a1a2e, #16213e); border-radius: 12px; padding: 20px; text-align: center; }
        .status-label { font-size: 11px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
        .status-value { font-size: 28px; font-weight: 700; margin-top: 8px; }
        .status-value.idle { color: #8892b0; }
        .status-value.running { color: #4CAF50; }
        .status-value.done { color: #2196F3; }
        .status-value.time { color: #64ffda; font-family: 'Consolas', monospace; }

        /* Pipeline */
        .pipeline { display: flex; align-items: center; justify-content: center; gap: 8px; padding: 24px; background: linear-gradient(145deg, #1a1a2e, #16213e); border-radius: 16px; margin: 20px 0; }
        .pipeline-stage { display: flex; flex-direction: column; align-items: center; padding: 12px 20px; border-radius: 12px; background: rgba(255,255,255,0.03); border: 2px solid transparent; min-width: 100px; transition: all 0.3s; }
        .pipeline-stage.active { background: rgba(76,175,80,0.15); border-color: #4CAF50; box-shadow: 0 0 25px rgba(76,175,80,0.4); transform: scale(1.05); }
        .pipeline-stage.done { background: rgba(33,150,243,0.1); border-color: #2196F3; }
        .pipeline-icon { font-size: 32px; margin-bottom: 6px; }
        .pipeline-name { font-size: 12px; color: #ccd6f6; font-weight: 600; }
        .pipeline-info { font-size: 10px; color: #8892b0; margin-top: 4px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .pipeline-arrow { font-size: 24px; color: #4CAF50; }

        /* Metric Cards */
        .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
        .metric-card { background: linear-gradient(145deg, #1a1a2e, #16213e); border-radius: 12px; padding: 20px; text-align: center; }
        .metric-value { font-size: 32px; font-weight: 700; background: linear-gradient(135deg, #4CAF50, #2196F3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .metric-label { font-size: 11px; color: #8892b0; text-transform: uppercase; margin-top: 8px; }

        /* Two Column Layout */
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: linear-gradient(145deg, #1a1a2e, #16213e); border-radius: 12px; padding: 20px; }
        .panel-title { color: #ccd6f6; font-size: 14px; font-weight: 600; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); }

        /* Current Item */
        .current-item { display: flex; align-items: center; gap: 16px; padding: 16px; background: #0d0d1a; border-radius: 10px; }
        .current-icon { font-size: 36px; }
        .current-info { flex: 1; }
        .current-id { color: #ccd6f6; font-size: 14px; font-weight: 600; }
        .current-detail { color: #8892b0; font-size: 12px; margin-top: 4px; }

        /* Log Box */
        .log-box { background: #0d0d1a; border-radius: 10px; padding: 16px; max-height: 300px; overflow-y: auto; font-family: 'Consolas', monospace; font-size: 12px; }
        .log-entry { padding: 6px 10px; border-left: 3px solid; margin-bottom: 4px; border-radius: 0 4px 4px 0; }
        .log-entry.info { color: #64ffda; border-color: #64ffda; background: rgba(100,255,218,0.05); }
        .log-entry.success { color: #4CAF50; border-color: #4CAF50; background: rgba(76,175,80,0.05); }
        .log-entry.warning { color: #FF9800; border-color: #FF9800; background: rgba(255,152,0,0.05); }
        .log-entry.error { color: #f44336; border-color: #f44336; background: rgba(244,67,54,0.05); }

        /* Chart */
        .chart-container { background: linear-gradient(145deg, #1a1a2e, #16213e); border-radius: 12px; padding: 20px; }
        .chart-wrapper { max-height: 350px; }

        /* Results Table */
        .results-table { background: #0d0d1a; border-radius: 10px; overflow: hidden; }
        .results-table table { width: 100%; border-collapse: collapse; }
        .results-table th { background: #16213e; color: #8892b0; padding: 12px 16px; text-align: left; font-size: 11px; text-transform: uppercase; }
        .results-table td { padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #ccd6f6; font-size: 13px; }
        .results-table tr:hover { background: rgba(255,255,255,0.02); }
        .result-ok { color: #4CAF50; font-weight: 600; }
        .result-fail { color: #f44336; font-weight: 600; }

        /* Tabs styling */
        .tabs .tab-nav { background: #1a1a2e; border-radius: 10px; padding: 4px; margin-bottom: 20px; }
        """

        with gr.Blocks(title="LLM Evaluation Monitor", css=self._css) as demo:
            gr.HTML("""
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <div class="header">
                <h1>🤖 LLM Evaluation Monitor</h1>
                <p>Real-time evaluation tracking and visualization</p>
            </div>
            """)

            with gr.Tabs():
                # Tab 1: Overview
                with gr.TabItem("📊 总览"):
                    with gr.Column():
                        # Status Cards
                        gr.HTML('<div class="status-grid" id="status-grid"></div>')
                        # Pipeline
                        gr.HTML('<div class="pipeline" id="pipeline"></div>')
                        # Metric Cards
                        gr.HTML('<div class="metric-grid" id="metric-grid"></div>')

                # Tab 2: Details
                with gr.TabItem("📋 详情"):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.HTML('<div class="panel"><div class="panel-title">当前处理项</div><div id="current-item"></div></div>')
                            with gr.Column():
                                gr.HTML('<div class="panel"><div class="panel-title">处理日志</div><div class="log-box" id="log-box"></div></div>')

                # Tab 3: Charts
                with gr.TabItem("📈 图表"):
                    with gr.Column():
                        gr.HTML('<div class="chart-container"><div class="chart-wrapper"><canvas id="mainChart"></canvas></div></div>')

                # Tab 4: Results
                with gr.TabItem("📄 结果"):
                    with gr.Column():
                        gr.HTML('<div class="results-table"><table id="results-table"><thead><tr><th>ID</th><th>Prediction</th><th>Metrics</th><th>Status</th></tr></thead><tbody id="results-body"></tbody></table></div>')

            # 刷新按钮
            refresh_btn = gr.Button("🔄 刷新", variant="primary", size="lg")

            # 更新函数
            def update_all():
                if not self.monitor:
                    return self._get_empty_state()

                if self._start_time is None:
                    self._start_time = time.time()

                info = self.monitor.get_progress()
                return self._build_display(info)

            def _get_empty_state(self):
                status_html = '''
                <div class="status-grid">
                    <div class="status-card"><div class="status-label">Status</div><div class="status-value idle">Idle</div></div>
                    <div class="status-card"><div class="status-label">Progress</div><div class="status-value">0/0</div></div>
                    <div class="status-card"><div class="status-label">Elapsed</div><div class="status-value time">00:00</div></div>
                    <div class="status-card"><div class="status-label">Accuracy</div><div class="status-value">--</div></div>
                </div>'''
                pipeline_html = self._build_pipeline_html("idle", "", "")
                metrics_html = '<div class="metric-grid"><div class="metric-card"><div class="metric-value">--</div><div class="metric-label">Accuracy</div></div><div class="metric-card"><div class="metric-value">--</div><div class="metric-label">Loss</div></div><div class="metric-card"><div class="metric-value">--</div><div class="metric-label">F1</div></div><div class="metric-card"><div class="metric-value">--</div><div class="metric-label">Custom</div></div></div>'
                current_item = '<div class="current-item"><div class="current-icon">⏸️</div><div class="current-info"><div class="current-id">Idle</div><div class="current-detail">等待开始...</div></div></div>'
                log_html = '<div class="log-entry info">System initialized, waiting for evaluation...</div>'
                chart_html = ''
                results_html = '<tr><td colspan="4" style="text-align:center;color:#8892b0;">No results yet</td></tr>'
                return [status_html, pipeline_html, metrics_html, current_item, log_html, chart_html, results_html]

            # 刷新事件
            refresh_btn.click(fn=update_all, outputs=[
                gr.HTML(), gr.HTML(), gr.HTML(), gr.HTML(), gr.HTML(), gr.HTML(), gr.HTML()
            ])

            # 定时器自动刷新
            timer = gr.Timer(value=1.5, active=True)
            timer.tick(fn=update_all, outputs=[
                gr.HTML(), gr.HTML(), gr.HTML(), gr.HTML(), gr.HTML(), gr.HTML(), gr.HTML()
            ])

        return demo

    def _build_display(self, info):
        done = info["done"]
        total = info["total"]
        progress = info["progress"]
        metrics = info["metrics"]
        history = info["metric_history"]
        results = info["results"]
        current_stage = info.get("current_stage", "idle")
        current_item_id = info.get("current_item_id", "")
        current_item_detail = info.get("current_item_detail", "")
        logs = info.get("logs", [])

        elapsed = int(time.time() - self._start_time) if self._start_time else 0
        elapsed_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

        # Status
        if done == 0:
            status_cls = "idle"
            status_text = "Idle"
        elif done >= total:
            status_cls = "done"
            status_text = "Done"
        else:
            status_cls = "running"
            status_text = "Running"

        accuracy = metrics.get("accuracy", metrics.get("acc", None))
        accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else "--"
        loss = metrics.get("loss", None)
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "--"
        f1 = metrics.get("f1", metrics.get("f1_score", None))
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "--"
        custom_keys = [k for k in metrics.keys() if k not in ("accuracy", "acc", "loss", "f1", "f1_score")]
        custom_val = metrics.get(custom_keys[0]) if custom_keys else None
        custom_str = f"{custom_val:.4f}" if isinstance(custom_val, (int, float)) else "--"

        status_html = f'''
        <div class="status-grid">
            <div class="status-card"><div class="status-label">Status</div><div class="status-value {status_cls}">{status_text}</div></div>
            <div class="status-card"><div class="status-label">Progress</div><div class="status-value">{done}/{total}</div></div>
            <div class="status-card"><div class="status-label">Elapsed</div><div class="status-value time">{elapsed_str}</div></div>
            <div class="status-card"><div class="status-label">Accuracy</div><div class="status-value">{accuracy_str}</div></div>
        </div>'''

        pipeline_html = self._build_pipeline_html(current_stage, current_item_id, current_item_detail)

        metrics_html = f'''
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-value">{accuracy_str}</div><div class="metric-label">Accuracy</div></div>
            <div class="metric-card"><div class="metric-value">{loss_str}</div><div class="metric-label">Loss</div></div>
            <div class="metric-card"><div class="metric-value">{f1_str}</div><div class="metric-label">F1 Score</div></div>
            <div class="metric-card"><div class="metric-value">{custom_str}</div><div class="metric-label">{custom_keys[0] if custom_keys else "Custom"}</div></div>
        </div>'''

        if current_stage != "idle":
            current_item = f'''
            <div class="current-item">
                <div class="current-icon">⏳</div>
                <div class="current-info">
                    <div class="current-id">{current_item_id or "Processing..."}</div>
                    <div class="current-detail">{current_item_detail or ""}</div>
                </div>
            </div>'''
        else:
            current_item = '''
            <div class="current-item">
                <div class="current-icon">⏸️</div>
                <div class="current-info">
                    <div class="current-id">Idle</div>
                    <div class="current-detail">Waiting to start...</div>
                </div>
            </div>'''

        log_html = ""
        for log in logs[-30:]:
            if "ERROR" in log:
                cls = "error"
            elif "WARNING" in log:
                cls = "warning"
            elif "SUCCESS" in log or "完成" in log:
                cls = "success"
            else:
                cls = "info"
            log_html += f'<div class="log-entry {cls}">{log}</div>'
        if not log_html:
            log_html = '<div class="log-entry info">No logs yet</div>'

        # Chart
        chart_config = {
            "type": "line",
            "data": {
                "labels": list(range(1, len(history.get("accuracy", [0])) + 1)),
                "datasets": []
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"legend": {"labels": {"color": "#ccd6f6"}}},
                "scales": {
                    "x": {"grid": {"color": "rgba(255,255,255,0.05)"}, "ticks": {"color": "#8892b0"}},
                    "y": {"grid": {"color": "rgba(255,255,255,0.05)"}, "ticks": {"color": "#8892b0"}}
                },
                "animation": {"duration": 300}
            }
        }
        for j, (key, values) in enumerate(history.items()):
            color = METRIC_COLORS[j % len(METRIC_COLORS)]
            chart_config["data"]["datasets"].append({
                "label": key,
                "data": values,
                "borderColor": color,
                "backgroundColor": "rgba(255,255,255,0.05)",
                "tension": 0.4,
                "fill": True
            })

        chart_html = f'''
        <div class="chart-container"><div class="chart-wrapper"><canvas id="mainChart"></canvas></div></div>
        <script>
        if (window.mainChart) {{ window.mainChart.destroy(); }}
        var ctx = document.getElementById('mainChart');
        window.mainChart = new Chart(ctx, {json.dumps(chart_config)});
        </script>'''

        # Results table (show last 50)
        rows_html = ""
        for r in results[-50:]:
            pred = str(r.get("prediction", ""))[:60]
            if len(str(r.get("prediction", ""))) > 60:
                pred += "..."
            score = r.get("metrics", {})
            score_str = json.dumps(score)[:30]
            is_correct = r.get("metrics", {}).get("accuracy", 0) >= 0.5 if isinstance(score, dict) else False
            cls = "result-ok" if is_correct else "result-fail"
            icon = "✓" if is_correct else "✗"
            rows_html += f'<tr><td>{r.get("id", "")}</td><td>{pred}</td><td>{score_str}</td><td class="{cls}">{icon}</td></tr>'

        if not rows_html:
            rows_html = '<tr><td colspan="4" style="text-align:center;color:#8892b0;">No results yet</td></tr>'

        return [status_html, pipeline_html, metrics_html, current_item, log_html, chart_html, rows_html]

    def _build_pipeline_html(self, current_stage, current_item_id, current_item_detail):
        stages = ["dataset", "prompt", "model", "eval", "result"]
        stage_names = ["Dataset", "Prompt", "Model", "Eval", "Result"]
        stage_icons = ["📂", "📝", "🤖", "✅", "📊"]
        stage_map = {s: i for i, s in enumerate(stages)}
        current_idx = stage_map.get(current_stage, -1)

        html = '<div class="pipeline">'
        for i, (s, name, icon) in enumerate(zip(stages, stage_names, stage_icons)):
            if i == current_idx:
                cls = "active"
                info_text = current_item_id[:20] + "..." if current_item_id and len(str(current_item_id)) > 20 else current_item_id
            elif i < current_idx:
                cls = "done"
                info_text = "完成"
            else:
                cls = ""
                info_text = "等待"

            html += f'<div class="pipeline-stage {cls}"><div class="pipeline-icon">{icon}</div><div class="pipeline-name">{name}</div><div class="pipeline-info">{info_text}</div></div>'
            if i < 4:
                html += '<div class="pipeline-arrow">→</div>'
        html += '</div>'
        return html

    def visualize(self, data=None):
        import sys
        if not self.interface:
            print("Setting up Gradio interface...", flush=True)
            success = self.setup()
            if not success:
                logger.warning("Gradio is not installed. Visualization will be skipped.")
                return
        print(f"\n🚀 Gradio: http://127.0.0.1:{self.config.get('port', 7860)}\n", flush=True)
        sys.stdout.flush()
        self.interface.launch(
            share=self.config.get('share', False),
            server_name="127.0.0.1",
            server_port=self.config.get('port', 7860)
        )

    def start_monitoring(self, result_file: str, summary_file: str, total: int = None):
        self.monitor = LiveMonitor(result_file, summary_file, total)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        if self.monitor:
            self.monitor.stop()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)

    def _monitor_loop(self):
        while self.monitor and not self.monitor.is_stopped():
            time.sleep(2)

    def save(self):
        self.stop_monitoring()


class WandbVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
        self.run = None

    def setup(self):
        try:
            import wandb
            self.run = wandb.init(project=self.config.get('project', 'model_eval'), name=self.config.get('name', 'eval_run'))
            return True
        except ImportError:
            raise ImportError("Please install wandb: pip install wandb")

    def visualize(self, data):
        if not self.run:
            self.setup()
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    self.run.log(value, step=0)
                else:
                    self.run.log({key: value}, step=0)

    def save(self):
        if self.run:
            self.run.finish()


class SwanLabVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        try:
            import swanlab
            swanlab.init(project=self.config.get('project', 'model_eval'), experiment_name=self.config.get('name', 'eval_run'))
            return True
        except ImportError:
            raise ImportError("Please install swanlab: pip install swanlab")

    def visualize(self, data):
        import swanlab
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        swanlab.log({f"{key}_{sub_key}": sub_value})
                else:
                    swanlab.log({key: value})

    def save(self):
        pass


class ClearMLVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
        self.task = None

    def setup(self):
        try:
            from clearml import Task
            self.task = Task.init(project_name=self.config.get('project', 'model_eval'), task_name=self.config.get('name', 'eval_run'))
            return True
        except ImportError:
            raise ImportError("Please install clearml: pip install clearml")

    def visualize(self, data):
        if not self.task:
            self.setup()
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        self.task.get_logger().report_scalar(title=key, series=sub_key, value=sub_value, iteration=0)
                else:
                    self.task.get_logger().report_scalar(title="Metrics", series=key, value=value, iteration=0)

    def save(self):
        if self.task:
            self.task.close()
