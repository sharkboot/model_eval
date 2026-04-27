"""Gradio UI components for evaluation monitoring dashboard."""
import threading
import time

import gradio as gr

from .events import get_state, put_event, start_consumer
from .state import MonitorState


# ============================================================================
# Chart Rendering
# ============================================================================

CHART_CDN = """<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>"""

METRIC_COLORS = {
    "accuracy": "#4CAF50",
    "precision": "#2196F3",
    "recall": "#FF9800",
    "f1": "#9C27B0",
}


def _render_metrics_cards(metrics: dict) -> str:
    """Render metric values as styled cards."""
    if not metrics:
        return '<div style="color:#888;font-size:14px;">No metrics yet</div>'

    cards = []
    for k, v in metrics.items():
        color = METRIC_COLORS.get(k, "#666")
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        cards.append(f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border: 1px solid {color}44;
            border-radius: 12px;
            padding: 16px 24px;
            display: inline-block;
            margin: 4px;
            min-width: 120px;
        ">
            <div style="color:#888;font-size:12px;text-transform:uppercase;">{k}</div>
            <div style="color:{color};font-size:24px;font-weight:bold;">{val_str}</div>
        </div>
        """)
    return "".join(cards)


def _render_chart(history: dict, height: int = 280) -> str:
    """Render Chart.js line chart with all metrics."""
    if not history:
        return '<div style="color:#888;padding:20px;text-align:center;">No data yet</div>'

    datasets = []
    for i, (metric, data) in enumerate(history.items()):
        color = METRIC_COLORS.get(metric, f"hsl({i * 60}, 70%, 50%)")
        datasets.append(f"""{{
            label: "{metric}",
            data: {data},
            borderColor: "{color}",
            backgroundColor: "{color}22",
            tension: 0.3,
            fill: true,
            pointRadius: 2,
            pointHoverRadius: 5
        }}""")

    max_len = max((len(v) for v in history.values()), default=0)
    labels = list(range(max_len))

    return f"""
    {CHART_CDN}
    <div style="position:relative;height:{height}px;">
        <canvas id="dash_chart"></canvas>
    </div>
    <script>
    (function() {{
        const ctx = document.getElementById("dash_chart");
        if (window._chart) window._chart.destroy();
        window._chart = new Chart(ctx, {{
            type: "line",
            data: {{
                labels: {labels},
                datasets: [{",".join(datasets)}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: "index", intersect: false }},
                plugins: {{
                    legend: {{ position: "top" }},
                    tooltip: {{ enabled: true }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, grid: {{ color: "#eee" }} }},
                    x: {{ grid: {{ display: false }} }}
                }},
                animation: {{ duration: 200 }}
            }}
        }});
    }})();
    </script>
    """


# ============================================================================
# Log Streamer (maintains cursor position for incremental log fetching)
# ============================================================================

class LogStreamer:
    """Streaming state for incremental log fetching."""

    def __init__(self):
        self.last_idx = 0
        self.lock = threading.Lock()

    def fetch_new(self) -> str:
        """Return only NEW logs since last fetch."""
        with self.lock:
            state = get_state()
            _, _, logs, _, _ = state.snapshot()
            if self.last_idx >= len(logs):
                return ""
            new_logs = logs[self.last_idx:]
            self.last_idx = len(logs)
            return "\n".join(new_logs)

    def reset(self):
        """Reset streamer state."""
        with self.lock:
            self.last_idx = 0

    def generator(self):
        """Generator for streaming Textbox - yields new logs continuously."""
        while True:
            time.sleep(0.3)
            new_logs = self.fetch_new()
            if new_logs:
                yield new_logs


# ============================================================================
# Data Fetchers
# ============================================================================

def _fetch_overview():
    """Fetch overview data - metrics and chart."""
    state = get_state()
    metrics, history, _, _, current = state.snapshot()

    cards = _render_metrics_cards(metrics)
    chart = _render_chart(history)
    stage = current.get("stage", "idle") if current else "idle"
    status = f'<span style="color:#4CAF50;">● {stage}</span>' if stage != "idle" else '<span style="color:#888;">○ idle</span>'

    return cards, chart, status


def _fetch_results(only_fail: bool, threshold: float):
    """Fetch filtered results table."""
    state = get_state()
    _, _, _, results, _ = state.snapshot()

    rows = []
    for r in results:
        score = r.get("metrics", {}).get("accuracy", 0)
        if only_fail and score >= threshold:
            continue
        rows.append([r.get("id", ""), round(score, 4)])

    return rows, len(results)


def _fetch_detail(idx: int):
    """Fetch single result detail."""
    state = get_state()
    _, _, _, results, _ = state.snapshot()
    if 0 <= idx < len(results):
        return results[idx]
    return {}


# ============================================================================
# Demo Evaluator
# ============================================================================

def _demo_evaluator(num_samples: int = 50, delay: float = 0.3):
    """Generate fake evaluation events for demo."""
    import random

    for i in range(num_samples):
        time.sleep(delay)

        stage = "eval" if i % 10 == 0 else "processing"
        put_event({"type": "stage", "data": {"stage": stage, "id": f"sample_{i}"}})

        acc = random.random()
        loss = random.uniform(0.1, 2.0)
        put_event({"type": "metric", "data": {"accuracy": acc, "loss": loss}})

        put_event({
            "type": "result",
            "data": {
                "id": i,
                "prediction": f"pred_{i}",
                "gt": f"gt_{i}",
                "metrics": {"accuracy": acc, "loss": loss},
            },
        })

        put_event({"type": "log", "message": f"[{stage}] sample {i}: acc={acc:.4f}"})


# ============================================================================
# UI Builder
# ============================================================================

def build_app(
    title: str = "Evaluation Monitor",
    evaluator_fn=None,
    num_samples: int = 50,
    poll_interval: float = 0.5,
) -> gr.Blocks:
    """Build the Gradio monitoring dashboard.

    Args:
        title: Dashboard title.
        evaluator_fn: Custom evaluator function. If None, uses demo.
        num_samples: Number of samples for demo evaluator.
        poll_interval: Seconds between auto-refresh for metrics/results.

    Returns:
        Gradio Blocks app.
    """
    start_consumer()

    log_streamer = LogStreamer()

    with gr.Blocks(title=title, theme=gr.themes.Soft()) as app:

        gr.Markdown(f"# {title}")

        # ---- Status Row ----
        with gr.Row():
            status_html = gr.HTML('<span style="color:#888;">○ idle</span>', elem_id="status")
            progress_html = gr.HTML('<span style="color:#888;">0 / 0</span>', elem_id="progress")

        with gr.Row():
            start_btn = gr.Button("Start Evaluation", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop", visible=False)

        gr.Divider()

        # ---- Tabs ----
        with gr.Tabs():

            # ===== Overview Tab =====
            with gr.TabItem("Overview"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        gr.Markdown("### Current Metrics")
                        metrics_html = gr.HTML()
                    with gr.Column(scale=2):
                        gr.Markdown("### History Chart")
                        chart_html = gr.HTML()

                # Metrics + Chart update together via polling
                gr.Timer(value=poll_interval).tick(
                    fn=_fetch_overview,
                    outputs=[metrics_html, chart_html, status_html],
                )

            # ===== Results Tab =====
            with gr.TabItem("Results"):
                with gr.Row():
                    only_fail = gr.Checkbox(label="Show Failed Only", value=False)
                    threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    count_html = gr.HTML('<span style="color:#888;">0 results</span>')

                results_table = gr.Dataframe(
                    headers=["id", "accuracy"],
                    interactive=False,
                    wrap=True,
                    label="Samples",
                    height=300,
                )

                detail_json = gr.JSON(label="Selected Detail", height=200)

                gr.Timer(value=poll_interval).tick(
                    fn=_fetch_results,
                    inputs=[only_fail, threshold],
                    outputs=[results_table, count_html],
                )

                results_table.select(
                    fn=_fetch_detail,
                    inputs=None,
                    outputs=detail_json,
                )

            # ===== Logs Tab =====
            with gr.TabItem("Logs"):
                log_text = gr.Textbox(
                    lines=20,
                    label="Event Log",
                    autoscroll=True,
                    interactive=False,
                    show_label=True,
                )

                # Logs use streaming generator - appends new lines incrementally
                log_text.stream(
                    fn=log_streamer.generator,
                    inputs=None,
                    outputs=[log_text],
                    queue=True,
                )

        # ---- Button Handlers ----
        def on_start():
            log_streamer.reset()
            return (
                gr.Button(variant="secondary", visible=False),
                gr.Button(variant="stop", visible=True),
                '<span style="color:#FF9800;">● RUNNING</span>',
            )

        def on_stop():
            return (
                gr.Button(variant="primary", visible=True),
                gr.Button(variant="stop", visible=False),
                '<span style="color:#888;">○ idle</span>',
            )

        start_btn.click(fn=on_start, outputs=[start_btn, stop_btn, status_html])
        stop_btn.click(fn=on_stop, outputs=[start_btn, stop_btn, status_html])

        def run_eval():
            log_streamer.reset()
            fn = evaluator_fn or (lambda: _demo_evaluator(num_samples))
            threading.Thread(target=fn, daemon=True).start()

        start_btn.click(fn=run_eval)

        # Initial load
        app.load(fn=_fetch_overview, outputs=[metrics_html, chart_html, status_html])

    return app


def launch(
    title: str = "Evaluation Monitor",
    evaluator_fn=None,
    num_samples: int = 50,
    poll_interval: float = 0.5,
    **kwargs,
) -> None:
    """Build and launch the monitoring dashboard."""
    app = build_app(title, evaluator_fn, num_samples, poll_interval)
    app.launch(**kwargs)
