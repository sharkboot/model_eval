import os
import json
import threading
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tasks.task_runner import BaseTaskRunner
from core.registry import Registry
from core.data_filter import DataFilter
from core.logger import get_logger

logger = get_logger()


def load_results_from_jsonl(path: str) -> list:
    """Load results from jsonl file."""
    results = []
    if not path or not os.path.exists(path):
        return results
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def build_run_dir(base_path: str, run_name: str = None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if run_name:
        run_dir = os.path.join(base_path, run_name)
    else:
        run_dir = os.path.join(base_path, f"run_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def safe_serialize(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


class StandardTaskRunner(BaseTaskRunner):

    def __init__(self, config):
        self.config = config
        self.visualizer = None

        # ========== 组件初始化 ==========
        self.dataset = Registry.create(
            config["dataset"]["name"],
            "dataset",
            **config["dataset"].get("params", {})
        )

        self.model = Registry.create(
            config["model"]["name"],
            "model",
            **config["model"].get("params", {})
        )

        self.evaluators = [
            Registry.create(e["name"], "evaluator", **e.get("params", {}))
            for e in config["evaluators"]
        ]

        self.prompt_builder = Registry.create(
            config["prompt_builder"]["name"],
            "prompt_builder",
            **config["prompt_builder"].get("params", {})
        )

        # ========== 并发 ==========
        self.num_workers = config.get("num_workers", 1)

        # ========== 输出目录 ==========
        base_output = config.get("output_path", "outputs")
        run_name = config.get("run_name")

        self.run_dir = build_run_dir(base_output, run_name)

        self.result_file = os.path.join(self.run_dir, "results.jsonl")
        self.summary_file = os.path.join(self.run_dir, "summary.json")
        self.config_file = os.path.join(self.run_dir, "config.json")

        # ========== DataFilter ==========
        filter_config = config.get("filter")
        self.filter = DataFilter(
            categories_include=filter_config.get("categories_include") if filter_config else None,
            categories_exclude=filter_config.get("categories_exclude") if filter_config else None,
            custom_filter=filter_config.get("custom_filter") if filter_config else None,
        ) if filter_config else None

        # ========== 文件锁 ==========
        self._lock = threading.Lock()

        # ========== 报告配置 ==========
        self.report_config = config.get("report", {})
        self.report_formats = self.report_config.get("formats", ["json", "markdown"])
        self.report_output_dir = self.report_config.get("output_dir", self.run_dir)

        # 保存 config
        self._save_config()

    def set_visualizer(self, visualizer):
        """设置可视化器"""
        self.visualizer = visualizer
        # 启动监控
        if hasattr(visualizer, 'start_monitoring'):
            visualizer.start_monitoring(self.result_file, self.summary_file)

    def _update_stage(self, stage: str, item_id: str = None, detail: str = None):
        """更新监控阶段"""
        if self.visualizer and hasattr(self.visualizer, 'monitor') and self.visualizer.monitor:
            self.visualizer.monitor.set_stage(stage, item_id, detail)

    def _append_log(self, message: str, level: str = "info"):
        """添加日志"""
        if self.visualizer and hasattr(self.visualizer, 'monitor') and self.visualizer.monitor:
            self.visualizer.monitor.add_log(message, level)

    def _save_config(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def _append_record(self, record):
        with self._lock:
            with open(self.result_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _process_one(self, item):
        try:
            # ===== 阶段: Dataset (已完成，加载下一条) =====
            self._update_stage("dataset", item.id, f"处理数据: {item.id[:20]}...")

            # ===== 构造输入 =====
            self._update_stage("prompt", item.id, "构建 Prompt")
            model_input = self.prompt_builder.build(item)

            # ===== 模型推理 =====
            self._update_stage("model", item.id, "模型推理中...")
            model_output = self.model.generate(model_input)

            # ===== 评估 =====
            self._update_stage("eval", item.id, "评估中...")
            metrics_all = {}
            for evaluator in self.evaluators:
                metrics = evaluator.evaluate(model_output.get_text(), item)
                metrics_all.update(metrics)

            # ===== 完整记录 =====
            self._update_stage("result", item.id, "写入结果")
            record = {
                "id": item.id,
                "item": safe_serialize(item.__dict__),
                "model_input": safe_serialize(
                    getattr(model_input, "__dict__", model_input)
                ),
                "model_output": safe_serialize(
                    getattr(model_output, "__dict__", model_output)
                ),
                "prediction": model_output.get_text(),
                "metrics": metrics_all,
            }
            self._append_log(f"完成评估: {item.id}", "success")

        except Exception as e:
            # 出错也记录
            self._append_log(f"评估失败: {item.id} - {str(e)}", "error")
            record = {
                "id": getattr(item, "id", None),
                "item": safe_serialize(getattr(item, "__dict__", {})),
                "error": str(e),
            }

        # ===== 立即写入（关键）=====
        self._append_record(record)

        return record

    def _load_done_ids(self):
        done_ids = set()

        if not os.path.exists(self.result_file):
            return done_ids

        with open(self.result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        done_ids.add(data["id"])
                except Exception:
                    continue

        return done_ids

    def _generate_reports(self):
        """Generate reports in configured formats."""
        if not self.report_formats:
            return

        logger.info(f"Generating reports: {self.report_formats}")

        # Load results from jsonl
        results = load_results_from_jsonl(self.result_file)
        if not results:
            logger.warning("No results to generate report")
            return

        for format_name in self.report_formats:
            try:
                report = Registry.create(format_name, "report")
                report.add_results(results)
                report.set_metadata("run_dir", self.run_dir)
                report.set_metadata("total_items", len(results))

                output_path = os.path.join(
                    self.report_output_dir,
                    f"report_{format_name}.{format_name}"
                )
                report.save(output_path)
                logger.info(f"Report saved: {output_path}")
            except Exception as e:
                logger.error(f"Failed to generate {format_name} report: {e}")

    def run(self):
        data = self.dataset.load()
        total = len(data)
        logger.info(f"Loaded {total} items from dataset")

        # ===== 应用 DataFilter =====
        if self.filter:
            data = self.filter.apply(data)
            logger.info(f"After filtering: {len(data)} items")

        # ===== 断点续跑 =====
        done_ids = self._load_done_ids()
        remaining = [item for item in data if item.id not in done_ids]
        logger.info(f"Resuming: {len(remaining)} items to process (already done: {len(done_ids)})")

        metric_agg = defaultdict(list)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []

            for item in remaining:
                futures.append(executor.submit(self._process_one, item))

            completed = 0
            for future in as_completed(futures):
                record = future.result()
                completed += 1

                if completed % 10 == 0 or completed == len(remaining):
                    logger.info(f"Progress: {completed}/{len(remaining)}")

                if "metrics" in record:
                    for k, v in record["metrics"].items():
                        metric_agg[k].append(v)

        # ===== 汇总指标 =====
        final_metrics = {
            k: sum(v) / len(v)
            for k, v in metric_agg.items()
            if len(v) > 0
        }

        logger.info(f"Evaluation complete. Metrics: {final_metrics}")

        # ===== 保存 summary =====
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)

        # ===== 生成报告 =====
        self._generate_reports()

        return final_metrics