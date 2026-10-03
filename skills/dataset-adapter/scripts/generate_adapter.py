#!/usr/bin/env python3
"""
数据集适配器生成器

从数据集分析报告和实际数据文件生成适配器代码。

用法:
    python generate_adapter.py <dataset_name> [--report path/report.md] [--data-path path/data.jsonl]
    python generate_adapter.py <dataset_name> --mode qa|multi_choice|generative

示例:
    python generate_adapter.py GPQA --report dataset_report/GPQA_Dataset_Report.md
    python generate_adapter.py GPQA --data-path data/GPQA/test.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ADAPTER_DIR = BASE_DIR / "adapter"
CONFIG_DIR = BASE_DIR / "configs"
REPORT_DIR = BASE_DIR / "dataset_report"


def to_slug(name: str) -> str:
    """数据集名转小写蛇形"""
    s = re.sub(r'[^a-zA-Z0-9]', '_', name)
    s = re.sub(r'_+', '_', s).strip('_').lower()
    return s


def to_class_name(name: str) -> str:
    """数据集名转驼峰类名"""
    s = re.sub(r'[^a-zA-Z0-9]', ' ', name)
    words = s.split()
    return ''.join(w.capitalize() for w in words)


def read_report(report_path: str) -> str:
    """读取报告文件"""
    if not os.path.exists(report_path):
        print(f"⚠️  报告不存在: {report_path}")
        return ""
    with open(report_path, encoding="utf-8") as f:
        return f.read()


def read_data_sample(data_path: str) -> Tuple[Optional[Dict], str]:
    """读取数据样本"""
    if not os.path.exists(data_path):
        return None, ""

    ext = Path(data_path).suffix.lower()
    try:
        if ext == ".jsonl":
            with open(data_path, encoding="utf-8-sig") as f:
                line = f.readline().strip()
                if line:
                    return json.loads(line), "jsonl"
        elif ext == ".json":
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data[0], "json"
                elif isinstance(data, dict):
                    # 可能带 split 的 HF 格式
                    for key in ['train', 'test', 'validation', 'data', 'records']:
                        if key in data and isinstance(data[key], list) and data[key]:
                            return data[key][0], "json"
                    return list(data.values())[0][0] if data else None, "json"
        return None, ext
    except Exception as e:
        print(f"⚠️  读取数据失败: {e}")
        return None, ext


def extract_fields_from_report(report_text: str) -> Dict:
    """从报告中提取字段信息"""
    fields = {}

    # 查找数据字段说明表
    table_pattern = r'\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*([^|]+)\s*\|'
    in_field_section = False
    saw_rows = False
    for line in report_text.split('\n'):
        line = line.strip()
        if '数据字段说明' in line or '| 字段名 | 类型 | 说明' in line.replace(' ', ''):
            in_field_section = True
            continue
        if in_field_section:
            if not line.startswith('|') or line.startswith('|--') or line.startswith('| -'):
                # 空行或分隔行之后，遇到非表格内容结束
                if saw_rows and (not line.startswith('|')):
                    break
                continue
            match = re.match(table_pattern, line)
            if match:
                saw_rows = True
                if match.group(1) not in ('字段名', '---'):
                    fields[match.group(1)] = {
                        'type': match.group(2),
                        'desc': match.group(3).strip()
                    }

    # 查找来源链接
    source_urls = {}
    url_patterns = [
        r'数据集链接[：:]\s*(https?://[^\s\)]+)',
        r'项目仓库[：:]\s*(https?://[^\s\)]+)',
        r'论文链接[：:]\s*(https?://[^\s\)]+)',
    ]
    for pat in url_patterns:
        match = re.search(pat, report_text)
        if match:
            key = pat.split(r'[：:]')[0]
            source_urls[key] = match.group(1).rstrip(')')

    return {
        'fields': fields,
        'source_urls': source_urls,
    }


def guess_mode(sample: Dict, fields: Dict) -> str:
    """猜测适配模式"""
    if not sample:
        return "qa"

    keys = set(sample.keys())
    # 多选题检测
    direct_opts = ['A', 'B', 'C', 'D', 'choice_A', 'choice_B', 'choice_C', 'choice_D']
    if all(opt in keys for opt in ['A', 'B', 'C', 'D']):
        return "multi_choice"
    # GPQA 风格: option_A / option_B / option_C / option_D
    option_keys = [k for k in keys if re.match(r'^(option|choice)_[A-D]$', k)]
    if len(option_keys) >= 2:
        return "multi_choice"
    if any('option' in k.lower() or 'choice' in k.lower() for k in keys) and any('answer' in k.lower() for k in keys):
        return "multi_choice"
    if 'correct_answer' in keys or 'choices' in keys:
        return "multi_choice"

    # 检查是否有 checklist / criteria（LLM-as-Judge 模式）
    if 'checklist' in keys or 'criteria' in keys:
        return "generative"

    # 检查是否有评分标准
    if any('score' in k.lower() or 'judge' in k.lower() or 'eval' in k.lower() for k in keys):
        return "generative"

    # 默认为 QA
    return "qa"


def generate_adapter_init(
    dataset_name: str,
    sample: Dict,
    mode: str,
    report_info: Dict,
    output_dir: Path
) -> str:
    """生成适配器 __init__.py"""
    slug = to_slug(dataset_name)
    class_name = to_class_name(dataset_name)
    registry_name = dataset_name

    # 从报告中提取来源
    source_urls = report_info.get('source_urls', {})
    sources = '; '.join(f'{k}: {v}' for k, v in source_urls.items())

    # 字段映射分析
    question_fields = ['question', 'problem', 'prompt', 'query', 'instruction', 'input']
    answer_fields = ['correct_answer', 'answer', 'reference', 'solution', 'label', 'output', 'target']
    category_fields = ['category', 'subject', 'domain', 'topic', 'task', 'type', 'field']
    difficulty_fields = ['difficulty', 'level', 'grade', 'hardness']

    keys = set(sample.keys()) if sample else set()
    q_field = next((k for k in question_fields if k in keys), 'question')
    a_field = next((k for k in answer_fields if k in keys), 'answer')
    c_field = next((k for k in category_fields if k in keys), None)
    d_field = next((k for k in difficulty_fields if k in keys), None)

    option_fields = [k for k in keys if re.match(r'^(option|choice)_?[A-D]$', k) or re.match(r'^[A-D]$', k)]

    # 获取字段说明
    fields_info = report_info.get('fields', {})

    # 生成字段说明注释
    field_comments = "\n".join(
        f"    - {f}: {info['desc']}" if isinstance(info, dict) and 'desc' in info
        else f"    - {f}: {info}"
        for f, info in list(fields_info.items())[:10]
    )

    # 多选题特殊处理
    options_code = ""
    if mode == "multi_choice":
        # 判断选项字段格式：直接 A/B/C/D 还是 option_A / choice_A
        option_suffix = ''  # 最终确定为 '_'、'' 或 None
        if sample:
            candidates = ['option_', 'choice_', '_option', '_choice', '']
            for cand in candidates:
                if cand + 'A' in sample or cand + 'B' in sample:
                    option_suffix = cand
                    break
        options_code = f"""
        # 构建完整问题（包含选项）
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            val = data_item.get('{option_suffix}' + opt) if '{option_suffix}' else data_item.get(opt)
            if val:
                options.append(f"{{opt}}. {{val}}")
        if options:
            question = f"{{question}}\\n" + "\\n".join(options)
"""

    # 元数据字段
    meta_fields = []
    if c_field:
        meta_fields.append(f"'{c_field}': data_item.get('{c_field}', '')")
    if d_field:
        meta_fields.append(f"'difficulty': data_item.get('{d_field}', '')")
    # 添加其他可能有用的字段
    useful_extras = ['id', 'index', 'source', 'year', 'explanation']
    for ef in useful_extras:
        if sample and ef in sample:
            meta_fields.append(f"'{ef}': data_item.get('{ef}', '')")
    if mode == "generative":
        meta_fields.append("'checklist': data_item.get('checklist', [])")
        meta_fields.append("'criteria': data_item.get('criteria', [])")

    meta_code = ",\n                ".join(meta_fields) if meta_fields else ""

    # 分类代码
    category_code = ""
    if c_field:
        category_code = f"""
            category=[
                data_item.get('{c_field}', ''),
            ],"""
    if d_field:
        category_code += f"""
            difficulty=data_item.get('{d_field}', ''),"""

    # 生成适配器代码
    code = f'''r"""
{dataset_name} 适配器

数据集来源: {sources if sources else "待补充"}
数据集报告: dataset_report/{dataset_name}_Dataset_Report.md
"""

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("{registry_name}", "dataset")
class {class_name}Dataset(BaseDataset):
    """
    {dataset_name} 数据集适配器{field_comments}
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("{registry_name} requires 'data_path' config")
        self.dataset_name = "{registry_name}"

    def load_raw_data(self):
        import os
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {{self.data_path}}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        question = (
            data_item.get('{q_field}') or
            data_item.get('question') or
            data_item.get('problem') or
            data_item.get('prompt', '')
        )
        answer = (
            data_item.get('{a_field}') or
            data_item.get('answer') or
            data_item.get('reference') or
            data_item.get('solution', '')
        ){options_code}
        return DataItem(
            id=self.build_id(data_item.get('id', data_item.get('index', ''))),
            prompt=question,
            reference=answer,
            metadata={{{meta_code}
            }},{category_code}
        )
'''

    return code


def generate_evaluator(dataset_name: str, mode: str) -> str:
    """生成评估器代码"""
    slug = to_slug(dataset_name)
    class_name = to_class_name(dataset_name)

    if mode == "multi_choice":
        return f'''"""
{dataset_name} 评估器
"""

import re
from core.base import DataItem
from core.registry import Registry
from evaluators.base import BaseEvaluator


@Registry.register("{slug}", "evaluator")
class {class_name}Evaluator(BaseEvaluator):
    """{dataset_name} 多选题评估器"""

    def __init__(self, config):
        super().__init__(config)

    def evaluate(self, pred: str, item: DataItem) -> dict:
        reference = str(item.reference).strip().upper()
        pred = pred.strip().upper()
        # 优先匹配 "answer is X" / "选 X" 等强信号模式
        strong_patterns = [
            r"answer\\s+is\\s+([A-D])",
            r"选\\s*([A-D])",
            r"my\\s+answer\\s+is\\s+([A-D])",
            r"the\\s+answer\\s+is\\s+([A-D])",
        ]
        for pat in strong_patterns:
            match = re.search(pat, pred, re.IGNORECASE)
            if match:
                selected = match.group(1)
                return {{"accuracy": 1.0 if selected == reference else 0.0}}
        # 提取最后一个独立出现的选项字母
        matches = re.findall(r"\\b([A-D])\\b", pred)
        selected = matches[-1] if matches else None
        if selected:
            return {{"accuracy": 1.0 if selected == reference else 0.0}}
        # 兜底：直接比对
        return {{"accuracy": 1.0 if pred == reference else 0.0}}
'''

    elif mode == "generative":
        return f'''"""
{dataset_name} 评估器

TODO: 此评估器为桩实现，仅返回 accuracy=0.0。
      请参考 adapter/writingbench/evaluator.py 实现真实 LLM-as-Judge 逻辑。
"""

import logging
from core.base import DataItem
from core.registry import Registry
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


@Registry.register("{slug}", "evaluator")
class {class_name}Evaluator(BaseEvaluator):
    """{dataset_name} 生成式评估器（LLM-as-Judge）"""

    def __init__(self, config):
        super().__init__(config)
        self.judge_model = config.get("judge_model", "claude")
        logger.warning(
            "[{slug}] 此评估器为桩实现，返回 accuracy=0.0，"
            "请替换为真实 LLM-as-Judge 评分逻辑。"
            "参考 adapter/writingbench/evaluator.py"
        )

    def evaluate(self, pred: str, item: DataItem) -> dict:
        # TODO: 实现真实 LLM-as-Judge 评分逻辑
        return {{"accuracy": 0.0, "note": "LLM-as-Judge 评估需要实现详细评分逻辑"}}


@Registry.register("{slug}_simple", "evaluator")
class {class_name}SimpleEvaluator(BaseEvaluator):
    """{dataset_name} 简单评估器"""

    def evaluate(self, pred: str, item: DataItem) -> dict:
        reference = str(item.reference)
        if not pred or not reference:
            return {{"accuracy": 0.0}}
        # 简单关键词匹配
        return {{"accuracy": 1.0 if reference.lower() in pred.lower() else 0.0}}
'''

    else:
        return f'''"""
{dataset_name} 评估器
"""

from core.base import DataItem
from core.registry import Registry
from evaluators.base import BaseEvaluator


@Registry.register("{slug}", "evaluator")
class {class_name}Evaluator(BaseEvaluator):
    """{dataset_name} 标准评估器"""

    def __init__(self, config):
        super().__init__(config)

    def evaluate(self, pred: str, item: DataItem) -> dict:
        reference = str(item.reference).strip()
        pred = pred.strip()
        accuracy = 1.0 if pred == reference else 0.0
        return {{"accuracy": accuracy}}
'''


def generate_config(dataset_name: str, data_path: str, mode: str) -> str:
    """生成配置 YAML"""
    registry_name = dataset_name
    if mode == "multi_choice":
        evaluator = f"      - name: {to_slug(dataset_name)}"
    elif mode == "generative":
        evaluator = f"      - name: {to_slug(dataset_name)}"
    else:
        evaluator = "      - name: accuracy"

    return f"""tasks:
  - name: standard
    type: standard
    dataset:
      name: {registry_name}
      params:
        data_path: "{data_path}"
        limits: 10  # 测试用，完成后可移除
    prompt_builder:
      name: qa_builder
    model:
      name: MiniMax
      params:
        generation_config:
          temperature: 0.5
    evaluators:
{evaluator}
    output_path: results/{to_slug(dataset_name)}
"""


def main():
    import sys
    # Windows console encoding fix for emoji output
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="数据集适配器生成器")
    parser.add_argument("dataset_name", help="数据集名称（如 GPQA, MMLU-Pro）")
    parser.add_argument("--report", help="数据集报告路径")
    parser.add_argument("--data-path", help="数据文件路径")
    parser.add_argument("--mode", choices=["qa", "multi_choice", "generative"],
                        help="适配模式（自动检测不指定时）")
    parser.add_argument("--output-dir", default=str(ADAPTER_DIR),
                        help="适配器输出目录")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    slug = to_slug(dataset_name)
    output_dir = Path(args.output_dir) / slug

    # 读取报告
    if not args.report:
        auto_report = REPORT_DIR / f"{dataset_name}_Dataset_Report.md"
        if auto_report.exists():
            args.report = str(auto_report)
            print(f"📄 自动找到报告: {args.report}")
        else:
            print(f"⚠️  未找到报告，尝试不依赖报告生成")

    report_text = read_report(args.report) if args.report else ""
    report_info = extract_fields_from_report(report_text) if report_text else {}
    print(f"📊 从报告中提取的字段: {list(report_info.get('fields', {}).keys())}")

    # 读取数据样本
    if not args.data_path:
        data_dir = BASE_DIR / "data" / dataset_name
        if data_dir.exists():
            files = list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.json")) + \
                    list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))
            if files:
                args.data_path = str(files[0])
                print(f"📁 自动找到数据: {args.data_path}")

    sample, data_fmt = read_data_sample(args.data_path) if args.data_path else (None, "")
    if sample:
        print(f"📝 数据样本示例:")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:800])
    else:
        print(f"⚠️  未找到或无法读取数据文件")

    # 确定模式
    mode = args.mode or guess_mode(sample, report_info.get('fields', {}))
    print(f"🔧 适配模式: {mode}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成适配器
    init_code = generate_adapter_init(
        dataset_name, sample, mode, report_info, output_dir
    )
    with open(output_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(init_code)
    print(f"✅ 生成适配器: {output_dir / '__init__.py'}")

    # 生成评估器（非 QA 模式需要自定义评估器）
    if mode != "qa":
        eval_code = generate_evaluator(dataset_name, mode)
        with open(output_dir / "evaluator.py", "w", encoding="utf-8") as f:
            f.write(eval_code)
        print(f"✅ 生成评估器: {output_dir / 'evaluator.py'}")

    # 生成配置
    data_path = args.data_path or f"data/{dataset_name}/data.{data_fmt or 'jsonl'}"
    config_code = generate_config(dataset_name, data_path, mode)
    config_path = CONFIG_DIR / f"{slug}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_code)
    print(f"✅ 生成配置: {config_path}")

    print(f"\n🎉 适配器生成完成!")
    print(f"  适配器目录: {output_dir}")
    print(f"  配置: {config_path}")
    print(f"  注册名: {dataset_name}")
    print(f"  评估器: {to_slug(dataset_name) if mode != 'qa' else 'accuracy'}")
    print(f"\n下一步:")
    print(f"  1. 检查生成的适配器代码")
    print(f"  2. 运行测试: python cli/main.py --config {config_path}")
    print(f"  3. 如果数据字段有差异，编辑 {output_dir / '__init__.py'} 修正映射")


if __name__ == "__main__":
    main()