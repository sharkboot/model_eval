#!/usr/bin/env python3
"""
数据集下载与分析工具

用法:
    python download_dataset.py <dataset_name> --hf <hf_repo_id> [--file <filename>]
    python download_dataset.py <dataset_name> --url <direct_url> [--file <filename>]
    python download_dataset.py <dataset_name> --analyze-only --data-path <path>

示例:
    python download_dataset.py MyDataset --hf myorg/mydataset
    python download_dataset.py MyDataset --hf myorg/mydataset --file test.jsonl
    python download_dataset.py GPQA --url https://huggingface.co/datasets/..../file.jsonl
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def download_from_hf(dataset_name, hf_repo_id, filename=None):
    """从 Hugging Face 下载数据"""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("❌ 请先安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    target_dir = ensure_dir(BASE_DATA_DIR / dataset_name)
    print(f"📁 目标目录: {target_dir}")

    if filename:
        print(f"⬇️  下载文件: {filename} from {hf_repo_id} ...")
        local_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(target_dir),
        )
        print(f"✅ 下载完成: {local_path}")
        return local_path
    else:
        print(f"⬇️  下载整个数据集: {hf_repo_id} ...")
        downloaded = snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
        )
        print(f"✅ 下载完成: {downloaded}")
        return downloaded


def download_from_url(dataset_name, url, filename=None):
    """从直接 URL 下载"""
    target_dir = ensure_dir(BASE_DATA_DIR / dataset_name)
    if not filename:
        filename = url.split("/")[-1].split("?")[0] or "data.bin"
    local_path = target_dir / filename

    print(f"⬇️  下载: {url}")
    print(f"  → {local_path}")
    urllib.request.urlretrieve(url, str(local_path))
    print(f"✅ 下载完成: {local_path}")
    return str(local_path)


def analyze_data(path):
    """分析数据文件结构"""
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return

    ext = Path(path).suffix.lower()
    print(f"\n=== 数据分析: {path} ===")

    try:
        if ext == ".jsonl":
            data = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif ext == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # 检查是否包含 split 键
                data = data.get("data", data)
        elif ext in (".csv",):
            import csv
            reader = csv.DictReader(open(path, encoding="utf-8"))
            data = list(reader)
        elif ext == ".parquet":
            import pandas as pd
            data = pd.read_parquet(path).to_dict("records")
        else:
            # 尝试用 pandas 读取
            try:
                import pandas as pd
                data = pd.read_file(path).to_dict("records")
            except Exception:
                # 尝试当做 jsonl
                data = []
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        print("  文件可能不是标准 JSON/JSONL 格式")
        # 尝试读取前几行原文
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  RAW: {line[:200]}")
        return

    if not data:
        print("❌ 数据为空")
        return

    print(f"总数据量: {len(data)}")

    if isinstance(data[0], dict):
        keys = list(data[0].keys())
        print(f"字段: {keys}")

        # 字段类型分布
        types = {}
        for k in keys:
            vals = [d.get(k) for d in data[:100] if isinstance(d, dict)]
            if vals:
                types[k] = type(vals[0]).__name__
        print(f"字段类型: {types}")

        # 每条数据大小
        sizes = []
        for d in data[:50]:
            try:
                sizes.append(len(json.dumps(d, ensure_ascii=False)))
            except Exception:
                pass
        if sizes:
            print(f"平均大小: {sum(sizes)/len(sizes):.0f} bytes")
            print(f"大小范围: {min(sizes)} - {max(sizes)} bytes")

        # 关键字段检查
        print("\n关键字段检查:")
        for field in ["question", "problem", "prompt", "answer", "reference",
                      "solution", "category", "difficulty", "A", "B", "C", "D"]:
            present = field in keys
            if present:
                sample = data[0].get(field)
                print(f"  [OK] {field}: {str(sample)[:100]}")
            else:
                also_present = any(field.lower() in str(k).lower() for k in keys)
                if also_present:
                    print(f"  [!] {field}: 未精确匹配，但存在近似字段")

    else:
        print(f"数据类型: {type(data[0]).__name__}")
        print(f"示例: {str(data[0])[:500]}")

    # 输出摘要 JSON
    summary_path = Path(path).with_suffix(".summary.json")
    summary = {
        "path": str(path),
        "count": len(data),
        "fields": list(data[0].keys()) if isinstance(data[0], dict) else None,
        "sample": data[0] if data else None,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[摘要] 已保存: {summary_path}")


def main():
    import sys
    # Windows console encoding fix
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="数据集下载与分析工具")
    parser.add_argument("dataset_name", help="数据集名称（用于建目录）")
    parser.add_argument("--hf", help="Hugging Face repo id，如 myorg/mydataset")
    parser.add_argument("--file", help="只下载指定文件")
    parser.add_argument("--url", help="直接下载 URL")
    parser.add_argument("--data-path", help="分析已有数据文件")
    parser.add_argument("--analyze-only", action="store_true",
                        help="只分析已有文件，不下载")

    args = parser.parse_args()

    if args.analyze_only or args.data_path:
        if not args.data_path:
            auto = BASE_DATA_DIR / args.dataset_name
            candidates = list(auto.glob("*"))
            if len(candidates) == 1:
                args.data_path = str(candidates[0])
            else:
                print("❌ 请用 --data-path 指定文件路径")
                sys.exit(1)
        analyze_data(args.data_path)
        return

    if not args.hf and not args.url:
        print("❌ 需要指定 --hf 或 --url")
        parser.print_help()
        sys.exit(1)

    if args.hf:
        local_path = download_from_hf(args.dataset_name, args.hf, args.file)
    else:
        local_path = download_from_url(args.dataset_name, args.url, args.file)

    print(f"\n[完成] 数据集下载完成!")
    print(f"  存储路径: {local_path}")

    # 自动分析
    analyze_data(local_path)


if __name__ == "__main__":
    main()