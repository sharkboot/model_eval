import json
import csv
import pandas as pd
import os

def read_json(file_path):
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def read_csv(file_path):
    """读取CSV文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def read_xlsx(file_path):
    """读取XLSX文件"""
    df = pd.read_excel(file_path)
    return df.to_dict('records')

def read_parquet(file_path):
    """读取Parquet文件"""
    df = pd.read_parquet(file_path)
    return df.to_dict('records')

def read_file(file_path):
    """根据文件扩展名读取不同格式的文件"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        return read_json(file_path)
    elif ext == '.jsonl':
        return read_jsonl(file_path)
    elif ext == '.csv':
        return read_csv(file_path)
    elif ext == '.xlsx':
        return read_xlsx(file_path)
    elif ext == '.parquet':
        return read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
