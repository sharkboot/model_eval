"""共享数据字段归一化层

提供统一的 QA 字段映射工具，消除各 adapter 中重复的字段发现逻辑。
"""

from typing import Any, Dict, List, Optional


# 问题字段映射：按优先级顺序
QUESTION_KEYS = [
    "question", "problem", "prompt", "query", "instruction", "input",
]

# 答案字段映射：按优先级顺序
ANSWER_KEYS = [
    "answer", "reference", "solution", "label", "output", "target",
]

# 分类字段映射
CATEGORY_KEYS = [
    "category", "subject", "domain", "topic", "task", "type", "field",
]

# 难度字段映射
DIFFICULTY_KEYS = [
    "difficulty", "level", "grade", "hardness",
]


def normalize_qa_item(
    raw: Dict[str, Any],
    q_keys: Optional[List[str]] = None,
    a_keys: Optional[List[str]] = None,
    c_keys: Optional[List[str]] = None,
    d_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """将原始数据字典归一化为标准 QA 字段。

    Args:
        raw: 原始数据字典
        q_keys: 问题字段优先级列表，默认为 QUESTION_KEYS
        a_keys: 答案字段优先级列表，默认为 ANSWER_KEYS
        c_keys: 分类字段优先级列表，默认为 CATEGORY_KEYS
        d_keys: 难度字段优先级列表，默认为 DIFFICULTY_KEYS

    Returns:
        包含 question, answer, category, difficulty, metadata 的标准字典
    """
    q_keys = q_keys or QUESTION_KEYS
    a_keys = a_keys or ANSWER_KEYS
    c_keys = c_keys or CATEGORY_KEYS
    d_keys = d_keys or DIFFICULTY_KEYS

    # 提取问题（返回第一个非空值）
    question = ""
    for key in q_keys:
        val = raw.get(key)
        if val:
            question = str(val)
            break

    # 提取答案
    answer = ""
    for key in a_keys:
        val = raw.get(key)
        if val is not None:
            answer = str(val)
            break

    # 提取分类（支持 list 或 str）
    category = []
    for key in c_keys:
        val = raw.get(key)
        if val:
            if isinstance(val, list):
                category = [str(c) for c in val if c]
            else:
                category = [str(val)]
            break

    # 提取难度
    difficulty = ""
    for key in d_keys:
        val = raw.get(key)
        if val is not None:
            difficulty = str(val)
            break

    # 收集剩余字段作为 metadata
    known_keys = set(q_keys + a_keys + c_keys + d_keys)
    metadata = {k: v for k, v in raw.items() if k not in known_keys and v is not None}

    return {
        "question": question,
        "answer": answer,
        "category": category,
        "difficulty": difficulty,
        "metadata": metadata,
    }


def extract_options(raw: Dict[str, Any]) -> List[str]:
    """从原始数据中提取 A/B/C/D 选项。

    支持格式:
    - 直接键: A, B, C, D
    - 带前缀: option_A, choice_A, _option_A
    - 带后缀: A_value, A_text

    Returns:
        选项列表，如 ["A. 选项内容A", "B. 选项内容B", ...]
    """
    options = []

    # 尝试常见前缀模式
    prefixes = ["option_", "choice_", ""]
    for prefix in prefixes:
        has_all = all((prefix + opt) in raw for opt in ["A", "B", "C", "D"])
        if has_all:
            for opt in ["A", "B", "C", "D"]:
                val = raw.get(prefix + opt)
                if val:
                    options.append(f"{opt}. {val}")
            if options:
                return options

    # 尝试直接键
    direct_options = [opt for opt in ["A", "B", "C", "D"] if raw.get(opt)]
    if direct_options:
        for opt in ["A", "B", "C", "D"]:
            val = raw.get(opt)
            if val:
                options.append(f"{opt}. {val}")
        return options

    return options


def validate_qa_item(normalized: Dict[str, Any]) -> List[str]:
    """验证归一化后的 QA 项，返回缺失字段的警告列表。

    Returns:
        警告消息列表，为空表示验证通过
    """
    warnings = []
    if not normalized.get("question"):
        warnings.append("question 字段为空")
    if not normalized.get("answer"):
        warnings.append("answer 字段为空")
    return warnings
