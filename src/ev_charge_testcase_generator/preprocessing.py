"""文本预处理模块。"""

from __future__ import annotations


def preprocess_text(text: str) -> str:
    """对自然语言测试需求进行最小化清洗。"""

    # 第一阶段仅保留基础清洗，后续可加入分词、规范化等处理。
    return " ".join(text.strip().split())
