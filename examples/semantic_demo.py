"""语义抽取功能交互式测试脚本。"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ev_charge_testcase_generator.semantic_extractor import SemanticExtractor


def main() -> None:
    """运行交互式语义抽取演示。"""

    parser = argparse.ArgumentParser(description="测试自然语言充电测试需求语义抽取功能")
    parser.add_argument("--text", help="待抽取的自然语言测试需求")
    args = parser.parse_args()
    extractor = SemanticExtractor()

    if args.text:
        result = extractor.extract(args.text)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 80)
    print("🔍 语义抽取功能 - 交互式工具")
    print("=" * 80)
    print("输入自然语言测试需求，系统会进行语义抽取。")
    print("输入 'quit' 或 'exit' 退出。\n")

    while True:
        try:
            user_input = input("请输入测试需求 > ").strip()

            if not user_input:
                print("⚠️  输入不能为空，请重新输入。\n")
                continue

            if user_input.lower() in ("quit", "exit"):
                print("\n👋 再见！\n")
                break

            result = extractor.extract(user_input)
            print("\n" + "=" * 80)
            print(f"📝 输入: {user_input}")
            print("=" * 80)
            print("\n📊 语义抽取结果:\n")
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            print("\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见！\n")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
