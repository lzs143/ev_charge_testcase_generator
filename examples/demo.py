"""完整流水线运行示例。"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ev_charge_testcase_generator.exporter import export_cases_to_excel, export_cases_to_json
from ev_charge_testcase_generator.pipeline import TestCaseGenerationPipeline


DEMO_REQUIREMENTS = [
    "车辆完成握手后，设置直流充电电压为500V，电流为30A。",
    "直流充电过程中，当BMS通信中断时，系统应停止输出并记录故障信息。",
    "直流充电过程中，当绝缘故障发生时，应停止输出并进入停机流程。",
    "交流充电过程中，当CP信号异常时，应停止充电并提示异常状态。",
    "交流充电连接确认完成后，进入导引控制并判断是否允许充电。",
]


def main() -> None:
    """生成测试用例，打印 JSON，并导出到 outputs 目录。"""

    project_root = Path(__file__).resolve().parents[1]
    pipeline = TestCaseGenerationPipeline()
    output_dir = project_root / "outputs"

    results = pipeline.run_batch(DEMO_REQUIREMENTS)
    test_cases = [result.test_case for result in results]

    for test_case in test_cases:
        print(test_case.to_json())

    export_cases_to_json(test_cases, output_dir)
    export_cases_to_excel(test_cases, output_dir / "test_cases.xlsx")


if __name__ == "__main__":
    main()
