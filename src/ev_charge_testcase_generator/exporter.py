"""可执行测试用例导出模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook

from .models import ExecutableTestCase


EXECUTABLE_EXCEL_HEADERS = [
    "section",
    "step_id",
    "action_id",
    "action_name",
    "action_type",
    "target",
    "message",
    "signal",
    "parameters",
    "duration_ms",
    "timeout_ms",
    "assertion_id",
    "assertion_type",
    "operator",
    "expected_value",
    "description",
    "required",
]


def export_to_json(test_case: ExecutableTestCase, output_path: Path | str) -> Path:
    """导出单个可执行测试用例到 JSON 文件。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(test_case.to_json(), encoding="utf-8")
    return path


def export_to_excel(test_case: ExecutableTestCase, output_path: Path | str) -> Path:
    """导出单个可执行测试用例到 Excel 文件。"""

    return export_cases_to_excel([test_case], output_path)


def export_cases_to_json(test_cases: list[ExecutableTestCase], output_dir: Path | str) -> list[Path]:
    """批量导出可执行测试用例到 JSON 文件。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        export_to_json(test_case, directory / f"{test_case.case_id}.json")
        for test_case in test_cases
    ]


def export_cases_to_excel(test_cases: list[ExecutableTestCase], output_path: Path | str) -> Path:
    """批量导出可执行测试用例到 Excel 文件。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "test_cases"
    worksheet.append(
        [
            "case_id",
            "case_name",
            "scene_type",
            "condition_type",
            "test_type",
            "standard_source",
            "test_stage",
            "target_object",
            *EXECUTABLE_EXCEL_HEADERS,
        ]
    )
    for test_case in test_cases:
        _append_executable_case_rows(worksheet, test_case)
    workbook.save(path)
    return path


def _append_executable_case_rows(worksheet: Any, test_case: ExecutableTestCase) -> None:
    """将可执行测试用例拆分为前置、步骤、判据和清理行。"""

    base = [
        test_case.case_id,
        test_case.case_name,
        test_case.scene_type,
        test_case.condition_type,
        test_case.test_type,
        test_case.standard_source or "",
        test_case.test_stage or "",
        test_case.target_object or "",
    ]

    for precondition in test_case.preconditions:
        worksheet.append(
            [
                *base,
                "precondition",
                precondition.condition_id,
                "",
                precondition.description,
                "",
                precondition.target or "",
                "",
                "",
                json.dumps(precondition.parameters, ensure_ascii=False),
                "",
                "",
                "",
                "",
                "",
                "",
                precondition.description,
                precondition.required,
            ]
        )

    for step in test_case.steps:
        worksheet.append(
            [
                *base,
                "step",
                step.step_id,
                step.action_id or "",
                step.action_name,
                step.action_type,
                step.target or "",
                step.message or "",
                step.signal or "",
                json.dumps(step.parameters, ensure_ascii=False),
                step.duration_ms or "",
                step.timeout_ms or "",
                "",
                "",
                "",
                "",
                step.description,
                step.required,
            ]
        )

    for assertion in test_case.assertions:
        worksheet.append(
            [
                *base,
                "assertion",
                "",
                "",
                "",
                "",
                assertion.target or "",
                assertion.message or "",
                assertion.signal or "",
                "",
                "",
                assertion.timeout_ms or "",
                assertion.assertion_id or "",
                assertion.assertion_type,
                assertion.operator or "",
                assertion.expected_value or "",
                assertion.description,
                True,
            ]
        )

    for cleanup_step in test_case.cleanup_steps:
        worksheet.append(
            [
                *base,
                "cleanup",
                cleanup_step.step_id,
                cleanup_step.action_id or "",
                cleanup_step.action_name,
                "cleanup",
                "",
                "",
                "",
                json.dumps(cleanup_step.parameters, ensure_ascii=False),
                "",
                "",
                "",
                "",
                "",
                "",
                cleanup_step.action_name,
                cleanup_step.required,
            ]
        )


class JsonExporter:
    """兼容旧调用方式的 JSON 导出器。"""

    def export(self, test_case: ExecutableTestCase, output_path: Path | str) -> None:
        """导出 JSON 文件。"""

        export_to_json(test_case, output_path)
