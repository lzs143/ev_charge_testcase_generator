"""自然语言需求到动作级可执行测试用例的主流程。"""

from __future__ import annotations

from dataclasses import dataclass

from .executable_extractor import RuleBasedExecutableExtractor
from .executable_generator import ExecutableTestCaseGenerator
from .extractor import RuleBasedExtractor
from .models import CheckResult, ExecutableTestCase, ExtractedInfo, Requirement
from .preprocessing import preprocess_text
from .semantic_extraction_factory import RULE_BASED_METHOD, build_semantic_extractor


@dataclass
class PipelineResult:
    """主流程输出结果。"""

    requirement: Requirement
    extracted_info: ExtractedInfo
    executable_info: dict
    test_case: ExecutableTestCase
    check_result: CheckResult


class TestCaseGenerationPipeline:
    """规则版可执行语义测试用例生成流程。"""

    def __init__(
        self,
        semantic_method: str = RULE_BASED_METHOD,
        model_dir: str | None = None,
        device: str = "auto",
    ) -> None:
        """初始化抽取器和生成器。"""

        self.extractor = RuleBasedExtractor()
        self.semantic_method = semantic_method
        self.semantic_extractor = build_semantic_extractor(semantic_method, model_dir=model_dir, device=device)
        self.executable_extractor = RuleBasedExecutableExtractor(self.semantic_extractor)
        self.generator = ExecutableTestCaseGenerator()

    def run(self, text: str) -> PipelineResult:
        """根据一条自然语言需求生成动作级可执行测试用例。"""

        normalized_text = preprocess_text(text)
        if not normalized_text:
            raise ValueError("输入需求不能为空")

        requirement = Requirement(
            requirement_id=self._build_requirement_id(normalized_text),
            text=normalized_text,
        )
        extracted_info = self.extractor.extract(normalized_text)
        executable_info = self.executable_extractor.extract(normalized_text, extracted_info)
        test_case = self.generator.generate(executable_info)
        if test_case is None:
            reason = executable_info.get("reject_reason", "输入文本不是有效的充电测试需求")
            raise ValueError(reason)

        check_result = self._check_executable_case(test_case)
        return PipelineResult(
            requirement=requirement,
            extracted_info=extracted_info,
            executable_info=executable_info,
            test_case=test_case,
            check_result=check_result,
        )

    def run_batch(self, texts: list[str]) -> list[PipelineResult]:
        """批量生成动作级可执行测试用例。"""

        return [self.run(text) for text in texts]

    @staticmethod
    def _check_executable_case(test_case: ExecutableTestCase) -> CheckResult:
        """检查可执行测试用例是否具备自动化执行所需的基本结构。"""

        errors: list[str] = []
        warnings: list[str] = []

        if not test_case.preconditions:
            errors.append("可执行测试用例缺少前置条件")
        if not test_case.steps:
            errors.append("可执行测试用例缺少动作步骤")
        if not test_case.cleanup_steps:
            warnings.append("可执行测试用例缺少清理步骤")
        if any(step.action_id == "UNMAPPED" for step in test_case.steps):
            warnings.append("存在未映射的动作步骤")
        if any(step.action_id == "UNMAPPED" for step in test_case.cleanup_steps):
            warnings.append("存在未映射的清理动作")
        if any(assertion.assertion_id == "PJ-UNMAPPED" for assertion in test_case.assertions):
            warnings.append("存在未映射的判据")

        return CheckResult(passed=not errors, errors=errors, warnings=warnings)

    @staticmethod
    def _build_requirement_id(text: str) -> str:
        """生成稳定的需求编号。"""

        return f"REQ_{abs(hash(text)) % 1_000_000:06d}"
