"""交互式 GUI 工具：输入有效性判断与语义信息抽取。"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ev_charge_testcase_generator.executable_extractor import RuleBasedExecutableExtractor
from ev_charge_testcase_generator.semantic_extraction_factory import (
    RULE_BASED_METHOD,
    available_semantic_methods,
    build_semantic_extractor,
)
from ev_charge_testcase_generator.semantic_extractor import SemanticExtractor


class SemanticExtractionGUI:
    """语义抽取 GUI 应用。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("🔍 语义信息抽取工具")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # 初始化提取器
        self.method_labels = available_semantic_methods()
        self.method_display_to_key = {label: key for key, label in self.method_labels.items()}
        self.method_var = tk.StringVar(value=self.method_labels[RULE_BASED_METHOD])
        self.extractor_cache: dict[str, SemanticExtractor] = {}
        self.extractor = self._get_selected_extractor()
        self.executable_extractor = RuleBasedExecutableExtractor(self.extractor)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """设置用户界面。"""

        # 主容器
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 标题
        title = ttk.Label(main_frame, text="🔍 语义信息抽取工具", font=("Arial", 14, "bold"))
        title.pack(pady=(0, 10))

        code_note = ttk.Label(
            main_frame,
            text=(
                "界面文件：examples/gui_tool.py    "
                "核心功能：src/ev_charge_testcase_generator/semantic_extractor.py    "
                "下一步：src/ev_charge_testcase_generator/executable_extractor.py"
            ),
            foreground="#666666",
            font=("Arial", 9),
        )
        code_note.pack(pady=(0, 10))

        # ========== 上部：输入区域 ==========
        input_frame = ttk.LabelFrame(main_frame, text="📝 输入测试需求", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(input_frame, text="请输入自然语言形式的充电测试需求：").pack(anchor=tk.W)
        self.input_text = ttk.Entry(input_frame, width=80)
        self.input_text.pack(fill=tk.X, pady=5)
        self.input_text.bind("<Return>", lambda e: self._process())

        method_frame = ttk.Frame(input_frame)
        method_frame.pack(fill=tk.X, pady=5)
        ttk.Label(method_frame, text="语义抽取方式：").pack(side=tk.LEFT, padx=5)
        self.method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.method_var,
            values=list(self.method_display_to_key),
            state="readonly",
            width=32,
        )
        self.method_combo.pack(side=tk.LEFT, padx=5)
        self.method_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_extractors())

        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=5)
        self.process_button = ttk.Button(button_frame, text="✨ 检查并抽取", command=self._process)
        self.process_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🗑️  清空", command=self._clear).pack(side=tk.LEFT, padx=5)

        # ========== 下部：结果显示区域 ==========
        result_frame = ttk.LabelFrame(main_frame, text="📊 处理结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        # 创建 Notebook（标签页）
        notebook = ttk.Notebook(result_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # 1. 有效性检查页面
        self.validity_frame = ttk.Frame(notebook)
        notebook.add(self.validity_frame, text="1️⃣ 有效性判断（semantic_extractor.py）")
        self._setup_validity_tab()

        # 2. 语义抽取结果页面
        self.semantic_frame = ttk.Frame(notebook)
        notebook.add(self.semantic_frame, text="2️⃣ 语义信息（semantic_extractor.py）")
        self._setup_semantic_tab()

        # 3. 可执行用例语义页面
        self.executable_frame = ttk.Frame(notebook)
        notebook.add(self.executable_frame, text="3️⃣ 可执行语义（executable_extractor.py）")
        self._setup_executable_tab()

        # 4. 知识库展开测试步骤页面
        self.steps_frame = ttk.Frame(notebook)
        notebook.add(self.steps_frame, text="4️⃣ 测试步骤（sequence_expander.py）")
        self._setup_steps_tab()

    def _setup_validity_tab(self) -> None:
        """设置有效性判断显示页面。"""

        # 状态指示
        status_frame = ttk.Frame(self.validity_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=10)

        self.validity_icon = ttk.Label(status_frame, text="⏳", font=("Arial", 24))
        self.validity_icon.pack(side=tk.LEFT, padx=5)

        self.validity_label = ttk.Label(status_frame, text="等待输入...", font=("Arial", 12, "bold"))
        self.validity_label.pack(side=tk.LEFT, padx=10)

        # 有效性详情
        separator = ttk.Separator(self.validity_frame, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, padx=10, pady=5)

        # 创建可滚动文本框
        self.validity_text = scrolledtext.ScrolledText(
            self.validity_frame,
            wrap=tk.WORD,
            font=("Courier New", 10),
            state=tk.DISABLED,
        )
        self.validity_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 配置文本标签
        self.validity_text.tag_config("title", foreground="#1976D2", font=("Courier New", 11, "bold"))
        self.validity_text.tag_config("valid", foreground="#388E3C", font=("Courier New", 10, "bold"))
        self.validity_text.tag_config("invalid", foreground="#D32F2F", font=("Courier New", 10, "bold"))
        self.validity_text.tag_config("section", foreground="#F57C00", font=("Courier New", 10, "bold"))
        self.validity_text.tag_config("content", foreground="#333333")

    def _setup_semantic_tab(self) -> None:
        """设置语义信息显示页面。"""

        # 创建可滚动文本框显示 JSON 结果
        self.semantic_text = scrolledtext.ScrolledText(
            self.semantic_frame,
            wrap=tk.WORD,
            font=("Courier New", 9),
            state=tk.DISABLED,
        )
        self.semantic_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _setup_executable_tab(self) -> None:
        """设置可执行用例语义显示页面。"""

        # 展示语义抽取的下一步：面向可执行用例生成的结构化输入。
        self.executable_text = scrolledtext.ScrolledText(
            self.executable_frame,
            wrap=tk.WORD,
            font=("Courier New", 9),
            state=tk.DISABLED,
        )
        self.executable_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _setup_steps_tab(self) -> None:
        """设置知识库展开测试步骤显示页面。"""

        self.steps_text = scrolledtext.ScrolledText(
            self.steps_frame,
            wrap=tk.WORD,
            font=("Courier New", 10),
            state=tk.DISABLED,
        )
        self.steps_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _process(self) -> None:
        """处理用户输入。"""

        text = self.input_text.get().strip()
        if not text:
            self._show_validity("⏳", "等待输入...", "请在上方输入框中输入测试需求。\n")
            self._clear_semantic_text()
            self._clear_executable_text()
            self._clear_steps_text()
            return

        try:
            self._sync_extractors()
            self.process_button.config(text="处理中...", state=tk.DISABLED)
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            # 进行语义抽取（包括有效性判断）
            result = self.extractor.extract(text)

            # 显示有效性判断结果
            if result.is_valid:
                self._show_validity(
                    "✅",
                    "输入有效",
                    self._format_validity_valid(result),
                )
            else:
                self._show_validity(
                    "❌",
                    "输入无效",
                    self._format_validity_invalid(result),
                )

            # 显示语义抽取结果
            self._show_semantic(result)
            self._show_executable(result)

        except Exception as e:
            self._show_validity("❌", "处理出错", f"错误: {e}\n")
            self._clear_semantic_text()
            self._show_executable_error(str(e))
        finally:
            self.process_button.config(text="✨ 检查并抽取", state=tk.NORMAL)
            self.root.config(cursor="")

    def _selected_method_key(self) -> str:
        """读取当前 GUI 选择的语义抽取方式。"""

        return self.method_display_to_key.get(self.method_var.get(), RULE_BASED_METHOD)

    def _get_selected_extractor(self) -> SemanticExtractor:
        """按需创建并缓存语义抽取器，模型方式只在首次选择时加载。"""

        method = self._selected_method_key()
        if method not in self.extractor_cache:
            self.extractor_cache[method] = build_semantic_extractor(method)
        return self.extractor_cache[method]

    def _sync_extractors(self) -> None:
        """同步语义抽取器和后续可执行语义构建器。"""

        self.extractor = self._get_selected_extractor()
        self.executable_extractor = RuleBasedExecutableExtractor(self.extractor)

    def _format_validity_valid(self, result) -> str:
        """格式化有效输入的信息。"""

        output = ""
        output += f"✅ {result.message}\n\n"
        output += f"抽取方式：{self.method_var.get()}\n\n"

        output += "📋 识别信息：\n"
        output += f"  • 场景类型：{result.scene_type or '未识别'}\n"
        output += f"  • 工况类型：{result.condition_type}\n"
        output += f"  • 测试类型：{result.test_type}\n"
        output += f"  • 测试阶段：{result.test_stage or '未识别'}\n"
        output += f"  • 被测对象：{result.target_object or '未识别'}\n"
        output += f"  • 测试系统角色：{result.tester_role or '未识别'}\n"
        output += f"  • 主时序模板：{result.protocol_flow or '未识别'}\n"
        output += f"  • 置信度：{result.confidence:.1%}\n"

        warnings = self._visible_warnings(result.warnings)
        if warnings:
            output += f"\n⚠️ 警告信息：\n"
            for warning in warnings:
                output += f"  • {warning}\n"

        return output

    def _format_validity_invalid(self, result) -> str:
        """格式化无效输入的信息。"""

        output = ""
        output += f"❌ {result.message}\n\n"

        if result.warnings:
            output += "❌ 原因：\n"
            for warning in result.warnings:
                output += f"  • {warning}\n"

        output += "\n💡 正确输入示例：\n"
        for idx, example in enumerate(result.examples, 1):
            output += f"  {idx}. {example}\n"

        output += "\n📌 输入提示：\n"
        output += "  • 需要包含充电相关术语（充电、车辆、BMS、CP等）\n"
        output += "  • 需要包含测试意图（测试、当、若、应、设置等）\n"
        output += "  • 需要包含交流/直流场景，或包含可推断场景的协议、报文、信号等术语\n"

        return output

    def _show_validity(self, icon: str, status: str, content: str) -> None:
        """更新有效性检查面板。"""

        self.validity_icon.config(text=icon)
        self.validity_label.config(text=status)

        self.validity_text.config(state=tk.NORMAL)
        self.validity_text.delete(1.0, tk.END)
        self.validity_text.insert(tk.END, content)
        self.validity_text.config(state=tk.DISABLED)

    def _show_semantic(self, result) -> None:
        """显示语义抽取结果。"""

        # 构建结果字典，只显示有效时的语义信息
        if not result.is_valid:
            semantic_content = "❌ 输入无效，无法进行语义抽取。\n请根据上方「有效性判断」的建议修正输入。"
            self.semantic_text.config(state=tk.NORMAL)
            self.semantic_text.delete(1.0, tk.END)
            self.semantic_text.insert(tk.END, semantic_content)
            self.semantic_text.config(state=tk.DISABLED)
            return

        semantic_dict = self._build_semantic_dict(result)
        semantic_output = self._format_semantic_summary(result, semantic_dict)
        self.semantic_text.config(state=tk.NORMAL)
        self.semantic_text.delete(1.0, tk.END)
        self.semantic_text.insert(tk.END, semantic_output)
        self.semantic_text.config(state=tk.DISABLED)

    def _build_semantic_dict(self, result) -> dict:
        """构造语义抽取结构化结果，便于展示和调试。"""

        return {
            "status": "✅ 有效",
            "基本信息": {
                "原始文本": result.raw_text,
                "规范化文本": result.normalized_text,
                "语义抽取方式": self.method_var.get(),
            },
            "场景识别": {
                "场景类型": result.scene_type,
                "工况类型": result.condition_type,
                "测试类型": result.test_type,
                "标准来源": result.standard_source,
            },
            "阶段与对象": {
                "测试阶段": result.test_stage,
                "目标对象": result.target_object,
            },
            "协议流程上下文": {
                "被测对象": result.target_object,
                "测试系统角色": result.tester_role,
                "主时序模板": result.protocol_flow,
            },
            "测试要素": {
                "触发条件": result.trigger_condition,
                "故障类型": result.fault_type,
            },
            "参数与信号": {
                "参数": result.parameters,
                "报文类型": result.message_types,
                "信号": result.signals,
            },
            "测试动作": {
                "控制动作": result.actions,
                "预期结果": result.expected_results,
            },
            "质量指标": {
                "抽取置信度": f"{result.confidence:.1%}",
                "警告信息": self._visible_warnings(result.warnings),
            },
        }

    def _format_semantic_summary(self, result, semantic_dict: dict) -> str:
        """按测试用例生成视角展示关键语义，而不是直接倾倒 JSON。"""

        warnings = self._visible_warnings(result.warnings)
        output = ""
        output += "✅ 语义抽取结果\n"
        output += f"抽取方式：{self.method_var.get()}\n"
        output += f"原始需求：{result.raw_text}\n\n"

        output += "一、生成测试用例的关键语义\n"
        output += f"  场景类型：{result.scene_type or '未识别'}\n"
        output += f"  工况类型：{result.condition_type or '未识别'}\n"
        output += f"  测试类型：{result.test_type or '未识别'}\n"
        output += f"  测试阶段：{result.test_stage or '未识别'}\n"
        output += f"  故障类型：{result.fault_type or '无'}\n"
        output += f"  主时序模板：{result.protocol_flow or '未识别'}\n"
        output += f"  标准来源：{result.standard_source or '未识别'}\n\n"

        output += "二、对象、报文与信号\n"
        output += f"  被测对象：{result.target_object or '未识别'}\n"
        output += f"  测试系统角色：{result.tester_role or '未识别'}\n"
        output += f"  报文类型：{self._format_list(result.message_types)}\n"
        output += f"  物理/控制信号：{self._format_list(result.signals)}\n\n"

        output += "三、动作、条件与期望\n"
        output += f"  触发条件：{result.trigger_condition or '未明确'}\n"
        output += f"  控制动作：{self._format_list(result.actions)}\n"
        output += f"  预期结果：{self._format_list(result.expected_results)}\n\n"

        output += "四、参数\n"
        output += self._format_parameters(result.parameters)
        output += "\n"

        output += "五、质量信息\n"
        output += f"  抽取置信度：{result.confidence:.1%}\n"
        if warnings:
            output += "  警告信息：\n"
            for warning in warnings:
                output += f"    - {warning}\n"
        else:
            output += "  警告信息：无\n"

        output += "\n" + "-" * 72 + "\n"
        output += "结构化结果 JSON（用于调试和后续接口对接）\n"
        output += json.dumps(semantic_dict, ensure_ascii=False, indent=2)
        return output

    @staticmethod
    def _format_list(values: list[str]) -> str:
        """将列表格式化成适合阅读的一行文本。"""

        return "、".join(values) if values else "未识别"

    @staticmethod
    def _format_parameters(parameters: dict[str, str]) -> str:
        """格式化测试参数。"""

        if not parameters:
            return "  暂未提取到显式参数\n"
        output = ""
        for key, value in parameters.items():
            output += f"  {key}：{value}\n"
        return output

    @staticmethod
    def _visible_warnings(warnings: list[str]) -> list[str]:
        """过滤已经改为正式展示字段的抽取方式提示。"""

        return [warning for warning in warnings if not warning.startswith("语义抽取方式:")]

    def _clear_semantic_text(self) -> None:
        """清空语义抽取结果面板。"""

        self.semantic_text.config(state=tk.NORMAL)
        self.semantic_text.delete(1.0, tk.END)
        self.semantic_text.config(state=tk.DISABLED)

    def _show_executable(self, result) -> None:
        """显示可执行用例语义结果。"""

        if not result.is_valid:
            executable_content = "❌ 输入无效，无法构建可执行用例语义。\n请先根据「有效性判断」修正输入。"
            self._set_executable_text(executable_content)
            self._set_steps_text(executable_content)
            return

        # executable_extractor 生成后续 ExecutableTestCaseGenerator 可直接使用的结构。
        extracted_info = self.extractor.base_extractor.extract(result.normalized_text)
        executable_info = self.executable_extractor.extract(result.normalized_text, extracted_info)
        executable_dict = {
            "status": "✅ 已构建可执行用例语义",
            "模块": "RuleBasedExecutableExtractor",
            "输出用途": "ExecutableTestCaseGenerator.generate(...) 的输入",
            "executable_info": executable_info,
        }
        self._set_executable_text(json.dumps(executable_dict, ensure_ascii=False, indent=2))
        self._show_steps(executable_info)

    def _show_executable_error(self, message: str) -> None:
        """显示可执行语义构建异常。"""

        self._set_executable_text(f"❌ 可执行语义构建失败：{message}\n")
        self._set_steps_text(f"❌ 测试步骤生成失败：{message}\n")

    def _show_steps(self, executable_info: dict) -> None:
        """显示知识库展开后的测试步骤、判据和清理步骤。"""

        metadata = executable_info.get("metadata", {}).get("sequence_knowledge", {})
        output = ""
        output += "📌 知识库匹配结果\n"
        output += f"  • 是否展开：{'是' if executable_info.get('metadata', {}).get('sequence_expanded') else '否'}\n"
        output += f"  • 主时序：{metadata.get('flow_id') or executable_info.get('protocol_flow') or '未匹配'}\n"
        output += f"  • 匹配阶段：{metadata.get('matched_stage_name') or '未匹配'}\n"
        output += f"  • 匹配交互：{metadata.get('matched_interaction_name') or '未匹配'}\n"

        if not executable_info.get("metadata", {}).get("sequence_expanded"):
            output += "\n⚠️ 未匹配到可展开的时序知识，当前显示的是规则版粗粒度步骤。\n"

        output += "\n▶ 执行步骤\n"
        for step in executable_info.get("steps", []):
            output += self._format_step_line(step)

        output += "\n✓ 判定条件\n"
        assertions = executable_info.get("assertions", [])
        if assertions:
            for index, assertion in enumerate(assertions, start=1):
                assertion_id = assertion.get("assertion_id") or "UNMAPPED"
                description = assertion.get("description") or ""
                message = assertion.get("message")
                expected_value = assertion.get("expected_value")
                output += f"  {index:02d}. {assertion_id}  {description}\n"
                if message:
                    output += f"      报文：{message}\n"
                if expected_value:
                    output += f"      期望：{expected_value}\n"
        else:
            output += "  （未生成判定条件）\n"

        output += "\n↩ 清理步骤\n"
        cleanup_steps = executable_info.get("cleanup_steps", [])
        if cleanup_steps:
            for step in cleanup_steps:
                step_id = int(step.get("step_id", 0))
                action_id = step.get("action_id") or "UNMAPPED"
                action_name = step.get("action_name") or ""
                output += f"  {step_id:02d}. {action_id}  {action_name}\n"
        else:
            output += "  （未生成清理步骤）\n"

        self._set_steps_text(output)

    @staticmethod
    def _format_step_line(step: dict) -> str:
        """格式化单条执行步骤。"""

        step_id = int(step.get("step_id", 0))
        action_id = step.get("action_id") or "UNMAPPED"
        action_name = step.get("action_name") or ""
        action_type = step.get("action_type") or ""
        description = step.get("description") or ""
        message = step.get("message")
        signal = step.get("signal")
        duration_ms = step.get("duration_ms")
        timeout_ms = step.get("timeout_ms")

        output = f"  {step_id:02d}. {action_id}  {action_name}（{action_type}）\n"
        if description and description != action_name:
            output += f"      说明：{description}\n"
        if message:
            output += f"      报文：{message}\n"
        if signal:
            output += f"      信号：{signal}\n"
        if duration_ms is not None:
            output += f"      持续时间：{duration_ms} ms\n"
        if timeout_ms is not None:
            output += f"      超时/监测时间：{timeout_ms} ms\n"
        return output

    def _set_executable_text(self, content: str) -> None:
        """更新可执行语义结果面板。"""

        self.executable_text.config(state=tk.NORMAL)
        self.executable_text.delete(1.0, tk.END)
        self.executable_text.insert(tk.END, content)
        self.executable_text.config(state=tk.DISABLED)

    def _clear_executable_text(self) -> None:
        """清空可执行语义结果面板。"""

        self.executable_text.config(state=tk.NORMAL)
        self.executable_text.delete(1.0, tk.END)
        self.executable_text.config(state=tk.DISABLED)

    def _set_steps_text(self, content: str) -> None:
        """更新测试步骤结果面板。"""

        self.steps_text.config(state=tk.NORMAL)
        self.steps_text.delete(1.0, tk.END)
        self.steps_text.insert(tk.END, content)
        self.steps_text.config(state=tk.DISABLED)

    def _clear_steps_text(self) -> None:
        """清空测试步骤结果面板。"""

        self.steps_text.config(state=tk.NORMAL)
        self.steps_text.delete(1.0, tk.END)
        self.steps_text.config(state=tk.DISABLED)

    def _clear(self) -> None:
        """清空输入和输出。"""

        self.input_text.delete(0, tk.END)
        self.validity_icon.config(text="⏳")
        self.validity_label.config(text="等待输入...")

        self.validity_text.config(state=tk.NORMAL)
        self.validity_text.delete(1.0, tk.END)
        self.validity_text.insert(tk.END, "请在上方输入框中输入测试需求。\n")
        self.validity_text.config(state=tk.DISABLED)

        self._clear_semantic_text()
        self._clear_executable_text()
        self._clear_steps_text()


def main() -> None:
    """启动 GUI 应用。"""
    # 启动前输出诊断信息，写入启动日志以便在无窗口环境下排查
    try:
        start_msg = "Starting SemanticExtractionGUI..."
        print(start_msg, flush=True)
        log_path = Path(__file__).resolve().parents[1] / "outputs" / "gui_tool_start.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(start_msg + "\n")

        root = tk.Tk()
        app = SemanticExtractionGUI(root)
        # 标记 GUI 已创建
        created_msg = "GUI created, entering mainloop"
        print(created_msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(created_msg + "\n")

        root.mainloop()
    except Exception as e:
        err_msg = f"GUI failed to start: {e!r}"
        print(err_msg, flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(err_msg + "\n")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
