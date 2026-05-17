import fs from "node:fs/promises";

const datasetPath = new URL("../evaluation/eval_dataset.json", import.meta.url);
const data = JSON.parse(await fs.readFile(datasetPath, "utf8"));
const messagePattern = /(?:EVCC|SECC|BMS|BHM|BRM|BCP|BCL|BCS|BSM|BEM|CHM|CRM|CML|BRO|CRO|CCS|CEM|BST|CST|BSD|CSD)/g;

function unique(values) {
  const result = [];
  for (const value of values) {
    if (value && !result.includes(value)) result.push(value);
  }
  return result;
}

function messagesFrom(item) {
  let values = item.text.match(messagePattern) || [];
  for (const expected of item.gold.expected_results || []) {
    values = values.concat(expected.match(messagePattern) || []);
  }
  return unique(values);
}

function targetObject(text, sceneType) {
  if (text.includes("EVCC")) return "EVCC";
  if (text.includes("SECC")) return "SECC";
  if (text.includes("BMS")) return "BMS";
  if (sceneType === "AC") return text.includes("车辆") ? "车辆" : "供电设备";
  return "充电系统";
}

function preconditions(item, target) {
  const descriptions =
    item.gold.scene_type === "DC"
      ? ["测试系统和被测对象完成物理连接", "直流充电通信配置完成"]
      : ["交流充电接口连接准备完成", "供电设备和车辆连接状态可检测"];
  if (item.test_stage) descriptions.push(`进入${item.test_stage}`);
  return descriptions.map((description, index) => ({
    condition_id: `PRE-${String(index + 1).padStart(3, "0")}`,
    description,
    target,
    parameters: {},
    required: true,
  }));
}

function actionType(description) {
  if (description.includes("等待") || description.includes("超时")) return "等待";
  if (description.includes("发送")) return "发送报文";
  if (
    description.includes("未收到") ||
    description.includes("不发送") ||
    description.includes("停发") ||
    description.includes("收不到")
  ) {
    return "故障注入";
  }
  if (description.includes("设置") || description.includes("配置")) return "设置参数";
  if (description.includes("停止") || description.includes("退出") || description.includes("中止")) return "状态控制";
  if (description.includes("插枪") || description.includes("连接")) return "连接控制";
  return "执行动作";
}

function steps(item, target, messages) {
  const sceneType = item.gold.scene_type;
  const parameters = item.gold.parameters || {};
  const result = [
    {
      step_id: 1,
      action_id: null,
      action_name: sceneType === "DC" ? "直流插枪初始化" : "交流插枪初始化",
      action_type: "初始化",
      target,
      parameters: {},
      message: null,
      signal: null,
      duration_ms: null,
      timeout_ms: null,
      required: true,
      description: "建立测试初始状态",
    },
    {
      step_id: 2,
      action_id: null,
      action_name: sceneType === "DC" ? "直流插枪" : "交流插枪",
      action_type: "连接控制",
      target,
      parameters: {},
      message: null,
      signal: sceneType === "DC" ? "CC2" : "CP/CC",
      duration_ms: null,
      timeout_ms: null,
      required: true,
      description: "建立充电连接",
    },
  ];

  let description = item.text;
  if (Object.keys(parameters).length) {
    description = "设置测试参数";
  } else if (
    messages.length &&
    !item.text.includes("未收到") &&
    !item.text.includes("不发送") &&
    !item.text.includes("收不到")
  ) {
    description = "按测试需求发送或等待相关报文";
  }
  const isTimeout = item.text.includes("超时") || item.text.includes("收不到") || item.text.includes("未收到");
  result.push({
    step_id: 3,
    action_id: null,
    action_name: description.slice(0, 40),
    action_type: actionType(description),
    target,
    parameters,
    message: messages.length ? messages.join(",") : null,
    signal: item.text.includes("CP") ? "CP" : item.text.includes("CC") || item.text.includes("CC2") ? "CC" : null,
    duration_ms: isTimeout ? 70000 : null,
    timeout_ms: isTimeout ? 70000 : null,
    required: true,
    description: item.text,
  });
  return result;
}

function assertions(item, target, messages) {
  return (item.gold.expected_results || []).map((expected) => {
    const expectedMessages = unique(expected.match(messagePattern) || []);
    const assertionType = expectedMessages.length || expected.includes("报文") ? "message" : "state";
    const assertionMessages = expectedMessages.length ? expectedMessages : messages;
    return {
      assertion_id: null,
      assertion_type: assertionType,
      description: expected,
      target,
      signal: null,
      message: assertionType === "message" && assertionMessages.length ? assertionMessages.join(",") : null,
      operator: expected.includes("发送") ? "should_send" : expected.includes("进入") ? "should_enter" : "should_equal",
      expected_value: expected,
      timeout_ms: expected.includes("超时") ? 70000 : null,
    };
  });
}

function cleanupSteps(sceneType) {
  const names = sceneType === "DC" ? ["清空消息", "直流高压复位", "直流低压复位"] : ["停止充电", "交流拔枪恢复"];
  return names.map((name, index) => ({
    step_id: index + 1,
    action_id: null,
    action_name: name,
    parameters: {},
    required: true,
  }));
}

for (const item of data) {
  const gold = item.gold;
  if (!gold.is_relevant) {
    item.executable_gold = {
      is_relevant: false,
      reject_reason: "无关输入文本，不生成可执行测试用例",
    };
    continue;
  }

  const messages = messagesFrom(item);
  const target = targetObject(item.text, gold.scene_type);
  let testType = gold.condition_type === "fault" ? "negative" : "positive";
  if (item.category === "robust") testType = `robust_${testType}`;

  item.executable_gold = {
    is_relevant: true,
    case_id: item.id,
    case_name: item.text.replace(/。$/, ""),
    scene_type: gold.scene_type,
    condition_type: gold.condition_type,
    test_type: testType,
    standard_source: item.standard_source,
    test_stage: item.test_stage,
    target_object: target,
    message_types: messages,
    preconditions: preconditions(item, target),
    steps: steps(item, target, messages),
    assertions: assertions(item, target, messages),
    cleanup_steps: cleanupSteps(gold.scene_type),
    parameters: gold.parameters || {},
    fault_type: gold.fault_type,
    raw_requirement: item.text,
    metadata: {
      category: item.category,
      annotation_version: "executable_v1",
    },
  };
}

await fs.writeFile(datasetPath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
console.log(`updated ${data.length} samples`);
