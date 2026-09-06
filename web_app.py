# -*- coding: utf-8 -*-
"""IR-RS-Agent local Web MVP: FastAPI + Qwen multimodal API + local detection tools."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

import agent as core_agent
import config
from tools import execute_tool


BASE_DIR = Path(__file__).resolve().parent
UI_FILE = BASE_DIR / "web" / "ui.html"
UPLOAD_DIR = BASE_DIR / "runtime" / "uploads"
RESULT_DIR = Path(config.RESULTS_DIR)
CONVERSATION_DIR = BASE_DIR / "runtime" / "conversations"
PERSONALITY_DIR = BASE_DIR / "personality_prompts"
PERSONALITY_FILES = {"catgirl": PERSONALITY_DIR / "catgirl.txt"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="IR-RS-Agent MVP", version="1.0.0")
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")

_model = None
_processor = None
_gpu_lock = threading.Lock()
_conversation_lock = threading.Lock()

ROUTING_ADDENDUM = """

## Web 交互额外要求
本系统面向红外影像处理。用户上传的单张图像、伪彩色图像和 B1/B2/B3 多波段图像，均应作为红外或热红外数据理解；不要因为文件是三通道、画面有颜色或经过伪彩映射，就把它描述成普通 RGB 可见光照片。

完整系统的任务范围包括目标检测、图像分割和目标识别。当前必须根据自然语言判断用户实际要完成的任务，而不是在看图前预设类别或默认目标。如果当前已注册工具不能完成用户要求，应明确说明能力边界，不能把分割或识别任务伪装成检测任务。

当前注册的三个工具暂时都是检测工具。若用户明确要求“分割/掩膜/像素区域”，或只要求“识别/分类/判断目标是什么”而不要求定位或画框，不得调用检测工具冒充结果；应选择 Terminate，并用不超过 30 个汉字的 ans 说明相应工具尚未接入。若用户要求“圈出/框出/定位/统计”目标，则属于检测任务，可正常选择检测工具。务必输出完整、合法且闭合的 JSON。

如果消息包含“历史对话”，且用户只是在追问、解释、质疑或讨论上一轮结果，没有要求重新检测、重新分割或重新识别，应选择 Terminate，并在 ans 中直接回答追问。不得把“这个结果可靠吗”“为什么是这个类别”“刚才检测了几个”等追问误当成新的检测任务。

在调用工具之前，先真正观察用户上传的红外图像，并分析用户的任务需求。观察顺序应为：
1. 确认这是红外/热红外或红外多波段影像；
2. 描述成像视角与场景，例如无人机俯视、卫星遥感、地面视角、海面、道路或建筑区域；
3. 描述明暗分布、热目标对比度、目标尺度、密集程度、噪声和清晰度；
4. 分析用户要求属于检测、分割还是识别，以及期望输出是框、掩膜、类别结论、数量统计还是自然语言解释；
5. 最后才选择能够完成任务的工具。

你的首轮输出仍然必须是单个 JSON，但增加以下字段：
{
  "image_description": "明确红外模态，并描述直接可见的场景、视角、热对比、目标尺度、密集程度和清晰度；不猜测精确数量",
  "task_analysis": "判断属于检测/分割/识别中的哪一类，解释目标对象、期望输出，以及哪些信息必须由任务工具确认",
  "thought": "说明选择该工具的理由",
  "actions": [{"name": "工具名", "arguments": {...}}]
}
图像描述属于视觉初步观察，不能把肉眼猜测的目标数量、类别或区域当成工具结果。检测数量以检测工具为准，分割范围以掩膜结果为准，识别类别以识别工具为准。
"""

FINAL_ANSWER_PROMPT = """你是红外遥感多任务分析助手，处理的输入均为红外、热红外或红外多波段影像。请像成熟的 GPT 多模态助手一样，结合图像、用户任务和任务工具的真实输出，给出清晰、连贯、容易理解的中文回答。

必须遵守：
1. 使用四个自然语言段落：前两段段首依次写“图像理解：”“任务分析：”，第四段写“结论与建议：”；第三段按实际任务使用“检测结果：”“分割结果：”或“识别结果：”。
2. 图像理解首先明确红外模态，再描述能直接观察到的场景、视角、热对比、目标尺度、密集程度、清晰度和可能影响处理的成像特点，不凭肉眼编造精确数量或类别。
3. 任务分析应明确这是检测、分割还是识别任务，并说明用户期望得到边界框、掩膜、类别、统计还是解释。
4. 第三个段落的段首应随任务改为“检测结果：”“分割结果：”或“识别结果：”。其中的类别、数量、置信度、区域或标签必须严格以工具结果为准，不得增加或遗漏。
5. 不逐条抄写边界框坐标，不输出 JSON，不输出 Markdown 标题符号，不提及服务器路径。
6. 如果工具没有返回目标或区域，要区分“本次未检出”和“图中一定不存在”；如果结果不确定，要明确建议人工复核。
7. 回答应当呈现可核查的分析依据，但不要暴露模型的隐藏思维链；只提供图像观察、任务解释、工具结果与结论。
8. 置信度是模型评分，不等于真实正确率；即使分数较高，也不能写成“可以确认”或“绝对可靠”，应保留误检、漏检或错分的不确定性。
"""


def get_brain():
    global _model, _processor
    if _model is None:
        _model, _processor = core_agent.load_model()
    return _model, _processor


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def conversation_path(conversation_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", conversation_id or ""):
        raise HTTPException(status_code=400, detail="会话 ID 格式无效。")
    return CONVERSATION_DIR / f"{conversation_id}.json"


def load_conversation(conversation_id: str) -> dict[str, Any]:
    path = conversation_path(conversation_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="找不到该历史会话。")
    with _conversation_lock:
        return json.loads(path.read_text(encoding="utf-8"))


def save_conversation(record: dict[str, Any]) -> None:
    path = conversation_path(str(record["id"]))
    temporary = path.with_suffix(".tmp")
    with _conversation_lock:
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def new_conversation(query: str) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "id": uuid.uuid4().hex,
        "title": query[:28] + ("…" if len(query) > 28 else ""),
        "created_at": timestamp,
        "updated_at": timestamp,
        "last_image_paths": [],
        "messages": [],
    }


def public_conversation(record: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in record.get("messages", []):
        messages.append({
            "id": message.get("id"),
            "role": message.get("role"),
            "content": message.get("content", ""),
            "created_at": message.get("created_at"),
            "image_names": message.get("image_names", []),
            "result": message.get("result"),
        })
    return {
        "id": record["id"],
        "title": record.get("title", "未命名会话"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "messages": messages,
    }


def cached_request_response(request_id: str) -> dict[str, Any] | None:
    if not request_id or not re.fullmatch(r"[0-9a-f]{32}", request_id):
        return None
    for path in CONVERSATION_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            for message in reversed(record.get("messages", [])):
                if message.get("role") == "assistant" and message.get("request_id") == request_id:
                    response = dict(message.get("result") or {})
                    response["answer"] = message.get("content", "")
                    response["conversation_id"] = record["id"]
                    response["replayed_request"] = True
                    return response
        except Exception:
            continue
    return None


def conversation_context(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    lines = ["以下是同一会话的历史对话，请用它理解代词、追问和上一轮工具结果："]
    for message in messages[-8:]:
        role = "用户" if message.get("role") == "user" else "Agent"
        content = str(message.get("content", ""))[:1600]
        lines.append(f"{role}：{content}")
        result = message.get("result") or {}
        if message.get("role") == "assistant" and result:
            lines.append(
                "上一轮结构化结果："
                + json.dumps({
                    "task_type": result.get("task_type"),
                    "tool": result.get("tool"),
                    "count_by_class": result.get("count_by_class"),
                    "result_analysis": result.get("result_analysis"),
                }, ensure_ascii=False)
            )
    return "\n".join(lines)[-7000:]


def latest_detection_result(record: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(record.get("messages", [])):
        result = message.get("result") or {}
        if message.get("role") == "assistant" and (
            result.get("count_by_class") or str(result.get("tool", "")).startswith("ir_detect")
        ):
            return result
    return None


def should_reuse_history(query: str, record: dict[str, Any], has_new_upload: bool) -> bool:
    if has_new_upload or not record.get("messages") or latest_detection_result(record) is None:
        return False
    lowered = query.lower().strip()
    explicit_rerun = (
        r"重新检测|再次检测|再检测|检测一下|请检测|帮我检测|"
        r"重新跑|再跑一次|重新处理|重新分析图|"
        r"换.{0,8}模型|更换.{0,8}模型|调整.{0,8}阈值|阈值.{0,8}(改|调)|"
        r"^\s*检测|^\s*detect|re-?detect|run.{0,12}again|new threshold|different model|"
        r"圈出|框出|重新定位"
    )
    return re.search(explicit_rerun, lowered) is None


FOLLOWUP_PROMPT = """你是红外遥感智能体的会话分析模块。当前用户是在追问同一张图和上一轮工具结果。

规则：
1. 本轮禁止调用或假装调用任何检测、分割、识别工具，只能分析提供的历史结构化结果。
2. 数量、类别、置信度必须以“最近一次权威工具结果”为准，不能根据图片肉眼重新计数，也不能采用更早轮次的冲突数字。
3. 用户询问原因、可靠性、阈值、是否报警或结果含义时，要结合历史结果进行具体分析，而不是简单重复一句话。
4. 如果用户询问的类别不在上一轮检测范围或结果中，不要武断地说不存在；应说明上一轮结果无法回答，并建议用户明确要求重新检测该类别。
5. 直接使用自然、连贯的中文回答，不输出 JSON，不提及服务器路径，不暴露隐藏思维链。
6. 明确说明回答基于上一轮结果，避免让用户误以为本轮又运行了模型。
7. 如果用户是在闲聊、表达情绪或要求文字互动，可以像通用聊天助手一样自然回应，不要生硬地说“我只负责红外检测”，也不要强行重复检测数字；只有问题与图像结果有关时才引用历史结果。
8. 如果用户此前设置了基于检测数量的文字条件（例如超过阈值就提醒、报警或“喵喵叫”），应依据最近一次权威数量判断条件，并自然执行相应的文字回应。
"""


def personality_prompt(name: str) -> str:
    """Load an optional presentation-only prompt; missing files safely mean neutral style."""
    path = PERSONALITY_FILES.get(name)
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def styled_system_prompt(base_prompt: str, personality: str) -> str:
    style = personality_prompt(personality)
    if not style:
        return base_prompt
    return (
        f"{base_prompt}\n\n## 可选表达风格（仅限最终回答的表现层）\n{style}\n\n"
        "再次强调：先保持全部事实、数字和不确定性结论原样，再改变措辞风格。"
    )


def style_fixed_answer(answer: str, personality: str) -> str:
    """Apply a tiny idempotent surface style without rewriting any factual content."""
    if personality != "catgirl" or not personality_prompt(personality):
        return answer
    styled = answer.strip()
    if not styled.startswith("主人"):
        styled = f"主人，{styled}"
    if not re.search(r"喵[～~。！!]*$", styled):
        styled = f"{styled} 喵～"
    return styled


def run_history_reply(
    query: str,
    image_paths: list[str],
    history_text: str,
    authoritative: dict[str, Any],
    personality: str = "none",
) -> dict[str, Any]:
    started = time.time()
    model, processor = get_brain()
    main_image = image_paths[0]
    pil_image = Image.open(main_image).convert("RGB")
    authoritative_context = {
        "task_type": authoritative.get("task_type"),
        "tool": authoritative.get("tool"),
        "model": authoritative.get("model"),
        "count_by_class": authoritative.get("count_by_class"),
        "conf_thr": authoritative.get("conf_thr"),
        "result_analysis": authoritative.get("result_analysis"),
        "reliability_assessment": authoritative.get("reliability_assessment"),
    }
    messages = [
        {"role": "system", "content": [{"type": "text", "text": styled_system_prompt(FOLLOWUP_PROMPT, personality)}]},
        {"role": "user", "content": [
            {"type": "image", "image": main_image},
            {"type": "text", "text": (
                f"历史对话：\n{history_text}\n\n"
                f"最近一次权威工具结果：{json.dumps(authoritative_context, ensure_ascii=False)}\n\n"
                f"用户当前追问：{query}"
            )},
        ]},
    ]
    answer = clean_answer(core_agent.generate(model, processor, messages, pil_image))
    if not answer:
        counts = authoritative.get("count_by_class") or {}
        count_text = "，".join(f"{label} {count} 个" for label, count in counts.items()) or "未返回目标"
        answer = style_fixed_answer(
            f"根据上一轮工具结果：{count_text}。本轮仅分析历史结果，没有重新运行检测模型。",
            personality,
        )
    answer = style_fixed_answer(answer, personality)
    elapsed = round((time.time() - started) * 1000, 1)
    return {
        "answer": answer,
        "tool": "Qwen 历史结果分析",
        "task_type": "连续对话与结果分析",
        "model": config.QWEN_DISPLAY_NAME,
        "detections": [],
        "count_by_class": {},
        "conf_thr": authoritative.get("conf_thr", 0.30),
        "timing_ms": elapsed,
        "total_ms": elapsed,
        "result_image_url": None,
        "image_description": authoritative.get("image_description", "沿用当前会话中的红外影像。"),
        "task_analysis": "当前问题可以直接依据上一轮结构化结果回答，无需重复调用检测工具。",
        "result_analysis": "本轮沿用最近一次权威工具结果，未重新执行检测模型。",
        "reliability_assessment": authoritative.get("reliability_assessment", "可靠性继承上一轮工具结果。"),
        "conversation_reply": True,
        "history_reused": True,
        "personality": personality,
        "trace": {
            "understanding": "识别为同一图像的结果追问",
            "image_description": authoritative.get("image_description", "沿用上一轮影像"),
            "routing_reason": "历史结果足以回答，跳过工具执行",
            "raw_action": "ReuseHistory",
        },
    }


def clean_answer(text: str) -> str:
    text = re.sub(r"\[可视化结果图\]\s*\S+[^\n]*", "", text or "")
    text = re.sub(r"/root/\S+", "", text)
    return text.strip(" \n;")


def partial_json_string(text: str, key: str, fallback: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text or "")
    if not match:
        return fallback
    try:
        return json.loads(f'"{match.group(1)}"')
    except Exception:
        return match.group(1)


def requests_unavailable_task(query: str) -> bool:
    lowered = query.lower()
    if any(word in lowered for word in ("分割", "掩膜", "像素区域", "segment", "mask")):
        return True
    recognition = any(word in lowered for word in ("识别", "分类", "判断是什么", "recognize", "classify", "classification"))
    localization = any(word in lowered for word in ("检测", "圈出", "框出", "定位", "统计", "detect", "locate", "bounding box", "count"))
    return recognition and not localization


def normalize_closed_classes(query: str, classes: Any) -> list[str] | None:
    """把自然语言类别归一化，并将“所有车辆”解释为地面车辆类别组。"""
    lowered_query = query.lower()
    broad_vehicle = any(phrase in lowered_query for phrase in (
        "所有车辆", "全部车辆", "全部车", "all vehicles", "every vehicle",
    ))
    if broad_vehicle:
        return ["car", "bicycle", "cyclist", "bus"]
    if not classes:
        return None
    if isinstance(classes, str):
        classes = [classes]
    aliases = {
        "person": ("person", "people", "人", "人员", "行人"),
        "car": ("car", "cars", "汽车", "车辆", "小车", "轿车"),
        "bicycle": ("bicycle", "bike", "自行车", "单车"),
        "cyclist": ("cyclist", "骑行者", "骑自行车的人"),
        "ship": ("ship", "boat", "船", "船舶", "船只"),
        "bus": ("bus", "公交车", "巴士", "公共汽车", "客车"),
        "drone": ("drone", "无人机", "飞行器"),
        "plane": ("plane", "aircraft", "飞机", "客机", "战机"),
    }
    normalized: list[str] = []
    for raw in classes:
        value = str(raw).strip().lower()
        for canonical, words in aliases.items():
            if value == canonical or any(word.lower() in value for word in words):
                if canonical not in normalized:
                    normalized.append(canonical)
                break
    return normalized or None


def deterministic_answer(result: dict[str, Any]) -> str:
    counts = result.get("count_by_class") or {}
    if not counts or sum(int(v) for v in counts.values()) == 0:
        return ("图像理解：已对上传影像进行视觉分析。\n\n任务分析：任务要求定位并统计指定目标。\n\n"
                "检测结果：当前检测工具未返回符合阈值的目标。\n\n"
                "结论与建议：这表示模型本次未检测到目标，并不等同于目标一定不存在；建议结合原图人工复核。")
    items = "，".join(f"{label} {count} 个" for label, count in counts.items())
    low = [d for d in result.get("detections", []) if d.get("confidence", 0) < result.get("conf_thr", 0.3)]
    suffix = f"其中 {len(low)} 个目标置信度偏低，建议人工复核。" if low else "整体检测结果已通过阈值筛选。"
    return (f"图像理解：已结合上传影像的场景和成像特点完成分析。\n\n"
            f"任务分析：任务要求检测并统计指定类别。\n\n"
            f"检测结果：检测到 {items}。\n\n结论与建议：{suffix}")


def zero_detection_result_image(image_path: str, confidence: float) -> str:
    """零检测时也返回明确的处理结果图，避免把原图误认为框图加载失败。"""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    banner_height = max(34, min(52, image.height // 10))
    draw.rectangle((0, 0, image.width, banner_height), fill=(245, 166, 35))
    draw.text(
        (10, max(8, (banner_height - 14) // 2)),
        f"No requested target detected | conf >= {confidence:.2f}",
        fill=(0, 0, 0),
    )
    output = RESULT_DIR / f"no_detection_{uuid.uuid4().hex}.jpg"
    image.save(output, quality=94)
    return str(output)


def result_context(result: dict[str, Any]) -> dict[str, Any]:
    detections = result.get("detections", [])
    threshold = float(result.get("conf_thr", 0.30))
    confidences = [float(item.get("confidence", 0)) for item in detections]
    per_class: dict[str, list[float]] = {}
    for item in detections:
        per_class.setdefault(str(item.get("label", "object")), []).append(float(item.get("confidence", 0)))
    class_confidence = {
        label: {
            "count": len(values),
            "average": round(sum(values) / len(values), 3),
            "minimum": round(min(values), 3),
            "maximum": round(max(values), 3),
        }
        for label, values in per_class.items()
    }
    return {
        "model": result.get("model", ""),
        "count_by_class": result.get("count_by_class", {}),
        "total": len(detections),
        "confidence_threshold": threshold,
        "overall_confidence": {
            "average": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "minimum": round(min(confidences), 3) if confidences else None,
            "maximum": round(max(confidences), 3) if confidences else None,
        },
        "per_class_confidence": class_confidence,
        "below_threshold_count": sum(value < threshold for value in confidences),
        "inference_ms": result.get("timing_ms", 0),
    }


def infer_task_type(tool_name: str) -> str:
    lowered = tool_name.lower()
    if "segment" in lowered or "mask" in lowered:
        return "红外图像分割"
    if "recogn" in lowered or "classif" in lowered:
        return "红外目标识别"
    return {
        "ir_detect_closed": "无人机/地面红外目标检测",
        "ir_detect_open": "红外开放词汇目标检测",
        "ir_ship_detect_satellite": "卫星三波段舰船检测",
    }.get(tool_name, "红外影像处理")


def requested_task_type(task_analysis: str) -> str:
    if any(word in task_analysis for word in ("分割", "掩膜", "像素")):
        return "红外图像分割（工具待接入）"
    if any(word in task_analysis for word in ("识别", "分类", "类别判断")):
        return "红外目标识别（工具待接入）"
    return "红外影像任务说明"


def band_paths(paths: list[str]) -> dict[str, str]:
    if len(paths) != 3:
        raise ValueError("卫星三波段检测需要同时上传 B1、B2、B3 三张图像。")
    mapped: dict[str, str] = {}
    for path in paths:
        name = Path(path).stem.lower()
        match = re.search(r"(?:^|[_\-])b([123])(?:[_\-]|$)", name)
        if match:
            mapped[f"b{match.group(1)}"] = path
    if len(mapped) == 3:
        return mapped
    return {"b1": paths[0], "b2": paths[1], "b3": paths[2]}


def run_structured(
    query: str,
    image_paths: list[str],
    confidence: float,
    history_text: str = "",
    personality: str = "none",
) -> dict[str, Any]:
    started = time.time()
    model, processor = get_brain()
    main_image = image_paths[0]
    pil_image = Image.open(main_image).convert("RGB")
    bands = band_paths(image_paths) if len(image_paths) == 3 else None
    path_context = (
        f"B1路径: {bands['b1']}\nB2路径: {bands['b2']}\nB3路径: {bands['b3']}"
        if bands else f"图片路径: {main_image}"
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": config.SYSTEM_PROMPT + ROUTING_ADDENDUM}]},
        {"role": "user", "content": [
            {"type": "image", "image": main_image},
            {"type": "text", "text": (
                f"{history_text}\n\n" if history_text else ""
            ) + f"{path_context}\n当前用户指令: {query}"},
        ]},
    ]

    raw = core_agent.generate(model, processor, messages, pil_image)
    obj = core_agent.parse_action(raw)
    if obj is None and requests_unavailable_task(query):
        obj = {
            "image_description": partial_json_string(raw, "image_description", "已读取并观察上传的红外影像。"),
            "task_analysis": partial_json_string(raw, "task_analysis", "该请求属于当前尚未接入专用工具的分割或识别任务。"),
            "thought": "任务类型已识别，但当前没有匹配的专用工具。",
            "actions": [{"name": "Terminate", "arguments": {"ans": "对应的分割或识别工具尚未接入当前 MVP。"}}],
        }
    if obj is None:
        raise RuntimeError(f"Agent 输出无法解析：{raw[:300]}")
    tool_name, args = core_agent.extract_action(obj)
    thought = str(obj.get("thought", "已理解用户指令"))
    image_description = str(obj.get("image_description", "已读取并观察上传图像。"))
    task_analysis = str(obj.get("task_analysis", f"用户希望完成：{query}"))
    if tool_name == "Terminate":
        args = args if isinstance(args, dict) else {}
        boundary = clean_answer(str(args.get("ans", "当前任务对应的专用工具尚未接入。")))
        unavailable = requests_unavailable_task(query)
        if unavailable:
            answer = (f"图像理解：{image_description}\n\n任务分析：{task_analysis}\n\n"
                      f"处理结果：{boundary}\n\n"
                      "结论与建议：Agent 已完成图像理解和任务分类，但不会用不匹配的检测工具冒充分割或识别结果；接入对应工具后即可继续执行。")
            task_type = requested_task_type(task_analysis)
            tool_label = "未调用不匹配工具"
        else:
            answer = boundary
            task_type = "连续对话与结果解释"
            tool_label = "Qwen 会话回答"
        answer = style_fixed_answer(answer, personality)
        return {
            "answer": answer, "tool": tool_label,
            "task_type": task_type, "model": config.QWEN_DISPLAY_NAME, "detections": [],
            "count_by_class": {}, "conf_thr": confidence, "timing_ms": 0,
            "total_ms": round((time.time() - started) * 1000, 1),
            "result_image_url": None,
            "image_description": image_description, "task_analysis": task_analysis,
            "result_analysis": "任务未调用视觉检测工具。", "reliability_assessment": "无检测结果可评估。",
            "conversation_reply": not unavailable,
            "personality": personality,
            "trace": {"understanding": task_analysis, "image_description": image_description, "routing_reason": thought},
        }
    if tool_name not in {"ir_detect_closed", "ir_detect_open", "ir_ship_detect_satellite"}:
        raise RuntimeError(f"Agent 选择了未知工具：{tool_name}")

    args = dict(args or {})
    if tool_name in {"ir_detect_closed", "ir_detect_open"}:
        if len(image_paths) != 1:
            raise RuntimeError("已上传三波段图像，但 Agent 未选择卫星三波段工具。请在指令中明确写出“卫星三波段”。")
        args["image"] = main_image
    if tool_name == "ir_detect_closed":
        args["conf"] = confidence
        args["classes"] = normalize_closed_classes(query, args.get("classes"))
    elif tool_name == "ir_ship_detect_satellite":
        if not bands:
            raise RuntimeError("Agent 选择了卫星三波段工具，请同时上传 B1、B2、B3 三张图像。")
        args.update(bands)
        args["conf"] = confidence

    result = execute_tool(tool_name, args)
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError((result or {}).get("error", "检测工具未返回有效结果"))

    context = result_context(result)
    if context["total"] == 0 and not result.get("result_image"):
        result["result_image"] = zero_detection_result_image(main_image, float(result.get("conf_thr", confidence)))
    count_text = "，".join(f"{label} {count} 个" for label, count in context["count_by_class"].items()) or "0 个目标"
    result_analysis = (
        f"{result.get('model', '检测模型')} 返回 {context['total']} 个检测框：{count_text}；"
        f"置信度范围 {context['overall_confidence']['minimum']} 至 {context['overall_confidence']['maximum']}，"
        f"平均置信度 {context['overall_confidence']['average']}。"
        if context["total"] else f"{result.get('model', '检测模型')} 在当前阈值下未返回检测框。"
    )
    if context["total"] == 0:
        reliability = "当前没有返回检测框，无法评价目标置信度；这只表示本次未检出，仍可能存在漏检。"
    elif context["below_threshold_count"]:
        reliability = f"有 {context['below_threshold_count']} 个结果低于参考阈值，需要人工复核。"
    else:
        reliability = "所有返回结果均达到当前检测阈值，但仍可能存在漏检或类别错误。"
    final_messages = [
        {"role": "system", "content": [{"type": "text", "text": styled_system_prompt(FINAL_ANSWER_PROMPT, personality)}]},
        {"role": "user", "content": [
            {"type": "image", "image": main_image},
            {"type": "text", "text": (
                f"用户原始任务：{query}\n"
                f"历史对话：{history_text if history_text else '无'}\n"
                f"首轮图像观察：{image_description}\n"
                f"任务需求分析：{task_analysis}\n"
                f"工具选择理由：{thought}\n"
                f"检测工具结构化结果：{json.dumps(context, ensure_ascii=False)}\n"
                "请基于以上可核查信息生成最终自然语言回答。"
            )},
        ]},
    ]
    final_raw = core_agent.generate(model, processor, final_messages, pil_image)
    answer = clean_answer(final_raw)
    if context["total"] == 0:
        answer = (
            f"图像理解：{image_description}\n\n"
            f"任务分析：{task_analysis}\n\n"
            f"检测结果：{result.get('model', '检测模型')} 在置信度阈值 "
            f"{float(result.get('conf_thr', confidence)):.2f} 下未检测到用户指定的目标，因此本次没有可绘制的目标框。\n\n"
            "结论与建议：本次“未检出”不代表图中一定不存在目标，也不能使用用户指令中的数量作为检测结果。"
            "建议检查目标类别是否选择正确；如肉眼确认存在目标，可尝试匹配的专用模型或进行人工复核。"
        )
    elif not answer:
        answer = style_fixed_answer(deterministic_answer(result), personality)
    answer = style_fixed_answer(answer, personality)

    result_path = result.get("result_image")
    result_url = f"/results/{Path(result_path).name}" if result_path and Path(result_path).exists() else None
    return {
        "answer": answer,
        "tool": tool_name,
        "task_type": infer_task_type(tool_name),
        "model": result.get("model", ""),
        "detections": result.get("detections", []),
        "count_by_class": result.get("count_by_class", {}),
        "conf_thr": result.get("conf_thr", confidence),
        "timing_ms": result.get("timing_ms", 0),
        "total_ms": round((time.time() - started) * 1000, 1),
        "result_image_url": result_url,
        "image_description": image_description,
        "task_analysis": task_analysis,
        "result_analysis": result_analysis,
        "reliability_assessment": reliability,
        "personality": personality,
        "trace": {
            "understanding": task_analysis,
            "image_description": image_description,
            "routing_reason": thought,
            "raw_action": tool_name,
        },
    }


@app.get("/")
def index():
    if not UI_FILE.exists():
        raise HTTPException(status_code=500, detail=f"UI 文件不存在：{UI_FILE}")
    return FileResponse(UI_FILE, media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "brain_mode": "qwen_api",
        "brain_model": config.QWEN_API_MODEL,
        "gpu_busy": _gpu_lock.locked(),
        "tools": ["ir_detect_closed", "ir_detect_open", "ir_ship_detect_satellite"],
    }


@app.get("/api/conversations")
def list_conversations():
    items = []
    for path in CONVERSATION_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            messages = record.get("messages", [])
            items.append({
                "id": record["id"],
                "title": record.get("title", "未命名会话"),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "message_count": len(messages),
                "preview": str(messages[-1].get("content", ""))[:80] if messages else "",
            })
        except Exception:
            continue
    items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return {"conversations": items}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    return public_conversation(load_conversation(conversation_id))


@app.post("/api/analyze")
def analyze(
    query: str = Form(...),
    confidence: float = Form(0.30),
    conversation_id: str | None = Form(None),
    request_id: str | None = Form(None),
    personality: str = Form("none"),
    files: list[UploadFile] | None = File(None),
):
    query = query.strip()
    if not query or len(query) > 500:
        raise HTTPException(status_code=400, detail="指令不能为空且不能超过 500 个字符。")
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(status_code=400, detail="置信度阈值必须在 0.05 到 0.95 之间。")
    personality = personality.strip().lower()
    if personality not in {"none", *PERSONALITY_FILES.keys()}:
        raise HTTPException(status_code=400, detail="未知的性格模式。")
    request_id = (request_id or uuid.uuid4().hex).replace("-", "").lower()
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise HTTPException(status_code=400, detail="请求 ID 格式无效。")
    cached = cached_request_response(request_id)
    if cached is not None:
        return cached

    uploads = list(files or [])
    if uploads and len(uploads) not in {1, 3}:
        raise HTTPException(status_code=400, detail="每轮可上传 1 张图，或卫星 B1、B2、B3 共 3 张图。")
    record = load_conversation(conversation_id) if conversation_id else new_conversation(query)
    history_text = conversation_context(record.get("messages", []))
    saved: list[str] = []
    total_bytes = 0
    try:
        if uploads:
            job_dir = UPLOAD_DIR / uuid.uuid4().hex
            job_dir.mkdir(parents=True, exist_ok=False)
            for index, upload in enumerate(uploads):
                suffix = Path(upload.filename or "image.png").suffix.lower()
                if suffix not in ALLOWED_SUFFIXES:
                    raise HTTPException(status_code=400, detail=f"不支持的图像格式：{suffix or '未知'}")
                data = upload.file.read()
                total_bytes += len(data)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="上传文件总大小不能超过 50 MB。")
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(upload.filename or f"image_{index}{suffix}").name)
                path = job_dir / f"{index}_{safe_name}"
                path.write_bytes(data)
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"文件不是有效图像：{upload.filename}") from exc
                saved.append(str(path))
        else:
            saved = [path for path in record.get("last_image_paths", []) if Path(path).exists()]
            if not saved:
                raise HTTPException(status_code=400, detail="该会话还没有图像，请先上传红外影像。")

        if not _gpu_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Agent 正在处理另一个任务，请稍后重试。")
        try:
            authoritative = latest_detection_result(record)
            if should_reuse_history(query, record, bool(uploads)) and authoritative is not None:
                response = run_history_reply(query, saved, history_text, authoritative, personality)
            else:
                response = run_structured(query, saved, confidence, history_text, personality)
            response["conversation_id"] = record["id"]
            timestamp = now_iso()
            record["last_image_paths"] = saved
            record["updated_at"] = timestamp
            record.setdefault("messages", []).extend([
                {
                    "id": uuid.uuid4().hex,
                    "role": "user",
                    "content": query,
                    "created_at": timestamp,
                    "image_names": [Path(path).name for path in saved] if uploads else [],
                    "request_id": request_id,
                },
                {
                    "id": uuid.uuid4().hex,
                    "role": "assistant",
                    "content": response.get("answer", ""),
                    "created_at": now_iso(),
                    "image_names": [],
                    "request_id": request_id,
                    "result": {key: value for key, value in response.items() if key not in {"answer", "conversation_id"}},
                },
            ])
            save_conversation(record)
            return response
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _gpu_lock.release()
    finally:
        for upload in uploads:
            upload.file.close()
