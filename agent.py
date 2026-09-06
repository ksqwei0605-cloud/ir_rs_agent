# -*- coding: utf-8 -*-
"""IR-RS-Agent：Qwen 多模态 API 认知核心 + 本地视觉工具。"""
import base64
import io
import json
import sys
from PIL import Image
from openai import OpenAI

import config
from tools import execute_tool


def load_model():
    """Create a lightweight API client; no local Qwen weights are loaded."""
    if not config.QWEN_API_KEY:
        raise RuntimeError("尚未配置 DASHSCOPE_API_KEY，请先填写项目根目录中的 .env 文件。")
    print(f"[agent] 连接 Qwen 多模态 API：{config.QWEN_API_MODEL}")
    client = OpenAI(
        api_key=config.QWEN_API_KEY,
        base_url=config.QWEN_API_BASE_URL,
        timeout=config.QWEN_API_TIMEOUT,
        max_retries=2,
    )
    return client, None


def _image_data_url(path: str) -> str:
    """Convert local/TIFF input into a compact JPEG Base64 Data URL for the API."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((config.QWEN_IMAGE_MAX_EDGE, config.QWEN_IMAGE_MAX_EDGE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _api_messages(messages):
    converted = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        blocks = []
        for block in content or []:
            block_type = block.get("type")
            if block_type == "text":
                blocks.append({"type": "text", "text": str(block.get("text", ""))})
            elif block_type == "image":
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(str(block.get("image")))},
                })
        if role == "system":
            system_text = "\n".join(item["text"] for item in blocks if item["type"] == "text")
            converted.append({"role": role, "content": system_text})
        else:
            converted.append({"role": role, "content": blocks})
    return converted


def generate(model, processor, messages, pil_image):
    """Keep the former local-model signature so the Agent/Web layers stay unchanged."""
    response = model.chat.completions.create(
        model=config.QWEN_API_MODEL,
        messages=_api_messages(messages),
        max_tokens=config.QWEN_MAX_TOKENS,
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def parse_action(text):
    """从模型输出中稳健地提取 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        seg = text[start:end + 1]
        for candidate in (seg, seg.replace("\n", " ").replace("\r", " ")):
            try:
                return json.loads(candidate)
            except Exception:
                continue
    for line in text.splitlines():
        line = line.strip()
        s, e = line.find("{"), line.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(line[s:e + 1])
            except Exception:
                continue
    return None


def extract_action(obj):
    if not isinstance(obj, dict):
        return None, None
    actions = obj.get("actions") or []
    if not actions:
        return None, None
    act = actions[0]
    return act.get("name"), act.get("arguments", {})


def format_detection(result, conf_thr=None):
    """把检测结果转成可读文本 + 结果自检。"""
    if not isinstance(result, dict) or "error" in result:
        return f"[错误] {result.get('error') if isinstance(result, dict) else result}"
    dets = result.get("detections", [])
    model_name = result.get("model", "")
    if conf_thr is None:
        conf_thr = result.get("conf_thr", 0.30)  # 用工具实际标定阈值
    if not dets:
        return f"模型 {model_name} 未检测到任何目标。"
    parts = [f"模型 {model_name} 检测到 {len(dets)} 个目标"]
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        parts.append(f"({x1},{y1},{x2},{y2}) {d['label']} {d['confidence']:.2f}")
    low = [d for d in dets if d["confidence"] < conf_thr]
    if low:
        parts.append(f"⚠ {len(low)} 个目标置信度偏低，结果可能不可信")
    if result.get("result_image"):
        parts.append(f"[可视化结果图] {result['result_image']}（绿色框=正常，红色框=低置信度需复核）")
    return "; ".join(parts)


def run(query, image_path, model=None, processor=None):
    if model is None:
        model, processor = load_model()
    pil_image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": config.SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"图片路径: {image_path}\n用户指令: {query}"},
        ]},
    ]

    for r in range(config.MAX_ROUNDS):
        raw = generate(model, processor, messages, pil_image)
        print(f"\n----- 第 {r + 1} 轮模型输出 -----\n{raw}")
        obj = parse_action(raw)
        if obj is None:
            return f"[无法解析] {raw}"
        name, args = extract_action(obj)

        if name == "Terminate":
            return args.get("ans", "") if isinstance(args, dict) else str(args)
        if name is None:
            return obj.get("thought", raw)

        result = execute_tool(name, args)
        formatted = format_detection(result)
        print(f"----- 工具 {name} 结果 -----\n{formatted}")

        messages.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})
        messages.append({"role": "user", "content": [{"type": "text", "text":
            f"[工具执行结果]\n{formatted}\n\n请基于以上结果，用 Terminate 动作给出最终回答。"}]})

    raw = generate(model, processor, messages, pil_image)
    obj = parse_action(raw)
    if obj:
        name, args = extract_action(obj)
        if name == "Terminate" and isinstance(args, dict):
            return args.get("ans", raw)
        if name is None:
            return obj.get("thought", raw)
    return raw


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python agent.py <图像路径> [用户指令]")
        sys.exit(1)
    image_path = sys.argv[1]
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "检测图中的目标"
    model, processor = load_model()
    ans = run(query, image_path, model, processor)
    print("\n===== 最终回答 =====\n" + ans)
