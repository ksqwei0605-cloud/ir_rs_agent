# -*- coding: utf-8 -*-
"""三个检测工具 + 工具注册表（MVP）"""
import os
import time
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import config

# ===== 工具 Schema（OpenAI function-calling 格式，供 Agent 认知） =====
TOOL_SCHEMAS = {
    "ir_detect_closed": {
        "name": "ir_detect_closed",
        "description": "检测无人机/地面红外图中的 8 类目标：person/car/bicycle/cyclist/ship/bus/drone/plane。仅接受红外灰度图。",
        "parameters": ["image", "classes", "conf"],
    },
    "ir_detect_open": {
        "name": "ir_detect_open",
        "description": "开放词汇检测，检测任意文字类别，也适用于 RGB 图。",
        "parameters": ["image", "classes"],
    },
    "ir_ship_detect_satellite": {
        "name": "ir_ship_detect_satellite",
        "description": "检测卫星三波段热红外海面舰船（ship 单类）。",
        "parameters": ["b1", "b2", "b3"],
    },
}


def _load(im):
    """图像路径或 ndarray -> ndarray（灰度/彩色按需）"""
    if isinstance(im, str):
        return np.array(Image.open(im))
    return np.asarray(im)


def _draw_and_save(base_img, detections, tool_name, conf_thr):
    """可视化：左图框上标序号（按置信度降序）+ 左上角统计 + 右侧图例面板。

    布局：
    - 左上角：黑底白字 "Detected: N | conf>=阈值"
    - 左图：检测框上标序号，低置信度红框、正常绿框
    - 右侧：白色图例面板，每行 "序号 类别 (x1,y1,x2,y2) conf"
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    if base_img.ndim == 2:
        img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        img = base_img[:, :, ::-1].copy()  # RGB -> BGR

    # 按置信度降序排序（序号 1 = 最可信）
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    H, W = img.shape[:2]

    # 1. 左上角统计条（金色底黑字）
    title = f"Detected: {len(dets)}  |  conf >= {conf_thr:.2f}"
    cv2.rectangle(img, (0, 0), (W, 36), (0, 215, 255), -1)  # BGR 金色
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # 2. 画框 + 序号（序号用带背景的小方块，颜色与框一致）
    for idx, d in enumerate(dets, 1):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        conf = d["confidence"]
        color = (0, 0, 255) if conf < conf_thr else (0, 255, 0)  # BGR：红=低置信度，绿=正常
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 序号方块（框左上角上方，避开顶部统计条）
        tag_y = max(40, y1 - 20)
        (tw, th), _ = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, tag_y - th - 4), (x1 + tw + 8, tag_y + 4), color, -1)
        cv2.putText(img, str(idx), (x1 + 4, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. 右侧图例面板（白色背景）
    panel_w = 340
    panel = np.full((H, panel_w, 3), 255, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, H - 1), (0, 0, 0), 1)  # 边框
    cv2.putText(panel, "No.  Class  BBox (x1,y1,x2,y2)  conf",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(panel, (0, 36), (panel_w, 36), (0, 0, 0), 1)
    line_h = 26
    for idx, d in enumerate(dets, 1):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        conf = d["confidence"]
        color = (0, 0, 200) if conf < conf_thr else (0, 0, 0)  # 低置信度红字
        row = f"{idx:<3} {d['label']:<10} ({x1},{y1},{x2},{y2})  {conf:.2f}"
        y = 36 + idx * line_h
        if y < H - 10:
            cv2.putText(panel, row, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 4. 拼接左图 + 右侧图例
    combined = np.hstack([img, panel])
    out_path = os.path.join(config.RESULTS_DIR, f"{tool_name}_{int(time.time() * 1000)}.jpg")
    cv2.imwrite(out_path, combined)
    return out_path


# ===== 工具A：无人机红外 8 类闭集检测 =====
def detect_uav_ir(image, classes=None, conf=None):
    t0 = time.time()
    base = _load(image)
    model = YOLO(config.UAV_WEIGHT)
    model.set_classes(config.UAV_CLASSES)
    res = model.predict(image, imgsz=1024, conf=conf or config.UAV_CONF, device=config.DETECTION_DEVICE, verbose=False)[0]
    detections = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            label = config.UAV_CLASSES[int(cls[i])]
            if classes and label not in classes:
                continue
            detections.append({
                "label": label,
                "bbox": [float(x) for x in xyxy[i]],
                "confidence": float(cf[i]),
            })
    count = {}
    for d in detections:
        count[d["label"]] = count.get(d["label"], 0) + 1
    _conf = conf or config.UAV_CONF
    result = {
        "model": "uav_ir_multiclass_v1",
        "detections": detections,
        "count_by_class": count,
        "conf_thr": _conf,
        "timing_ms": round((time.time() - t0) * 1000, 1),
    }
    if detections:
        result["result_image"] = _draw_and_save(base, detections, "uav_ir", _conf)
    return result


# ===== 工具B：开放词汇 / RGB 检测（原始 YOLO-World） =====
def detect_open_vocab(image, text_classes):
    t0 = time.time()
    base = _load(image)
    model = YOLO(config.WORLD_WEIGHT)
    model.set_classes(list(text_classes))
    res = model.predict(image, imgsz=1024, device=config.DETECTION_DEVICE, verbose=False)[0]
    detections = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            label = list(text_classes)[int(cls[i])] if int(cls[i]) < len(text_classes) else "object"
            detections.append({
                "label": label,
                "bbox": [float(x) for x in xyxy[i]],
                "confidence": float(cf[i]),
            })
    count = {}
    for d in detections:
        count[d["label"]] = count.get(d["label"], 0) + 1
    result = {
        "model": "yolov8s-worldv2",
        "detections": detections,
        "count_by_class": count,
        "conf_thr": 0.30,
        "timing_ms": round((time.time() - t0) * 1000, 1),
    }
    if detections:
        result["result_image"] = _draw_and_save(base, detections, "open_vocab", 0.30)
    return result


# ===== 工具C：卫星三波段热红外舰船检测 =====
def detect_tisd_ship(b1, b2, b3, conf=None):
    a1, a2, a3 = _load(b1), _load(b2), _load(b3)
    # 转灰度单通道，按 [B3, B2, B1] 顺序堆叠（训练时定死）
    g = lambda a: np.asarray(Image.fromarray(a).convert("L")) if a.ndim == 3 else a
    img = np.stack([g(a3), g(a2), g(a1)], axis=-1)

    t0 = time.time()
    model = YOLO(config.TISD_WEIGHT)
    res = model.predict(img, imgsz=1024, conf=conf or config.TISD_CONF, device=config.DETECTION_DEVICE, verbose=False)[0]
    detections = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            detections.append({"label": "ship", "bbox": [float(x) for x in xyxy[i]], "confidence": float(cf[i])})
    _conf = conf or config.TISD_CONF
    result = {
        "model": "tisd_satellite_ship_v1",
        "detections": detections,
        "count_by_class": {"ship": len(detections)},
        "conf_thr": _conf,
        "timing_ms": round((time.time() - t0) * 1000, 1),
    }
    if detections:
        result["result_image"] = _draw_and_save(img, detections, "tisd_ship", _conf)
    return result


# ===== 工具执行入口 =====
def execute_tool(name, args):
    try:
        if name == "ir_detect_closed":
            image = args.get("image")
            if not image:
                return {"error": "缺少 image 参数"}
            return detect_uav_ir(image, classes=args.get("classes"), conf=args.get("conf"))
        elif name == "ir_detect_open":
            image = args.get("image")
            classes = args.get("classes")
            if not image or not classes:
                return {"error": "缺少 image 或 classes 参数"}
            return detect_open_vocab(image, classes)
        elif name == "ir_ship_detect_satellite":
            b1, b2, b3 = args.get("b1"), args.get("b2"), args.get("b3")
            if not (b1 and b2 and b3):
                return {"error": "缺少 b1/b2/b3 三波段路径"}
            return detect_tisd_ship(b1, b2, b3)
        else:
            return {"error": f"未知工具 {name}"}
    except Exception as e:
        return {"error": f"工具 {name} 执行失败: {e}"}
