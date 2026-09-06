# -*- coding: utf-8 -*-
"""Validate local configuration, start FastAPI, and open the browser."""
import os
import sys
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
load_dotenv(BASE_DIR / ".env", override=False)
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "runtime" / "ultralytics"))


def fail(message: str) -> None:
    print(f"\n[无法启动] {message}\n")
    raise SystemExit(1)


def main() -> None:
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not key:
        fail("请用记事本打开 .env，把百炼 API Key 填到 DASHSCOPE_API_KEY= 后面。")

    device = os.getenv("DETECTION_DEVICE", "0").strip().lower()
    if device != "cpu":
        import torch
        if not torch.cuda.is_available():
            print("[提示] 当前还没有CUDA版PyTorch，本次自动使用CPU检测。")
            print("       如需启用RTX 4060加速，请双击“安装本地环境.bat”完成显卡环境安装。")
            os.environ["DETECTION_DEVICE"] = "cpu"

    required = [
        BASE_DIR / "models" / "uav_ir_multiclass_v1.pt",
        BASE_DIR / "models" / "yolov8s-worldv2.pt",
        BASE_DIR / "models" / "tisd_satellite_ship_v1.pt",
        BASE_DIR / "weights" / "clip" / "ViT-B-32.pt",
        BASE_DIR / "web" / "ui.html",
    ]
    missing = [str(path.relative_to(BASE_DIR)) for path in required if not path.is_file()]
    if missing:
        fail("缺少必要文件：" + "、".join(missing))

    host = os.getenv("LOCAL_HOST", "127.0.0.1")
    port = int(os.getenv("LOCAL_PORT", "7860"))
    url = f"http://{host}:{port}"
    print(f"IR-RS-Agent 正在启动：{url}")
    print("Qwen 通过 API 调用；三个检测模型在本机显卡运行。")
    print("请保持此窗口开启，按 Ctrl+C 停止网站。")
    threading.Timer(1.8, lambda: webbrowser.open(url)).start()
    uvicorn.run("web_app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
