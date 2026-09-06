# -*- coding: utf-8 -*-
"""Beginner-friendly local installation and artifact check."""
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    import torch
    import ultralytics
    import fastapi
    import openai

    print("\n===== IR-RS-Agent 环境检查 =====")
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("FastAPI:", fastapi.__version__)
    print("OpenAI SDK:", openai.__version__)
    print("CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("本机显卡:", torch.cuda.get_device_name(0))
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存: {total:.1f} GB")

    files = {
        "无人机红外模型": BASE_DIR / "models" / "uav_ir_multiclass_v1.pt",
        "原始YOLO-World": BASE_DIR / "models" / "yolov8s-worldv2.pt",
        "TISD船舶模型": BASE_DIR / "models" / "tisd_satellite_ship_v1.pt",
        "CLIP文本编码器": BASE_DIR / "weights" / "clip" / "ViT-B-32.pt",
        "网页": BASE_DIR / "web" / "ui.html",
    }
    missing = []
    for name, path in files.items():
        if path.is_file():
            print(f"{name}: OK ({path.stat().st_size / 1024**2:.1f} MB)")
        else:
            print(f"{name}: 缺失")
            missing.append(name)

    if not torch.cuda.is_available():
        print("\n警告：没有检测到CUDA显卡。可把 .env 的 DETECTION_DEVICE 改为 cpu，但速度会较慢。")
    if missing:
        print("\n环境检查失败，缺少：" + "、".join(missing))
        return 1
    print("\n环境与模型文件检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
