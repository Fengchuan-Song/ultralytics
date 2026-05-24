from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict images with trained weights and save one txt file per image."
    )
    parser.add_argument("--weights", default='/data/yolov5/weights/yolov5_best.pt', help="Trained weight path, e.g. runs/detect/exp/weights/best.pt")
    parser.add_argument("--source", default='/data_ssd/datasets/WaterScenes', help="Image file, directory, glob, or txt list to predict")
    parser.add_argument("--output", default="/data/yolov5/predict_results", help="Directory used to save txt prediction files")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.35, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0 or cpu")
    parser.add_argument(
        "--model-type",
        choices=("auto", "yolo", "rtdetr"),
        default="auto",
        help="Model loader to use. auto selects RTDETR when the weights path contains 'rtdetr'.",
    )
    parser.add_argument("--score-decimals", type=int, default=6, help="Decimal places for confidence scores")
    parser.add_argument("--box-decimals", type=int, default=2, help="Decimal places for xyxy pixel boxes")
    return parser.parse_args()


def resolve_path(path: str) -> str:
    """Keep globs/URLs untouched, but resolve normal local paths from the repo root."""
    if any(ch in path for ch in "*?") or "://" in path:
        return path

    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(ROOT / path_obj)


def load_model(weights: str, model_type: str):
    if model_type == "auto":
        model_type = "rtdetr" if "rtdetr" in Path(weights).name.lower() else "yolo"
    return RTDETR(weights) if model_type == "rtdetr" else YOLO(weights)


def unique_txt_path(output_dir: Path, image_path: str, used_names: set[str]) -> Path:
    stem = Path(image_path).stem or "image"
    name = f"{stem}.txt"
    if name not in used_names:
        used_names.add(name)
        return output_dir / name

    index = 2
    while True:
        name = f"{stem}_{index}.txt"
        if name not in used_names:
            used_names.add(name)
            return output_dir / name
        index += 1


def write_detection_txt(result, txt_path: Path, score_decimals: int, box_decimals: int) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        txt_path.write_text("", encoding="utf-8")
        return

    classes = boxes.cls.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    xyxy = boxes.xyxy.detach().cpu().tolist()

    lines = []
    score_fmt = f"{{:.{score_decimals}f}}"
    box_fmt = f"{{:.{box_decimals}f}}"
    for class_id, score, (left, top, right, bottom) in zip(classes, scores, xyxy):
        lines.append(
            " ".join(
                (
                    str(int(class_id)),
                    score_fmt.format(float(score)),
                    box_fmt.format(float(left)),
                    box_fmt.format(float(top)),
                    box_fmt.format(float(right)),
                    box_fmt.format(float(bottom)),
                )
            )
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    weights = resolve_path(args.weights)
    source = resolve_path(args.source)
    output_dir = Path(resolve_path(args.output))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(weights, args.model_type)
    predict_kwargs = {
        "source": source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "stream": True,
        "save": False,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    used_names: set[str] = set()
    count = 0
    for result in model.predict(**predict_kwargs):
        txt_path = unique_txt_path(output_dir, result.path, used_names)
        write_detection_txt(result, txt_path, args.score_decimals, args.box_decimals)
        count += 1

    print(f"Saved {count} txt files to {output_dir}")


if __name__ == "__main__":
    main()
