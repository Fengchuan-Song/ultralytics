from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "waterscenes.yaml"


def parse_args():
    parser = ArgumentParser(description="Train YOLOv8 on the WaterScenes dataset.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights or yaml, e.g. yolov8n.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--device", type=int, default=0, help="Training device, e.g. 0, 0,1, cpu")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--project", default="runs/detect", help="Save results to project/name")
    parser.add_argument("--name", default="waterscenes_yolov8_320", help="Experiment name")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
