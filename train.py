from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "waterscenes.yaml"


def parse_args():
    parser = ArgumentParser(description="Train YOLOv8 on the WaterScenes dataset.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights or yaml, e.g. yolov8n.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0, 0,1, cpu")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--project", default="runs/detect", help="Save results to project/name")
    parser.add_argument("--name", default="yolov8", help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default='Achelous++', help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", default=None, help="Weights & Biases entity/team name")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"], help="W&B mode")
    return parser.parse_args()


def init_wandb(args):
    if args.no_wandb or args.wandb_mode == "disabled":
        SETTINGS.update({"wandb": False})
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Weights & Biases is not installed. Install it with: pip install wandb") from exc

    SETTINGS.update({"wandb": True})
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.name,
        mode=args.wandb_mode,
        config=vars(args),
    )


def _get_metric(metrics, *keys):
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def log_map_metrics_to_wandb(trainer):
    """Log mAP50, mAP75 and mAP50-95 to W&B at the end of every fit epoch."""
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    metrics = trainer.metrics or {}
    log_data = {}

    map50 = _get_metric(metrics, "metrics/mAP50(B)", "metrics/mAP50")
    map5095 = _get_metric(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95")
    if map50 is not None:
        log_data["mAP50"] = map50
    if map5095 is not None:
        log_data["mAP50-95"] = map5095

    validator_metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    box_metrics = getattr(validator_metrics, "box", None)
    map75 = getattr(box_metrics, "map75", None)
    if map75 is not None:
        log_data["mAP75"] = float(map75)

    if log_data:
        log_data["epoch"] = trainer.epoch + 1
        wandb.log(log_data, step=trainer.epoch + 1, commit=False)


def main():
    args = parse_args()
    init_wandb(args)
    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", log_map_metrics_to_wandb)

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
