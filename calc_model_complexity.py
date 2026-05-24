from __future__ import annotations

import argparse
import ast
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params


ROOT = Path(__file__).resolve().parent
DEFAULT_SCRIPTS = ("train.py", "train_v5n.py", "train_v11n.py")


def _literal_arg(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def read_training_defaults(script_path: Path) -> dict[str, object]:
    """Read default --model, --imgsz and --name values from a training script."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue

        arg_name = _literal_arg(node.args[0])
        if arg_name not in {"--model", "--imgsz", "--name"}:
            continue

        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[arg_name.lstrip("-")] = _literal_arg(keyword.value)
                break

    missing = {"model", "imgsz"} - defaults.keys()
    if missing:
        raise ValueError(f"{script_path.name} is missing defaults for: {', '.join(sorted(missing))}")

    return defaults


def measure_model(model_cfg: str, imgsz: int | list[int]) -> tuple[float, float]:
    model = YOLO(model_cfg).model
    params_m = get_num_params(model) / 1e6
    flops_g = get_flops(model, imgsz=imgsz)
    return params_m, flops_g


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Params and GFLOPs for models used by training scripts.")
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=list(DEFAULT_SCRIPTS),
        help="Training scripts to inspect. Defaults: train.py train_v5n.py train_v11n.py",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size from the training scripts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for script in args.scripts:
        script_path = Path(script)
        if not script_path.is_absolute():
            script_path = ROOT / script_path

        defaults = read_training_defaults(script_path)
        model_cfg = str(defaults["model"])
        imgsz = args.imgsz if args.imgsz is not None else int(defaults["imgsz"])
        params_m, flops_g = measure_model(model_cfg, imgsz)

        rows.append(
            {
                "script": script_path.name,
                "name": str(defaults.get("name", "-")),
                "model": model_cfg,
                "imgsz": imgsz,
                "params": f"{params_m:.2f}",
                "flops": f"{flops_g:.2f}",
            }
        )

    headers = ("Script", "Name", "Model", "ImgSz", "Params(M)", "GFLOPs")
    keys = ("script", "name", "model", "imgsz", "params", "flops")
    widths = [
        max(len(str(row[key])) for row in rows + [dict(zip(keys, headers))])
        for key in keys
    ]

    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row[key]).ljust(width) for key, width in zip(keys, widths)))

    if any(row["flops"] == "0.00" for row in rows):
        print("\nGFLOPs is 0.00 for at least one model. Install ultralytics-thop if FLOPs are unavailable.")


if __name__ == "__main__":
    main()
