import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm


def load_classes(class_list):
    """Load class names from a txt file, one class per line."""
    with open(class_list, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_image_stems(image_list):
    """Read image list and return image filename stems for matching VOC XML files."""
    with open(image_list, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    stems = []
    for image_path in image_paths:
        normalized = image_path.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        stems.append(Path(normalized).stem)
    return stems


def voc_box_to_yolo(xml_box, image_width, image_height):
    xmin = float(xml_box.findtext("xmin"))
    ymin = float(xml_box.findtext("ymin"))
    xmax = float(xml_box.findtext("xmax"))
    ymax = float(xml_box.findtext("ymax"))

    xmin = max(0.0, min(xmin, image_width))
    xmax = max(0.0, min(xmax, image_width))
    ymin = max(0.0, min(ymin, image_height))
    ymax = max(0.0, min(ymax, image_height))

    box_width = xmax - xmin
    box_height = ymax - ymin
    if box_width <= 0 or box_height <= 0:
        return None

    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    norm_width = box_width / image_width
    norm_height = box_height / image_height

    return x_center, y_center, norm_width, norm_height


def convert_voc(xml_dir, class_list, output_dir, image_list=None):
    """
    Convert VOC XML annotations to YOLO txt labels.

    Output format per line:
        class_id x_center y_center width height
    All coordinates are normalized to [0, 1].
    """
    classes = load_classes(class_list)
    class_to_idx = {name: index for index, name in enumerate(classes)}

    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image_list:
        xml_files = [xml_dir / f"{stem}.xml" for stem in load_image_stems(image_list)]
    else:
        xml_files = sorted(xml_dir.glob("*.xml"))

    missing_xml_count = 0
    converted_count = 0
    skipped_object_count = 0

    for xml_file in tqdm(xml_files, desc="Converting VOC to YOLO"):
        if not xml_file.exists():
            missing_xml_count += 1
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            raise ValueError(f"Missing <size> in {xml_file}")

        image_width = float(size.findtext("width"))
        image_height = float(size.findtext("height"))
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image size in {xml_file}: {image_width}x{image_height}")

        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.findtext("name", "").strip()
            if class_name not in class_to_idx:
                skipped_object_count += 1
                continue

            xml_box = obj.find("bndbox")
            if xml_box is None:
                skipped_object_count += 1
                continue

            yolo_box = voc_box_to_yolo(xml_box, image_width, image_height)
            if yolo_box is None:
                skipped_object_count += 1
                continue

            # class_id = class_to_idx[class_name]
            class_id = 0
            x_center, y_center, norm_width, norm_height = yolo_box
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

        label_file = output_dir / f"{xml_file.stem}.txt"
        if yolo_lines:
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

        converted_count += 1

    print(f"Converted XML files: {converted_count}")
    print(f"Missing XML files: {missing_xml_count}")
    print(f"Skipped objects: {skipped_object_count}")
    print(f"YOLO labels saved to: {output_dir}")


if __name__ == "__main__":
    config = {
        # 可选：只转换列表中的图片；如果想转换 xml_dir 下全部 XML，将 image_list 设为 None。
        # "image_list": "H:/dataset/WaterScenes/test.txt",
        "xml_dir": "H:/dataset/WaterScenes/detection/xml/",
        "class_list": "E:/Research Code/Camera-4D mmWave Fusion/Achelous_v2.0/model_data/waterscenes_benchmark_ship_only copy.txt",
        # YOLO 标注输出目录，每张图片生成一个同名 .txt 文件。
        "output_dir": "H:/dataset/WaterScenes/detection/yolo_shipOnly",
    }

    convert_voc(**config)
