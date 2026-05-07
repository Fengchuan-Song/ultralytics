import os

import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm


def load_list(list_dir):
    with open(list_dir, 'r', encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_classes(class_list):
    """Load class names from a txt file, one class per line."""
    with open(class_list, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def filter(xml_dir, class_list, output_file, image_list=None):
    image_list = load_list(image_list)
    classes = load_classes(class_list)
    print(classes)

    xml_dir = Path(xml_dir)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w", encoding="utf-8") as f:
        for image_dir in tqdm(image_list):
            file_name = image_dir.split('/')[-1].split('.')[0]
            xml_file_path = os.path.join(xml_dir, file_name + '.xml')

            tree = ET.parse(xml_file_path)

            root = tree.getroot()

            target_num = 0
            for obj in root.findall("object"):
                class_name = obj.findtext("name", "").strip()
                if class_name not in classes:
                    continue

                target_num += 1

            if target_num > 0:
                f.write('.' + image_dir + '\n')


if __name__ == "__main__":
    config = {
        # 可选：只转换列表中的图片；如果想转换 xml_dir 下全部 XML，将 image_list 设为 None。
        "image_list": "H:/dataset/WaterScenes/val.txt",
        "xml_dir": "H:/dataset/WaterScenes/detection/xml/",
        "class_list": "E:/Research Code/Camera-4D mmWave Fusion/Achelous_v2.0/model_data/waterscenes_benchmark_ship_only copy.txt",
        # YOLO 标注输出目录，每张图片生成一个同名 .txt 文件。
        "output_file": "H:/dataset/WaterScenes/MIPC_Yolo/val.txt",
    }

    filter(**config)
