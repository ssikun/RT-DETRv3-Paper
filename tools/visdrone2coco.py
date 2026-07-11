import argparse
import json
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


VISDRONE_CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]


def get_image_size(image_path):
    if Image is None:
        raise RuntimeError("Pillow is required. Install it with `pip install pillow`.")
    with Image.open(image_path) as img:
        return img.width, img.height


def convert_split(image_dir, ann_dir, output_json):
    image_dir = Path(image_dir)
    ann_dir = Path(ann_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    image_paths = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png")))

    for image_id, image_path in enumerate(image_paths, 1):
        width, height = get_image_size(image_path)
        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        txt_path = ann_dir / (image_path.stem + ".txt")
        if not txt_path.exists():
            continue

        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue

                x, y, w, h = map(float, parts[:4])
                score = int(float(parts[4]))
                category_id = int(float(parts[5]))
                truncation = int(float(parts[6]))
                occlusion = int(float(parts[7]))

                if score == 0 or category_id not in range(1, 11):
                    continue
                if w <= 0 or h <= 0:
                    continue

                x = max(0.0, min(x, width - 1.0))
                y = max(0.0, min(y, height - 1.0))
                w = max(0.0, min(w, width - x))
                h = max(0.0, min(h, height - y))
                if w <= 0 or h <= 0:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "truncation": truncation,
                    "occlusion": occlusion,
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": VISDRONE_CATEGORIES,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f)

    print("Saved {} images and {} annotations to {}".format(
        len(images), len(annotations), output_json))


def main():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone2019-DET annotations to COCO format.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_split(args.image_dir, args.ann_dir, args.output)


if __name__ == "__main__":
    main()
