#!/usr/bin/env python3

"""Report fine-grained COCO small-object diagnostics for detections."""

import argparse
import csv
import json
import math
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


AREA_BUCKETS = (
    ("tiny_0_8", 0.0, 8.0**2),
    ("tiny_8_16", 8.0**2 + 1e-6, 16.0**2),
    ("small_16_24", 16.0**2 + 1e-6, 24.0**2),
    ("small_24_32", 24.0**2 + 1e-6, 32.0**2),
    ("small_0_32", 0.0, 32.0**2),
)

DENSITY_GROUPS = (
    ("sparse_1_10", 1, 10),
    ("moderate_11_30", 11, 30),
    ("dense_31_plus", 31, None),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report AP, AP50, AP75, and AR100 by fine size bucket, "
            "category, and per-image small-object density."))
    parser.add_argument(
        "--annotations", required=True, help="COCO ground-truth JSON file.")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=JSON",
        help="Named COCO result file. Repeat for multiple models.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for CSV/JSON output.")
    parser.add_argument(
        "--max-dets",
        type=int,
        default=100,
        help="Maximum detections per image (default: 100).")
    return parser.parse_args()


def parse_predictions(values):
    predictions = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--prediction must use NAME=JSON format: {!r}".format(value))
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(
                "--prediction must use non-empty NAME=JSON values")
        if name in names:
            raise ValueError("Duplicate prediction name: {}".format(name))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        names.add(name)
        predictions.append((name, path))
    return predictions


def valid_annotations(coco_gt):
    return [
        annotation for annotation in coco_gt.dataset.get("annotations", [])
        if not annotation.get("iscrowd", 0) and not annotation.get("ignore", 0)
    ]


def in_area(area, minimum, maximum):
    return minimum <= float(area) <= maximum


def create_evaluator(coco_gt, coco_dt, image_ids, area_buckets, max_dets):
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.params.areaRngLbl = [item[0] for item in area_buckets]
    evaluator.params.areaRng = [[item[1], item[2]] for item in area_buckets]
    evaluator.params.maxDets = [1, 10, max_dets]
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def mean_valid(values):
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def iou_index(evaluator, threshold):
    indices = np.flatnonzero(np.isclose(evaluator.params.iouThrs, threshold))
    if indices.size != 1:
        raise ValueError("IoU threshold {} is unavailable".format(threshold))
    return int(indices[0])


def max_det_index(evaluator, max_dets):
    values = list(evaluator.params.maxDets)
    if max_dets not in values:
        raise ValueError("maxDets={} is unavailable".format(max_dets))
    return values.index(max_dets)


def metrics(evaluator, area_idx, max_dets, category_idx=None):
    max_idx = max_det_index(evaluator, max_dets)
    precision = evaluator.eval["precision"][:, :, :, area_idx, max_idx]
    recall = evaluator.eval["recall"][:, :, area_idx, max_idx]

    if category_idx is not None:
        precision = precision[:, :, category_idx:category_idx + 1]
        recall = recall[:, category_idx:category_idx + 1]

    return {
        "ap": mean_valid(precision),
        "ap50": mean_valid(precision[iou_index(evaluator, 0.50)]),
        "ap75": mean_valid(precision[iou_index(evaluator, 0.75)]),
        "ar100": mean_valid(recall),
    }


def count_bucket_annotations(annotations, minimum, maximum, category_id=None):
    return sum(
        1 for annotation in annotations
        if (category_id is None or annotation["category_id"] == category_id)
        and in_area(annotation.get("area", 0.0), minimum, maximum))


def count_bucket_images(annotations, minimum, maximum, category_id=None):
    return len({
        annotation["image_id"] for annotation in annotations
        if (category_id is None or annotation["category_id"] == category_id)
        and in_area(annotation.get("area", 0.0), minimum, maximum)
    })


def density_image_groups(image_ids, annotations):
    small_counts = {image_id: 0 for image_id in image_ids}
    for annotation in annotations:
        if in_area(annotation.get("area", 0.0), 0.0, 32.0**2):
            small_counts[annotation["image_id"]] += 1

    groups = []
    for name, minimum, maximum in DENSITY_GROUPS:
        selected = [
            image_id for image_id in image_ids
            if small_counts[image_id] >= minimum and (
                maximum is None or small_counts[image_id] <= maximum)
        ]
        groups.append((name, minimum, maximum, selected))
    return small_counts, groups


def finite_or_none(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def serializable_rows(rows):
    return [
        {key: finite_or_none(value) for key, value in row.items()}
        for row in rows
    ]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if value is None else value)
                for key, value in row.items()
            })


def metric_text(value):
    if value is None:
        return "nan"
    return "{:.4f}".format(value)


def print_size_table(rows):
    print("\nFine-grained size diagnostics")
    print("model                 bucket          GT     AP   AP50   AP75  AR100")
    for row in rows:
        print("{:<21} {:<14} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
            row["model"], row["bucket"], row["gt_count"],
            metric_text(row["ap"]), metric_text(row["ap50"]),
            metric_text(row["ap75"]), metric_text(row["ar100"])))


def main():
    args = parse_args()
    if not os.path.isfile(args.annotations):
        raise FileNotFoundError(args.annotations)
    if args.max_dets < 10:
        raise ValueError("--max-dets must be at least 10")
    predictions = parse_predictions(args.prediction)
    os.makedirs(args.output_dir, exist_ok=True)

    coco_gt = COCO(args.annotations)
    image_ids = sorted(coco_gt.getImgIds())
    category_ids = sorted(coco_gt.getCatIds())
    category_names = {
        category["id"]: category["name"]
        for category in coco_gt.loadCats(category_ids)
    }
    annotations = valid_annotations(coco_gt)
    small_counts, density_groups = density_image_groups(
        image_ids, annotations)

    size_rows = []
    category_rows = []
    density_rows = []

    for model_name, result_path in predictions:
        print("\nLoading {}: {}".format(model_name, result_path), flush=True)
        coco_dt = coco_gt.loadRes(result_path)
        evaluator = create_evaluator(
            coco_gt, coco_dt, image_ids, AREA_BUCKETS, args.max_dets)

        for area_idx, (bucket, minimum, maximum) in enumerate(AREA_BUCKETS):
            row = {
                "model": model_name,
                "bucket": bucket,
                "min_area": minimum,
                "max_area": maximum,
                "image_count": count_bucket_images(
                    annotations, minimum, maximum),
                "gt_count": count_bucket_annotations(
                    annotations, minimum, maximum),
            }
            row.update(metrics(evaluator, area_idx, args.max_dets))
            size_rows.append(row)

        small_idx = [item[0] for item in AREA_BUCKETS].index("small_0_32")
        for category_idx, category_id in enumerate(category_ids):
            row = {
                "model": model_name,
                "category_id": category_id,
                "category_name": category_names[category_id],
                "image_count": count_bucket_images(
                    annotations, 0.0, 32.0**2, category_id),
                "gt_count": count_bucket_annotations(
                    annotations, 0.0, 32.0**2, category_id),
            }
            row.update(
                metrics(evaluator, small_idx, args.max_dets, category_idx))
            category_rows.append(row)

        small_bucket = (("small_0_32", 0.0, 32.0**2), )
        for group_name, minimum, maximum, group_image_ids in density_groups:
            if not group_image_ids:
                group_metrics = {
                    "ap": float("nan"),
                    "ap50": float("nan"),
                    "ap75": float("nan"),
                    "ar100": float("nan"),
                }
            else:
                group_evaluator = create_evaluator(
                    coco_gt, coco_dt, group_image_ids, small_bucket,
                    args.max_dets)
                group_metrics = metrics(group_evaluator, 0, args.max_dets)
            row = {
                "model": model_name,
                "density_group": group_name,
                "min_small_gt": minimum,
                "max_small_gt": maximum,
                "image_count": len(group_image_ids),
                "small_gt_count": sum(
                    small_counts[image_id] for image_id in group_image_ids),
            }
            row.update(group_metrics)
            density_rows.append(row)

    size_rows = serializable_rows(size_rows)
    category_rows = serializable_rows(category_rows)
    density_rows = serializable_rows(density_rows)

    size_path = os.path.join(args.output_dir, "size_bucket_metrics.csv")
    category_path = os.path.join(args.output_dir,
                                 "class_small_metrics.csv")
    density_path = os.path.join(args.output_dir, "density_metrics.csv")
    summary_path = os.path.join(args.output_dir, "diagnostic_summary.json")

    write_csv(size_path, size_rows, list(size_rows[0]))
    write_csv(category_path, category_rows, list(category_rows[0]))
    write_csv(density_path, density_rows, list(density_rows[0]))

    summary = {
        "annotations": os.path.abspath(args.annotations),
        "predictions": [
            {"name": name, "results": os.path.abspath(path)}
            for name, path in predictions
        ],
        "image_count": len(image_ids),
        "valid_gt_count": len(annotations),
        "images_without_small_gt": sum(
            count == 0 for count in small_counts.values()),
        "max_dets": args.max_dets,
        "area_bucket_definition": [
            {"name": name, "min_area": minimum, "max_area": maximum}
            for name, minimum, maximum in AREA_BUCKETS
        ],
        "density_group_definition": [
            {"name": name, "min_small_gt": minimum, "max_small_gt": maximum}
            for name, minimum, maximum in DENSITY_GROUPS
        ],
        "size_metrics": size_rows,
        "class_small_metrics": category_rows,
        "density_metrics": density_rows,
    }
    with open(summary_path, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=True,
                  allow_nan=False)
        output.write("\n")

    print_size_table(size_rows)
    print("\nSize CSV: {}".format(size_path))
    print("Class CSV: {}".format(category_path))
    print("Density CSV: {}".format(density_path))
    print("Summary JSON: {}".format(summary_path))


if __name__ == "__main__":
    main()
