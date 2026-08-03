#!/usr/bin/env python3

"""Compare two COCO detection result files by scale and paired bootstrap."""

import argparse
import csv
import json
import os
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


AREA_LABELS = ("all", "small", "large")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report scale-specific AP by IoU and paired image-bootstrap "
            "confidence intervals for two COCO bbox result files."))
    parser.add_argument(
        "--annotations", required=True, help="COCO ground-truth JSON file.")
    parser.add_argument(
        "--baseline", required=True, help="Baseline COCO bbox result JSON.")
    parser.add_argument(
        "--candidate", required=True, help="Candidate COCO bbox result JSON.")
    parser.add_argument(
        "--baseline-name", default="baseline", help="Baseline display name.")
    parser.add_argument(
        "--candidate-name", default="candidate", help="Candidate display name.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for CSV and JSON output.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of paired image bootstrap replicates (default: 1000).")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260803,
        help="Bootstrap random seed (default: 20260803).")
    parser.add_argument(
        "--max-dets",
        type=int,
        default=100,
        help="Maximum detections per image (default: 100).")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Percentile confidence level (default: 0.95).")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print bootstrap progress every N replicates (default: 50).")
    return parser.parse_args()


def validate_args(args):
    for path in (args.annotations, args.baseline, args.candidate):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.max_dets <= 0:
        raise ValueError("--max-dets must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")


def evaluate_predictions(coco_gt, result_file, image_ids, max_dets, name):
    print("Evaluating {}: {}".format(name, result_file), flush=True)
    coco_dt = coco_gt.loadRes(result_file)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.params.maxDets = sorted(set([1, 10, max_dets]))
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def area_index(evaluator, label):
    labels = list(evaluator.params.areaRngLbl)
    if label not in labels:
        raise ValueError(
            "Area label {!r} not found in {}".format(label, labels))
    return labels.index(label)


def max_det_index(evaluator, max_dets):
    values = list(evaluator.params.maxDets)
    if max_dets not in values:
        raise ValueError(
            "maxDets={} not found in {}".format(max_dets, values))
    return values.index(max_dets)


def mean_valid(values):
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def standard_ap(evaluator, area_label, max_dets):
    area_idx = area_index(evaluator, area_label)
    max_det_idx = max_det_index(evaluator, max_dets)
    precision = evaluator.eval["precision"][:, :, :, area_idx, max_det_idx]
    return mean_valid(precision)


def iou_ap_curve(evaluator, area_label, max_dets):
    area_idx = area_index(evaluator, area_label)
    max_det_idx = max_det_index(evaluator, max_dets)
    precision = evaluator.eval["precision"][:, :, :, area_idx, max_det_idx]
    return [
        (float(iou), mean_valid(precision[index]))
        for index, iou in enumerate(evaluator.params.iouThrs)
    ]


def reshape_eval_images(evaluator):
    params = evaluator._paramsEval
    category_count = len(params.catIds) if params.useCats else 1
    area_count = len(params.areaRng)
    image_count = len(params.imgIds)
    expected = category_count * area_count * image_count
    if len(evaluator.evalImgs) != expected:
        raise RuntimeError(
            "Unexpected evalImgs length: got {}, expected {}".format(
                len(evaluator.evalImgs), expected))

    result = []
    for category_index in range(category_count):
        category_areas = []
        for area_idx in range(area_count):
            offset = ((category_index * area_count + area_idx) * image_count)
            category_areas.append(
                evaluator.evalImgs[offset:offset + image_count])
        result.append(category_areas)
    return result


def sampled_ap(eval_images, sampled_indices, area_idx, max_dets, rec_thrs):
    """Reproduce COCO AP accumulation for one area on resampled images."""
    precision_values = []

    for category_areas in eval_images:
        records = [
            category_areas[area_idx][index] for index in sampled_indices
            if category_areas[area_idx][index] is not None
        ]
        if not records:
            continue

        gt_ignore = np.concatenate([record["gtIgnore"] for record in records])
        positive_gt = np.count_nonzero(gt_ignore == 0)
        if positive_gt == 0:
            continue

        dt_scores = np.concatenate([
            record["dtScores"][:max_dets] for record in records
        ])
        dt_matches = np.concatenate([
            record["dtMatches"][:, :max_dets] for record in records
        ], axis=1)
        dt_ignore = np.concatenate([
            record["dtIgnore"][:, :max_dets] for record in records
        ], axis=1)

        order = np.argsort(-dt_scores, kind="mergesort")
        dt_matches = dt_matches[:, order]
        dt_ignore = dt_ignore[:, order]

        true_positive = np.logical_and(dt_matches > 0,
                                       np.logical_not(dt_ignore))
        false_positive = np.logical_and(dt_matches == 0,
                                        np.logical_not(dt_ignore))
        true_positive = np.cumsum(true_positive, axis=1).astype(np.float64)
        false_positive = np.cumsum(false_positive, axis=1).astype(np.float64)

        for iou_idx in range(true_positive.shape[0]):
            tp = true_positive[iou_idx]
            fp = false_positive[iou_idx]
            if tp.size == 0:
                precision_values.extend(np.zeros_like(rec_thrs))
                continue

            recall = tp / positive_gt
            precision = tp / (tp + fp + np.spacing(1))
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            recall_indices = np.searchsorted(recall, rec_thrs, side="left")
            sampled_precision = np.zeros_like(rec_thrs)
            valid = recall_indices < precision.size
            sampled_precision[valid] = precision[recall_indices[valid]]
            precision_values.extend(sampled_precision)

    if not precision_values:
        return float("nan")
    return float(np.mean(precision_values))


def verify_custom_accumulator(evaluator, eval_images, area_labels, max_dets):
    image_indices = np.arange(len(evaluator._paramsEval.imgIds), dtype=np.int64)
    rec_thrs = evaluator.params.recThrs
    for label in area_labels:
        custom = sampled_ap(
            eval_images,
            image_indices,
            area_index(evaluator, label),
            max_dets,
            rec_thrs)
        standard = standard_ap(evaluator, label, max_dets)
        both_nan = np.isnan(custom) and np.isnan(standard)
        if not both_nan and not np.isclose(
                custom, standard, atol=1e-10, rtol=1e-10):
            raise RuntimeError(
                "Custom accumulator mismatch for {}: custom={}, standard={}".
                format(label, custom, standard))
        print("Accumulator check {}: {:.6f}".format(label, custom))


def percentile_interval(values, confidence):
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.nanquantile(values, alpha)),
        float(np.nanquantile(values, 1.0 - alpha)),
    )


def bootstrap_summary(rows, point_metrics, confidence):
    summary = {}
    for label in AREA_LABELS:
        metric_name = "AP" if label == "all" else "AP{}".format(
            label[0].upper())
        baseline = np.asarray(
            [row["baseline_{}".format(label)] for row in rows])
        candidate = np.asarray(
            [row["candidate_{}".format(label)] for row in rows])
        delta = candidate - baseline
        valid_delta = delta[np.isfinite(delta)]
        if valid_delta.size == 0:
            raise RuntimeError(
                "No finite bootstrap deltas for area={}".format(label))
        baseline_ci = percentile_interval(baseline, confidence)
        candidate_ci = percentile_interval(candidate, confidence)
        delta_ci = percentile_interval(delta, confidence)
        summary[metric_name] = {
            "baseline_point": point_metrics["baseline"][label],
            "candidate_point": point_metrics["candidate"][label],
            "delta_point": (point_metrics["candidate"][label] -
                            point_metrics["baseline"][label]),
            "baseline_ci": list(baseline_ci),
            "candidate_ci": list(candidate_ci),
            "delta_ci": list(delta_ci),
            "valid_bootstrap_samples": int(valid_delta.size),
            "probability_delta_gt_zero": float(np.mean(valid_delta > 0)),
        }
    return summary


def write_curve_csv(path, names, evaluators, max_dets):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=["model", "area", "iou", "ap"])
        writer.writeheader()
        for name, evaluator in zip(names, evaluators):
            for label in AREA_LABELS:
                for iou, ap in iou_ap_curve(evaluator, label, max_dets):
                    writer.writerow({
                        "model": name,
                        "area": label,
                        "iou": "{:.2f}".format(iou),
                        "ap": "{:.10f}".format(ap),
                    })


def write_bootstrap_csv(path, rows):
    fieldnames = [
        "replicate",
        "baseline_all",
        "candidate_all",
        "delta_all",
        "baseline_small",
        "candidate_small",
        "delta_small",
        "baseline_large",
        "candidate_large",
        "delta_large",
    ]
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = {"replicate": row["replicate"]}
            for label in AREA_LABELS:
                baseline = row["baseline_{}".format(label)]
                candidate = row["candidate_{}".format(label)]
                output_row["baseline_{}".format(label)] = "{:.10f}".format(
                    baseline)
                output_row["candidate_{}".format(label)] = "{:.10f}".format(
                    candidate)
                output_row["delta_{}".format(label)] = "{:.10f}".format(
                    candidate - baseline)
            writer.writerow(output_row)


def print_iou_table(names, evaluators, area_label, max_dets):
    print("\nAP by IoU for area={}".format(area_label))
    print("IoU    {:>12} {:>12} {:>12}".format(
        names[0], names[1], "delta"))
    baseline_curve = iou_ap_curve(evaluators[0], area_label, max_dets)
    candidate_curve = iou_ap_curve(evaluators[1], area_label, max_dets)
    for (iou, baseline), (_, candidate) in zip(baseline_curve,
                                                candidate_curve):
        print("{:.2f}   {:12.4f} {:12.4f} {:+12.4f}".format(
            iou, baseline, candidate, candidate - baseline))


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    coco_gt = COCO(args.annotations)
    image_ids = sorted(coco_gt.getImgIds())
    names = (args.baseline_name, args.candidate_name)
    evaluators = (
        evaluate_predictions(coco_gt, args.baseline, image_ids, args.max_dets,
                             names[0]),
        evaluate_predictions(coco_gt, args.candidate, image_ids, args.max_dets,
                             names[1]),
    )
    eval_images = tuple(reshape_eval_images(item) for item in evaluators)
    for evaluator, images in zip(evaluators, eval_images):
        verify_custom_accumulator(evaluator, images, AREA_LABELS, args.max_dets)

    point_metrics = {
        "baseline": {
            label: standard_ap(evaluators[0], label, args.max_dets)
            for label in AREA_LABELS
        },
        "candidate": {
            label: standard_ap(evaluators[1], label, args.max_dets)
            for label in AREA_LABELS
        },
    }

    print_iou_table(names, evaluators, "small", args.max_dets)
    print_iou_table(names, evaluators, "large", args.max_dets)

    curve_path = os.path.join(args.output_dir, "scale_iou_metrics.csv")
    write_curve_csv(curve_path, names, evaluators, args.max_dets)

    rng = np.random.default_rng(args.seed)
    image_count = len(image_ids)
    rows = []
    started = time.time()
    area_indices = tuple(
        area_index(evaluators[0], label) for label in AREA_LABELS)
    rec_thrs = evaluators[0].params.recThrs
    print("\nRunning {} paired bootstrap replicates over {} images...".format(
        args.bootstrap_samples, image_count), flush=True)
    for replicate in range(1, args.bootstrap_samples + 1):
        sampled_indices = rng.integers(
            0, image_count, size=image_count, dtype=np.int64)
        row = {"replicate": replicate}
        for label, area_idx in zip(AREA_LABELS, area_indices):
            row["baseline_{}".format(label)] = sampled_ap(
                eval_images[0], sampled_indices, area_idx, args.max_dets,
                rec_thrs)
            row["candidate_{}".format(label)] = sampled_ap(
                eval_images[1], sampled_indices, area_idx, args.max_dets,
                rec_thrs)
        rows.append(row)
        if (args.progress_every and replicate % args.progress_every == 0) or (
                replicate == args.bootstrap_samples):
            elapsed = time.time() - started
            print("Bootstrap {}/{} ({:.1f}s)".format(
                replicate, args.bootstrap_samples, elapsed), flush=True)

    bootstrap_path = os.path.join(args.output_dir,
                                  "paired_bootstrap_samples.csv")
    write_bootstrap_csv(bootstrap_path, rows)
    summary = {
        "annotations": os.path.abspath(args.annotations),
        "baseline": {
            "name": names[0],
            "results": os.path.abspath(args.baseline),
        },
        "candidate": {
            "name": names[1],
            "results": os.path.abspath(args.candidate),
        },
        "image_count": image_count,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "confidence": args.confidence,
        "max_dets": args.max_dets,
        "metrics": bootstrap_summary(rows, point_metrics, args.confidence),
    }
    summary_path = os.path.join(args.output_dir,
                                "paired_bootstrap_summary.json")
    with open(summary_path, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=True)
        output.write("\n")

    print("\nPaired bootstrap summary")
    for metric_name, values in summary["metrics"].items():
        print(("{}: baseline={:.4f}, candidate={:.4f}, delta={:+.4f}, "
               "delta CI=[{:+.4f}, {:+.4f}], P(delta>0)={:.3f}").format(
                   metric_name,
                   values["baseline_point"],
                   values["candidate_point"],
                   values["delta_point"],
                   values["delta_ci"][0],
                   values["delta_ci"][1],
                   values["probability_delta_gt_zero"]))
    print("\nCurve CSV: {}".format(curve_path))
    print("Bootstrap CSV: {}".format(bootstrap_path))
    print("Summary JSON: {}".format(summary_path))


if __name__ == "__main__":
    main()
