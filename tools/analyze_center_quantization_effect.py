#!/usr/bin/env python3
"""Relate stride-4 center quantization to 8-16 px detection quality."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_INPUT_SIZES = [
    480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800
]
GROUP_NAMES = ('low_error', 'medium_error', 'high_error')
METRICS = ('recall50', 'recall75', 'best_iou')


def parse_args():
    parser = argparse.ArgumentParser(description=(
        'Test whether stride-4 center quantization explains weak 8-16 px '
        'gains using paired baseline and candidate predictions.'))
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--baseline',
                        required=True,
                        metavar='NAME=JSON',
                        help='Baseline prediction name and COCO bbox JSON.')
    parser.add_argument('--candidate',
                        required=True,
                        metavar='NAME=JSON',
                        help='Candidate prediction name and COCO bbox JSON.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stride', type=float, default=4.0)
    parser.add_argument('--input-sizes',
                        type=int,
                        nargs='+',
                        default=DEFAULT_INPUT_SIZES)
    parser.add_argument('--max-dets', type=int, default=100)
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260901)
    return parser.parse_args()


def parse_named_path(value):
    if '=' not in value:
        raise ValueError('Expected NAME=JSON, got {!r}.'.format(value))
    name, path = value.split('=', 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise ValueError('NAME and JSON path must both be non-empty.')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return name, path


def valid_annotations(coco_gt):
    return [
        annotation for annotation in coco_gt.dataset.get('annotations', []) if
        not annotation.get('iscrowd', 0) and not annotation.get('ignore', 0)
    ]


def target_annotations(coco_gt):
    return [
        annotation for annotation in valid_annotations(coco_gt)
        if 8.0**2 < float(annotation.get('area', 0.0)) <= 16.0**2
    ]


def center_quantization_error(annotation, image, input_sizes, stride):
    x, y, width, height = annotation['bbox']
    center_x = x + width * 0.5
    center_y = y + height * 0.5
    errors = []
    for input_size in input_sizes:
        feature_x = center_x * input_size / image['width'] / stride
        feature_y = center_y * input_size / image['height'] / stride
        offset_x = feature_x - math.floor(feature_x)
        offset_y = feature_y - math.floor(feature_y)
        distance = math.sqrt((offset_x - 0.5)**2 + (offset_y - 0.5)**2)
        errors.append(distance / math.sqrt(0.5))
    return float(np.mean(errors)), float(errors[input_sizes.index(640)])


def iou_xywh(first, second):
    first_x2 = first[0] + first[2]
    first_y2 = first[1] + first[3]
    second_x2 = second[0] + second[2]
    second_y2 = second[1] + second[3]
    intersection_width = max(
        0.0,
        min(first_x2, second_x2) - max(first[0], second[0]))
    intersection_height = max(
        0.0,
        min(first_y2, second_y2) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(first[2], 0.0) * max(first[3], 0.0)
    second_area = max(second[2], 0.0) * max(second[3], 0.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def load_top_detections(path, max_dets):
    with open(path, 'r', encoding='utf-8') as source:
        predictions = json.load(source)
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[(int(prediction['image_id']),
                 int(prediction['category_id']))].append(prediction)
    for key in grouped:
        grouped[key] = sorted(grouped[key],
                              key=lambda item: item.get('score', 0.0),
                              reverse=True)[:max_dets]
    return grouped


def best_iou_by_gt(annotations, detections):
    values = {}
    for annotation in annotations:
        candidates = detections.get(
            (int(annotation['image_id']), int(annotation['category_id'])), [])
        values[int(annotation['id'])] = max(
            (iou_xywh(annotation['bbox'], candidate['bbox'])
             for candidate in candidates),
            default=0.0)
    return values


def coco_match_flags(coco_gt, result_path, max_dets):
    coco_dt = coco_gt.loadRes(result_path)
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.maxDets = [1, 10, max_dets]
    evaluator.params.areaRng = [[0.0, 1e10]]
    evaluator.params.areaRngLbl = ['all']
    evaluator.evaluate()

    threshold_indices = {}
    for threshold in (0.50, 0.75):
        indices = np.flatnonzero(
            np.isclose(evaluator.params.iouThrs, threshold))
        if len(indices) != 1:
            raise ValueError(
                'Missing COCO IoU threshold {}.'.format(threshold))
        threshold_indices[threshold] = int(indices[0])

    matches = {0.50: {}, 0.75: {}}
    for result in evaluator.evalImgs:
        if result is None:
            continue
        gt_ids = result['gtIds']
        ignored = np.asarray(result['gtIgnore']).astype(bool)
        gt_matches = np.asarray(result['gtMatches'])
        for gt_index, gt_id in enumerate(gt_ids):
            if ignored[gt_index]:
                continue
            for threshold, threshold_index in threshold_indices.items():
                matches[threshold][int(gt_id)] = bool(
                    gt_matches[threshold_index, gt_index] > 0)
    return matches


def quantization_groups(records):
    errors = np.asarray(
        [record['train_quantization_error'] for record in records],
        dtype=np.float64)
    edges = np.quantile(errors, [1.0 / 3.0, 2.0 / 3.0])
    for record in records:
        group_index = int(
            np.searchsorted(edges,
                            record['train_quantization_error'],
                            side='right'))
        record['quantization_group'] = GROUP_NAMES[min(group_index, 2)]
    return [float(value) for value in edges]


def metric_value(record, model_name, metric):
    return float(record['{}_{}'.format(model_name, metric)])


def summarize(records, baseline_name, candidate_name):
    rows = []
    for group in GROUP_NAMES + ('all', ):
        selected = records if group == 'all' else [
            record for record in records
            if record['quantization_group'] == group
        ]
        if not selected:
            continue
        row = {
            'quantization_group':
            group,
            'gt_count':
            len(selected),
            'image_count':
            len({record['image_id']
                 for record in selected}),
            'quantization_error_mean':
            float(
                np.mean([
                    record['train_quantization_error'] for record in selected
                ])),
            'quantization_error_640_mean':
            float(
                np.mean([
                    record['eval640_quantization_error'] for record in selected
                ])),
        }
        for metric in METRICS:
            baseline = float(
                np.mean([
                    metric_value(record, baseline_name, metric)
                    for record in selected
                ]))
            candidate = float(
                np.mean([
                    metric_value(record, candidate_name, metric)
                    for record in selected
                ]))
            row['{}_{}'.format(baseline_name, metric)] = baseline
            row['{}_{}'.format(candidate_name, metric)] = candidate
            row['delta_{}'.format(metric)] = candidate - baseline
        rows.append(row)
    return rows


def bootstrap_deltas(records, baseline_name, candidate_name, samples, seed):
    if samples <= 0:
        return []
    image_ids = sorted({record['image_id'] for record in records})
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    rng = np.random.default_rng(seed)
    rows = []

    for group in GROUP_NAMES + ('all', ):
        selected = records if group == 'all' else [
            record for record in records
            if record['quantization_group'] == group
        ]
        if not selected:
            continue
        for metric in METRICS:
            counts = np.zeros([len(image_ids)], dtype=np.float64)
            baseline_sums = np.zeros_like(counts)
            candidate_sums = np.zeros_like(counts)
            for record in selected:
                index = image_index[record['image_id']]
                counts[index] += 1.0
                baseline_sums[index] += metric_value(record, baseline_name,
                                                     metric)
                candidate_sums[index] += metric_value(record, candidate_name,
                                                      metric)

            deltas = []
            for _ in range(samples):
                draw = rng.integers(0, len(image_ids), size=len(image_ids))
                denominator = counts[draw].sum()
                if denominator <= 0:
                    continue
                baseline = baseline_sums[draw].sum() / denominator
                candidate = candidate_sums[draw].sum() / denominator
                deltas.append(candidate - baseline)
            delta_array = np.asarray(deltas, dtype=np.float64)
            rows.append({
                'quantization_group':
                group,
                'metric':
                metric,
                'bootstrap_samples':
                int(len(delta_array)),
                'delta_mean':
                float(delta_array.mean()),
                'delta_ci_low':
                float(np.quantile(delta_array, 0.025)),
                'delta_ci_high':
                float(np.quantile(delta_array, 0.975)),
                'probability_delta_gt_zero':
                float(np.mean(delta_array > 0.0)),
            })
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, baseline_name, candidate_name):
    print('\n8-16 px center quantization diagnostic')
    print('group          GT  q_error  base_R50  cand_R50  delta  '
          'base_R75  cand_R75  delta  IoU_delta')
    for row in rows:
        print('{:<12} {:>5} {:>8.4f} {:>9.4f} {:>9.4f} {:>7.4f} '
              '{:>9.4f} {:>9.4f} {:>7.4f} {:>9.4f}'.format(
                  row['quantization_group'], row['gt_count'],
                  row['quantization_error_mean'],
                  row['{}_recall50'.format(baseline_name)],
                  row['{}_recall50'.format(candidate_name)],
                  row['delta_recall50'],
                  row['{}_recall75'.format(baseline_name)],
                  row['{}_recall75'.format(candidate_name)],
                  row['delta_recall75'], row['delta_best_iou']))


def main():
    args = parse_args()
    if args.stride <= 0:
        raise ValueError('--stride must be positive.')
    if not args.input_sizes or any(size <= 0 for size in args.input_sizes):
        raise ValueError('--input-sizes must contain positive values.')
    if 640 not in args.input_sizes:
        raise ValueError(
            '--input-sizes must include 640 for eval diagnostics.')
    if args.max_dets < 1:
        raise ValueError('--max-dets must be positive.')
    if args.bootstrap_samples < 0:
        raise ValueError('--bootstrap-samples cannot be negative.')
    if not os.path.isfile(args.annotations):
        raise FileNotFoundError(args.annotations)

    baseline_name, baseline_path = parse_named_path(args.baseline)
    candidate_name, candidate_path = parse_named_path(args.candidate)
    if baseline_name == candidate_name:
        raise ValueError('Baseline and candidate names must differ.')

    coco_gt = COCO(args.annotations)
    annotations = target_annotations(coco_gt)
    if not annotations:
        raise RuntimeError('No valid 8-16 px annotations were found.')
    images = {
        int(image['id']): image
        for image in coco_gt.dataset.get('images', [])
    }

    model_inputs = {}
    for name, path in ((baseline_name, baseline_path), (candidate_name,
                                                        candidate_path)):
        print('\nEvaluating COCO matches for {}...'.format(name), flush=True)
        matches = coco_match_flags(coco_gt, path, args.max_dets)
        detections = load_top_detections(path, args.max_dets)
        model_inputs[name] = {
            'matches': matches,
            'best_iou': best_iou_by_gt(annotations, detections),
        }

    records = []
    for annotation in annotations:
        annotation_id = int(annotation['id'])
        image_id = int(annotation['image_id'])
        train_error, eval_error = center_quantization_error(
            annotation, images[image_id], args.input_sizes, args.stride)
        row = {
            'annotation_id': annotation_id,
            'image_id': image_id,
            'category_id': int(annotation['category_id']),
            'area': float(annotation['area']),
            'train_quantization_error': train_error,
            'eval640_quantization_error': eval_error,
        }
        for name in (baseline_name, candidate_name):
            row['{}_recall50'.format(name)] = int(
                model_inputs[name]['matches'][0.50].get(annotation_id, False))
            row['{}_recall75'.format(name)] = int(
                model_inputs[name]['matches'][0.75].get(annotation_id, False))
            row['{}_best_iou'.format(name)] = float(
                model_inputs[name]['best_iou'][annotation_id])
        records.append(row)

    quantile_edges = quantization_groups(records)
    summary_rows = summarize(records, baseline_name, candidate_name)
    bootstrap_rows = bootstrap_deltas(records, baseline_name, candidate_name,
                                      args.bootstrap_samples,
                                      args.bootstrap_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    records_path = os.path.join(args.output_dir,
                                'center_quantization_samples.csv')
    summary_path = os.path.join(args.output_dir,
                                'center_quantization_summary.csv')
    bootstrap_path = os.path.join(args.output_dir,
                                  'center_quantization_bootstrap.csv')
    json_path = os.path.join(args.output_dir, 'summary.json')
    write_csv(records_path, records)
    write_csv(summary_path, summary_rows)
    if bootstrap_rows:
        write_csv(bootstrap_path, bootstrap_rows)

    result = {
        'annotations':
        os.path.abspath(args.annotations),
        'baseline': {
            'name': baseline_name,
            'predictions': os.path.abspath(baseline_path),
        },
        'candidate': {
            'name': candidate_name,
            'predictions': os.path.abspath(candidate_path),
        },
        'target_definition':
        '64 < COCO annotation area <= 256',
        'stride':
        args.stride,
        'input_sizes':
        args.input_sizes,
        'quantization_error_definition':
        ('mean normalized distance from continuous center to the center of '
         'the floor(center/stride) cell across training input sizes'),
        'quantization_tertile_edges':
        quantile_edges,
        'max_dets':
        args.max_dets,
        'bootstrap_samples':
        args.bootstrap_samples,
        'bootstrap_seed':
        args.bootstrap_seed,
        'summary':
        summary_rows,
        'bootstrap':
        bootstrap_rows,
    }
    with open(json_path, 'w', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')

    print_summary(summary_rows, baseline_name, candidate_name)
    print('\nQuantization tertile edges: {:.4f}, {:.4f}'.format(
        quantile_edges[0], quantile_edges[1]))
    print('Samples CSV:', records_path)
    print('Summary CSV:', summary_path)
    if bootstrap_rows:
        print('Bootstrap CSV:', bootstrap_path)
    print('Summary JSON:', json_path)


if __name__ == '__main__':
    main()
