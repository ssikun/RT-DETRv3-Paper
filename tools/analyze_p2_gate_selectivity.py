#!/usr/bin/env python3
"""Measure whether the learned P2 enhancement gate is scale selective."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import math
import os
import sys
from collections import defaultdict

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import paddle

from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import set_random_seed
from ppdet.utils.checkpoint import load_pretrain_weight
from ppdet.utils.cli import ArgsParser

SIZE_BUCKETS = (
    ('tiny_0_8', 0., 8.**2),
    ('tiny_8_16', 8.**2, 16.**2),
    ('small_16_24', 16.**2, 24.**2),
    ('small_24_32', 24.**2, 32.**2),
    ('medium_32_96', 32.**2, 96.**2),
    ('large_96_plus', 96.**2, float('inf')),
)
BUCKET_ORDER = [item[0] for item in SIZE_BUCKETS] + ['background']
SAMPLE_FIELDS = (
    'image_index',
    'image_id',
    'region',
    'object_index',
    'original_width',
    'original_height',
    'original_area',
    'cell_count',
    'gate_mean',
    'gate_max',
    'gate_top25_mean',
    'residual_abs_mean',
    'p3_abs_mean',
    'relative_residual_mean',
    'relative_residual_top25',
)
SUMMARY_FIELDS = (
    'region',
    'sample_count',
    'cell_count',
    'gate_mean',
    'gate_mean_std',
    'gate_max',
    'gate_top25_mean',
    'residual_abs_mean',
    'p3_abs_mean',
    'relative_residual_mean',
    'relative_residual_mean_std',
    'relative_residual_top25',
)


class GateSelectivityArgsParser(ArgsParser):

    def __init__(self):
        super().__init__()
        self.add_argument('--weights',
                          required=True,
                          help='P2-center checkpoint path.')
        self.add_argument('--output-dir',
                          default='results/p2_gate_selectivity',
                          help='Directory for CSV and JSON outputs.')
        self.add_argument('--worker-num', type=int, default=0)
        self.add_argument('--seed', type=int, default=2026)
        self.add_argument('--device', choices=['gpu', 'cpu'], default='cpu')
        self.add_argument(
            '--max-images',
            type=int,
            default=0,
            help='Maximum images to scan; 0 scans the full validation set.')
        self.add_argument('--progress-every', type=int, default=50)


def _numpy(value):
    if isinstance(value, paddle.Tensor):
        value = value.numpy()
    return np.asarray(value)


def _single_image_array(value):
    array = _numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


def _image_id(data, fallback):
    if 'im_id' not in data:
        return fallback
    value = _single_image_array(data['im_id']).reshape([-1])
    return int(value[0]) if len(value) else fallback


def _boxes_and_valid_mask(data):
    boxes = _single_image_array(data['gt_bbox']).astype(np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape([-1, 4])
    valid_mask = np.ones([len(boxes)], dtype=bool)
    if 'is_crowd' in data:
        crowd = _single_image_array(data['is_crowd']).reshape([-1])
        valid_mask = ~crowd.astype(bool)
    return boxes, valid_mask


def _size_bucket(area):
    for index, (name, minimum, maximum) in enumerate(SIZE_BUCKETS):
        lower_ok = area >= minimum if index == 0 else area > minimum
        if lower_ok and area <= maximum:
            return name
    return None


def _top_fraction_indices(values, fraction=0.25):
    values = np.asarray(values).reshape([-1])
    count = max(1, int(math.ceil(len(values) * fraction)))
    if count >= len(values):
        return np.arange(len(values), dtype=np.int64)
    return np.argpartition(values, len(values) - count)[-count:]


def _region_stats(gate_map, residual_map, p3_map, mask):
    gate_values = gate_map[mask]
    residual_values = residual_map[mask]
    p3_values = p3_map[mask]
    if len(gate_values) == 0:
        return None

    gate_top = _top_fraction_indices(gate_values)
    residual_top = _top_fraction_indices(residual_values)
    residual_mean = float(residual_values.mean())
    p3_mean = float(p3_values.mean())
    top_residual_mean = float(residual_values[residual_top].mean())
    top_p3_mean = float(p3_values[residual_top].mean())
    epsilon = 1e-8
    return {
        'cell_count': int(len(gate_values)),
        'gate_mean': float(gate_values.mean()),
        'gate_max': float(gate_values.max()),
        'gate_top25_mean': float(gate_values[gate_top].mean()),
        'residual_abs_mean': residual_mean,
        'p3_abs_mean': p3_mean,
        'relative_residual_mean': residual_mean / max(p3_mean, epsilon),
        'relative_residual_top25':
        top_residual_mean / max(top_p3_mean, epsilon),
    }


def _box_cell_mask(box, x_centers, y_centers):
    x1, y1, x2, y2 = box
    x_inside = np.logical_and(x_centers >= x1, x_centers <= x2)
    y_inside = np.logical_and(y_centers >= y1, y_centers <= y2)
    mask = np.logical_and(y_inside[:, None], x_inside[None, :])
    if mask.any():
        return mask

    center_x = np.clip((x1 + x2) * 0.5, x_centers[0], x_centers[-1])
    center_y = np.clip((y1 + y2) * 0.5, y_centers[0], y_centers[-1])
    x_index = int(np.abs(x_centers - center_x).argmin())
    y_index = int(np.abs(y_centers - center_y).argmin())
    mask[y_index, x_index] = True
    return mask


class _EnhancerCapture(object):

    def __init__(self, enhancer):
        self.gate = None
        self.record = None
        self._handles = [
            enhancer.spatial_gate.register_forward_post_hook(
                self._capture_gate),
            enhancer.register_forward_post_hook(self._capture_enhancer),
        ]

    def reset(self):
        self.gate = None
        self.record = None

    def close(self):
        for handle in self._handles:
            handle.remove()

    def _capture_gate(self, layer, inputs, output):
        del layer, inputs
        self.gate = output.detach().numpy()

    def _capture_enhancer(self, layer, inputs, output):
        del layer
        if self.gate is None:
            raise RuntimeError('The spatial-gate hook did not run.')
        if len(inputs) != 2:
            raise RuntimeError('Expected P2 and P3 enhancer inputs.')
        p3_input = inputs[1]
        residual = paddle.abs(output - p3_input).mean(axis=1)
        p3_abs = paddle.abs(p3_input).mean(axis=1)
        gate = self.gate
        if gate.ndim != 4 or gate.shape[1] != 1:
            raise RuntimeError('Expected a single-channel spatial gate.')
        self.record = {
            'gate': gate[:, 0],
            'residual': residual.detach().numpy(),
            'p3_abs': p3_abs.detach().numpy(),
        }


def _summarize(samples):
    grouped = defaultdict(list)
    for row in samples:
        grouped[row['region']].append(row)

    rows = []
    for region in BUCKET_ORDER:
        region_samples = grouped.get(region, [])
        if not region_samples:
            continue

        def mean(field):
            return float(np.mean([sample[field] for sample in region_samples]))

        rows.append({
            'region':
            region,
            'sample_count':
            len(region_samples),
            'cell_count':
            int(sum(sample['cell_count'] for sample in region_samples)),
            'gate_mean':
            mean('gate_mean'),
            'gate_mean_std':
            float(np.std([sample['gate_mean'] for sample in region_samples])),
            'gate_max':
            mean('gate_max'),
            'gate_top25_mean':
            mean('gate_top25_mean'),
            'residual_abs_mean':
            mean('residual_abs_mean'),
            'p3_abs_mean':
            mean('p3_abs_mean'),
            'relative_residual_mean':
            mean('relative_residual_mean'),
            'relative_residual_mean_std':
            float(
                np.std([
                    sample['relative_residual_mean']
                    for sample in region_samples
                ])),
            'relative_residual_top25':
            mean('relative_residual_top25'),
        })
    return rows


def _write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows):
    print('\nP2 gate scale selectivity')
    print('region             samples gate_mean gate_max gate_top25 '
          'residual_ratio residual_top25')
    for row in rows:
        print('{:<18} {:>7} {:>9.4f} {:>8.4f} {:>10.4f} {:>14.4f} '
              '{:>14.4f}'.format(row['region'], row['sample_count'],
                                 row['gate_mean'], row['gate_max'],
                                 row['gate_top25_mean'],
                                 row['relative_residual_mean'],
                                 row['relative_residual_top25']))


def main():
    args = GateSelectivityArgsParser().parse_args()
    if args.max_images < 0:
        raise ValueError('--max-images cannot be negative.')
    if args.progress_every < 0:
        raise ValueError('--progress-every cannot be negative.')

    paddle.set_device(args.device)
    set_random_seed(args.seed)

    cfg = load_config(args.config)
    merge_config(args.opt)
    merge_config({
        'EvalDataset': {
            'data_fields': ['image', 'gt_bbox', 'gt_class', 'is_crowd']
        },
        'EvalReader': {
            'batch_size': 1,
            'shuffle': False,
            'drop_last': False,
            'use_shared_memory': False,
        }
    })

    dataset = create('EvalDataset')()
    loader = create('EvalReader')(dataset, args.worker_num)
    model = create(cfg.architecture)
    model.load_meanstd(cfg['TestReader']['sample_transforms'])
    load_pretrain_weight(model, args.weights)
    enhancer = getattr(model.neck, 'p2_enhancer', None)
    if enhancer is None:
        raise ValueError('The config must enable HybridEncoder.p2_enhancer.')
    if getattr(model.neck, 'p2_injection_position', None) != 'post_pan':
        raise ValueError('This diagnostic expects post_pan P2 injection.')
    model.eval()

    capture = _EnhancerCapture(enhancer)
    samples = []
    processed_images = 0
    try:
        with paddle.no_grad():
            for image_index, data in enumerate(loader):
                if args.max_images and image_index >= args.max_images:
                    break

                capture.reset()
                backbone_feats = model.backbone(data)
                model.neck(backbone_feats)
                if capture.record is None:
                    raise RuntimeError('The P2 enhancer hook did not run.')

                gate_map = capture.record['gate'][0]
                residual_map = capture.record['residual'][0]
                p3_map = capture.record['p3_abs'][0]
                gate_height, gate_width = gate_map.shape
                image_height = int(data['image'].shape[-2])
                image_width = int(data['image'].shape[-1])
                x_centers = (np.arange(gate_width, dtype=np.float32) +
                             0.5) * image_width / gate_width
                y_centers = (np.arange(gate_height, dtype=np.float32) +
                             0.5) * image_height / gate_height

                boxes, valid_mask = _boxes_and_valid_mask(data)
                scale_factor = _single_image_array(
                    data['scale_factor']).reshape([-1])
                scale_y, scale_x = float(scale_factor[0]), float(
                    scale_factor[1])
                occupied = np.zeros_like(gate_map, dtype=bool)
                current_image_id = _image_id(data, image_index)

                for object_index, box in enumerate(boxes):
                    mask = _box_cell_mask(box, x_centers, y_centers)
                    occupied |= mask
                    if not valid_mask[object_index]:
                        continue

                    original_width = max(
                        float(box[2] - box[0]) / max(scale_x, 1e-8), 0.)
                    original_height = max(
                        float(box[3] - box[1]) / max(scale_y, 1e-8), 0.)
                    original_area = original_width * original_height
                    region = _size_bucket(original_area)
                    if region is None or original_area <= 0.:
                        continue
                    stats = _region_stats(gate_map, residual_map, p3_map, mask)
                    samples.append({
                        'image_index': image_index,
                        'image_id': current_image_id,
                        'region': region,
                        'object_index': object_index,
                        'original_width': original_width,
                        'original_height': original_height,
                        'original_area': original_area,
                        **stats,
                    })

                background_stats = _region_stats(gate_map, residual_map,
                                                 p3_map, ~occupied)
                if background_stats is not None:
                    samples.append({
                        'image_index': image_index,
                        'image_id': current_image_id,
                        'region': 'background',
                        'object_index': -1,
                        'original_width': 0.,
                        'original_height': 0.,
                        'original_area': 0.,
                        **background_stats,
                    })

                processed_images += 1
                if args.progress_every and processed_images % \
                        args.progress_every == 0:
                    print('Processed {}/{} images'.format(
                        processed_images, len(dataset)))
    finally:
        capture.close()

    if not samples:
        raise RuntimeError('No gate samples were collected.')
    summary_rows = _summarize(samples)
    os.makedirs(args.output_dir, exist_ok=True)
    samples_path = os.path.join(args.output_dir, 'gate_object_samples.csv')
    summary_path = os.path.join(args.output_dir, 'gate_scale_summary.csv')
    json_path = os.path.join(args.output_dir, 'summary.json')
    _write_csv(samples_path, samples, SAMPLE_FIELDS)
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)

    alpha = float(enhancer.alpha.detach().numpy().reshape([-1])[0])
    result = {
        'config':
        os.path.abspath(args.config),
        'weights':
        os.path.abspath(args.weights),
        'device':
        args.device,
        'seed':
        args.seed,
        'processed_images':
        processed_images,
        'object_sample_count':
        sum(row['region'] != 'background' for row in samples),
        'background_sample_count':
        sum(row['region'] == 'background' for row in samples),
        'p2_alpha':
        alpha,
        'size_definition':
        'original-space bounding-box area',
        'summary':
        summary_rows,
    }
    with open(json_path, 'w', encoding='utf-8') as output:
        json.dump(result, output, indent=2)

    _print_summary(summary_rows)
    print('\nLearned P2 alpha: {:.6f}'.format(alpha))
    print('Object samples CSV:', samples_path)
    print('Scale summary CSV:', summary_path)
    print('Summary JSON:', json_path)


if __name__ == '__main__':
    main()
