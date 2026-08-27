#!/usr/bin/env python3
"""Measure P2-center heatmap proposal recall without retraining."""

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
import paddle.nn.functional as F

from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import set_random_seed
from ppdet.utils.checkpoint import load_pretrain_weight
from ppdet.utils.cli import ArgsParser

SIZE_BUCKETS = (
    ('tiny_0_8', 0., 8.**2),
    ('tiny_8_16', 8.**2 + 1e-6, 16.**2),
    ('small_16_24', 16.**2 + 1e-6, 24.**2),
    ('small_24_32', 24.**2 + 1e-6, 32.**2),
    ('small_0_32', 0., 32.**2),
)

DENSITY_GROUPS = (
    ('sparse_1_10', 1, 10),
    ('moderate_11_30', 11, 30),
    ('dense_31_plus', 31, None),
)


class ProposalRecallArgsParser(ArgsParser):

    def __init__(self):
        super().__init__()
        self.add_argument('--weights',
                          required=True,
                          help='P2-center checkpoint path.')
        self.add_argument('--output-dir',
                          default='results/p2_center_proposal_recall',
                          help='Directory for CSV and JSON outputs.')
        self.add_argument('--topk',
                          type=int,
                          nargs='+',
                          default=[50, 100, 300])
        self.add_argument('--radii', type=int, nargs='+', default=[0, 1])
        self.add_argument('--worker-num', type=int, default=0)
        self.add_argument('--seed', type=int, default=2026)
        self.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
        self.add_argument(
            '--max-images',
            type=int,
            default=0,
            help='Maximum images to scan; 0 scans the full dataset.')
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


def _valid_boxes(data):
    boxes = _single_image_array(data['gt_bbox']).astype(np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape([-1, 4])
    crowd = data.get('is_crowd', None)
    if crowd is None:
        return boxes
    crowd = _single_image_array(crowd).reshape([-1]).astype(bool)
    return boxes[~crowd]


def _size_bucket(area):
    for name, minimum, maximum in SIZE_BUCKETS[:-1]:
        if minimum <= area <= maximum:
            return name
    return None


def _density_group(small_count):
    for name, minimum, maximum in DENSITY_GROUPS:
        if small_count >= minimum and (maximum is None
                                       or small_count <= maximum):
            return name
    return 'no_small'


def _local_maximum_indices(score_map, maximum_k):
    pooled = F.max_pool2d(score_map.unsqueeze(1),
                          kernel_size=3,
                          stride=1,
                          padding=1)
    peaks = score_map.unsqueeze(1) >= pooled
    suppressed = paddle.where(peaks, score_map.unsqueeze(1),
                              paddle.full_like(score_map.unsqueeze(1), -1.))
    flattened = suppressed.flatten(1)
    k = min(maximum_k, int(flattened.shape[1]))
    values, indices = paddle.topk(flattened, k=k, axis=1)
    return values[0].numpy(), indices[0].numpy()


def _candidate_maps(center_scores):
    return {
        4:
        center_scores,
        8:
        F.max_pool2d(center_scores.unsqueeze(1), kernel_size=2,
                     stride=2).squeeze(1),
    }


def _covered(center_x, center_y, selected_x, selected_y, radius):
    if selected_x.size == 0:
        return False
    distance_x = np.abs(selected_x - center_x)
    distance_y = np.abs(selected_y - center_y)
    return bool(np.any(np.maximum(distance_x, distance_y) <= radius))


def _add_count(counts, key, covered):
    counts[key]['gt_count'] += 1
    counts[key]['covered_count'] += int(covered)


def _safe_recall(covered, total):
    if total == 0:
        return None
    return covered / total


def _write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_rows(counts):
    rows = []
    for key in sorted(counts):
        stride, size_bucket, density_group, topk, radius = key
        values = counts[key]
        rows.append({
            'stride':
            stride,
            'size_bucket':
            size_bucket,
            'density_group':
            density_group,
            'topk':
            topk,
            'radius':
            radius,
            'gt_count':
            values['gt_count'],
            'covered_count':
            values['covered_count'],
            'recall':
            _safe_recall(values['covered_count'], values['gt_count']),
        })
    return rows


def _build_collision_rows(center_sets, gt_counts):
    rows = []
    for stride, size_bucket in sorted(gt_counts):
        gt_count = gt_counts[(stride, size_bucket)]
        unique_count = len(center_sets[(stride, size_bucket)])
        collisions = max(gt_count - unique_count, 0)
        rows.append({
            'stride': stride,
            'size_bucket': size_bucket,
            'gt_count': gt_count,
            'unique_center_count': unique_count,
            'collision_count': collisions,
            'collision_rate': _safe_recall(collisions, gt_count),
        })
    return rows


def _print_focus_table(rows):
    print('\nDense 8-16 proposal recall')
    print('stride  topk  radius      GT  covered  recall')
    focus = [
        row for row in rows if row['size_bucket'] == 'tiny_8_16'
        and row['density_group'] == 'dense_31_plus'
    ]
    for row in focus:
        recall = row['recall']
        recall_text = 'nan' if recall is None else '{:.2%}'.format(recall)
        print('{:>6} {:>5} {:>7} {:>7} {:>8} {:>7}'.format(
            row['stride'], row['topk'], row['radius'], row['gt_count'],
            row['covered_count'], recall_text))


def main():
    args = ProposalRecallArgsParser().parse_args()
    if any(value <= 0 for value in args.topk):
        raise ValueError('--topk values must be positive.')
    if any(value < 0 for value in args.radii):
        raise ValueError('--radii values cannot be negative.')
    if args.progress_every < 0:
        raise ValueError('--progress-every cannot be negative.')
    if args.max_images < 0:
        raise ValueError('--max-images cannot be negative.')

    topk_values = sorted(set(args.topk))
    radii = sorted(set(args.radii))
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
    if getattr(model, 'p2_center_aux_head', None) is None:
        raise ValueError('The config must enable p2_center_aux_head.')
    model.eval()

    counts = defaultdict(lambda: {'gt_count': 0, 'covered_count': 0})
    center_sets = defaultdict(set)
    gt_counts = defaultdict(int)
    image_count = 0
    small_gt_count = 0
    maximum_k = max(topk_values)

    with paddle.no_grad():
        for image_index, data in enumerate(loader):
            if args.max_images and image_index >= args.max_images:
                break
            resized_boxes = _valid_boxes(data)
            scale_factor = _single_image_array(data['scale_factor']).reshape(
                [-1])
            scale_y, scale_x = float(scale_factor[0]), float(scale_factor[1])
            original_boxes = resized_boxes.copy()
            original_boxes[:, 0::2] /= scale_x
            original_boxes[:, 1::2] /= scale_y
            widths = np.clip(original_boxes[:, 2] - original_boxes[:, 0], 0.,
                             None)
            heights = np.clip(original_boxes[:, 3] - original_boxes[:, 1], 0.,
                              None)
            areas = widths * heights
            small_mask = np.logical_and(areas > 0., areas <= 32.**2)
            small_boxes = resized_boxes[small_mask]
            small_areas = areas[small_mask]
            density_group = _density_group(len(small_boxes))

            backbone_feats = model.backbone(data)
            head = model.p2_center_aux_head
            features = head.stem(backbone_feats[0])
            logits = head.pred(features)
            center_scores = F.sigmoid(logits).max(axis=1)
            maps = _candidate_maps(center_scores)

            image_id_array = _single_image_array(data['im_id']).reshape([-1])
            image_id = int(image_id_array[0])

            for stride, score_map in maps.items():
                height, width = int(score_map.shape[1]), int(
                    score_map.shape[2])
                _, selected_indices = _local_maximum_indices(
                    score_map, maximum_k)
                selected_y = selected_indices // width
                selected_x = selected_indices % width

                for box, area in zip(small_boxes, small_areas):
                    bucket = _size_bucket(float(area))
                    if bucket is None:
                        continue
                    center_x = (box[0] + box[2]) * 0.5 / stride
                    center_y = (box[1] + box[3]) * 0.5 / stride
                    center_x = int(np.clip(math.floor(center_x), 0, width - 1))
                    center_y = int(np.clip(math.floor(center_y), 0,
                                           height - 1))

                    for size_label in (bucket, 'small_0_32'):
                        gt_counts[(stride, size_label)] += 1
                        center_sets[(stride, size_label)].add(
                            (image_id, center_y, center_x))
                        for density_label in ('all', density_group):
                            for topk in topk_values:
                                current_x = selected_x[:topk]
                                current_y = selected_y[:topk]
                                for radius in radii:
                                    is_covered = _covered(
                                        center_x, center_y, current_x,
                                        current_y, radius)
                                    key = (stride, size_label, density_label,
                                           topk, radius)
                                    _add_count(counts, key, is_covered)

            image_count += 1
            small_gt_count += len(small_boxes)
            if args.progress_every and (image_index +
                                        1) % args.progress_every == 0:
                print('Processed {}/{} images'.format(image_index + 1,
                                                      len(loader)),
                      flush=True)

    rows = _build_rows(counts)
    collision_rows = _build_collision_rows(center_sets, gt_counts)
    os.makedirs(args.output_dir, exist_ok=True)
    recall_path = os.path.join(args.output_dir, 'proposal_recall.csv')
    collision_path = os.path.join(args.output_dir,
                                  'center_collision_summary.csv')
    summary_path = os.path.join(args.output_dir,
                                'proposal_recall_summary.json')
    _write_csv(recall_path, rows, list(rows[0]))
    _write_csv(collision_path, collision_rows, list(collision_rows[0]))

    summary = {
        'config': args.config,
        'weights': os.path.abspath(args.weights),
        'image_count': image_count,
        'small_gt_count': small_gt_count,
        'max_images': args.max_images,
        'topk': topk_values,
        'radii': radii,
        'proposal_recall': rows,
        'center_collisions': collision_rows,
    }
    with open(summary_path, 'w', encoding='utf-8') as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write('\n')

    _print_focus_table(rows)
    print('\nProposal recall CSV: {}'.format(recall_path))
    print('Collision CSV: {}'.format(collision_path))
    print('Summary JSON: {}'.format(summary_path))


if __name__ == '__main__':
    main()
