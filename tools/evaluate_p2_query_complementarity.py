#!/usr/bin/env python3
"""Evaluate whether P2-center candidates complement encoder Top-K queries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
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
    ('medium_32_96', 32.**2 + 1e-6, 96.**2),
    ('large_96_plus', 96.**2 + 1e-6, float('inf')),
)

DENSITY_GROUPS = (
    ('sparse_1_10', 1, 10),
    ('moderate_11_30', 11, 30),
    ('dense_31_plus', 31, None),
)


class ComplementarityArgsParser(ArgsParser):

    def __init__(self):
        super().__init__()
        self.add_argument('--weights',
                          required=True,
                          help='P2-center checkpoint path.')
        self.add_argument('--output-dir',
                          default='results/p2_query_complementarity',
                          help='Directory for CSV and JSON outputs.')
        self.add_argument('--reserve-queries',
                          type=int,
                          nargs='+',
                          default=[25, 50, 75, 100])
        self.add_argument('--p2-pool-size', type=int, default=300)
        self.add_argument('--iou-thresholds',
                          type=float,
                          nargs='+',
                          default=[0.3, 0.5])
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
    for name, minimum, maximum in SIZE_BUCKETS:
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
    values = values[0].numpy()
    indices = indices[0].numpy().astype(np.int64)
    return indices[values >= 0.]


def _reserve_selection(base_order, p2_order, total_queries, reserve_queries):
    keep_count = max(total_queries - reserve_queries, 0)
    selected = list(base_order[:keep_count])
    selected_set = set(selected)
    p2_added = 0

    for index in p2_order:
        index = int(index)
        if index in selected_set:
            continue
        selected.append(index)
        selected_set.add(index)
        p2_added += 1
        if p2_added >= reserve_queries or len(selected) >= total_queries:
            break

    if len(selected) < total_queries:
        for index in base_order[keep_count:]:
            index = int(index)
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            if len(selected) >= total_queries:
                break

    return np.asarray(selected[:total_queries], dtype=np.int64), p2_added


def _cxcywh_to_xyxy(boxes, image_height, image_width):
    boxes = np.asarray(boxes, dtype=np.float32)
    cx = boxes[:, 0] * image_width
    cy = boxes[:, 1] * image_height
    width = boxes[:, 2] * image_width
    height = boxes[:, 3] * image_height
    return np.stack([
        cx - width * 0.5, cy - height * 0.5, cx + width * 0.5,
        cy + height * 0.5
    ],
                    axis=1)


def _max_iou(gt_boxes, proposal_boxes):
    if len(gt_boxes) == 0:
        return np.zeros([0], dtype=np.float32)
    if len(proposal_boxes) == 0:
        return np.zeros([len(gt_boxes)], dtype=np.float32)

    top_left = np.maximum(gt_boxes[:, None, :2], proposal_boxes[None, :, :2])
    bottom_right = np.minimum(gt_boxes[:, None, 2:], proposal_boxes[None, :,
                                                                    2:])
    intersection_wh = np.clip(bottom_right - top_left, 0., None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    gt_wh = np.clip(gt_boxes[:, 2:] - gt_boxes[:, :2], 0., None)
    proposal_wh = np.clip(proposal_boxes[:, 2:] - proposal_boxes[:, :2], 0.,
                          None)
    gt_area = gt_wh[:, 0] * gt_wh[:, 1]
    proposal_area = proposal_wh[:, 0] * proposal_wh[:, 1]
    union = gt_area[:, None] + proposal_area[None, :] - intersection
    iou = intersection / np.clip(union, 1e-9, None)
    return iou.max(axis=1)


def _add_recall(counts, mode, bucket, density, threshold, covered):
    for density_label in ('all', density):
        key = (mode, bucket, density_label, threshold)
        counts[key]['gt_count'] += 1
        counts[key]['covered_count'] += int(covered)


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def _build_recall_rows(counts):
    rows = []
    for key in sorted(counts):
        mode, bucket, density, threshold = key
        values = counts[key]
        rows.append({
            'selection':
            mode,
            'size_bucket':
            bucket,
            'density_group':
            density,
            'iou_threshold':
            threshold,
            'gt_count':
            values['gt_count'],
            'covered_count':
            values['covered_count'],
            'recall':
            _safe_ratio(values['covered_count'], values['gt_count']),
        })
    return rows


def _build_overlap_rows(overlap_counts):
    rows = []
    for quota in sorted(overlap_counts):
        values = overlap_counts[quota]
        rows.append({
            'reserve_queries':
            quota,
            'image_count':
            values['image_count'],
            'p2_candidates_considered':
            values['p2_candidates_considered'],
            'p2_in_base_topk':
            values['p2_in_base_topk'],
            'p2_base_overlap_rate':
            _safe_ratio(values['p2_in_base_topk'],
                        values['p2_candidates_considered']),
            'p2_queries_added':
            values['p2_queries_added'],
            'selected_not_in_base_topk':
            values['selected_not_in_base_topk'],
        })
    return rows


def _write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_focus_table(rows):
    focus = [
        row for row in rows if row['size_bucket'] == 'tiny_8_16'
        and row['density_group'] == 'dense_31_plus'
    ]
    print('\nDense 8-16 encoder proposal recall')
    print('selection       IoU      GT  covered  recall')
    for row in focus:
        recall = row['recall']
        recall_text = 'nan' if recall is None else '{:.2%}'.format(recall)
        print('{:<14} {:>4.2f} {:>7} {:>8} {:>7}'.format(
            row['selection'], row['iou_threshold'], row['gt_count'],
            row['covered_count'], recall_text))


def main():
    args = ComplementarityArgsParser().parse_args()
    if any(value <= 0 for value in args.reserve_queries):
        raise ValueError('--reserve-queries values must be positive.')
    if args.p2_pool_size <= 0:
        raise ValueError('--p2-pool-size must be positive.')
    if any(value <= 0 or value > 1 for value in args.iou_thresholds):
        raise ValueError('--iou-thresholds values must be in (0, 1].')
    if args.progress_every < 0:
        raise ValueError('--progress-every cannot be negative.')
    if args.max_images < 0:
        raise ValueError('--max-images cannot be negative.')

    reserve_values = sorted(set(args.reserve_queries))
    thresholds = sorted(set(args.iou_thresholds))
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

    transformer = model.transformer
    total_queries = int(transformer.num_queries[0])
    if max(reserve_values) >= total_queries:
        raise ValueError(
            'Reserved queries must be smaller than {}.'.format(total_queries))

    recall_counts = defaultdict(lambda: {'gt_count': 0, 'covered_count': 0})
    overlap_counts = defaultdict(lambda: defaultdict(int))
    image_count = 0

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
            small_count = int(
                np.logical_and(areas > 0., areas <= 32.**2).sum())
            density = _density_group(small_count)

            backbone_feats = model.backbone(data)
            center_head = model.p2_center_aux_head
            center_features = center_head.stem(backbone_feats[0])
            center_logits = center_head.pred(center_features)
            center_scores = F.sigmoid(center_logits).max(axis=1)
            stride8_scores = F.max_pool2d(center_scores.unsqueeze(1),
                                          kernel_size=2,
                                          stride=2).squeeze(1)

            body_feats = model.neck(backbone_feats)

            memory, spatial_shapes, _ = transformer._get_encoder_input(
                body_feats)
            anchors, valid_mask = transformer._generate_anchors(spatial_shapes)
            memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
            output_memory = transformer.enc_output[0](memory)
            class_logits = transformer.enc_score_head[0](output_memory)
            box_unact = transformer.enc_bbox_head[0](output_memory) + anchors
            base_scores = class_logits.max(axis=-1)
            _, base_indices = paddle.topk(base_scores, k=total_queries, axis=1)
            base_order = base_indices[0].numpy().astype(np.int64)

            p3_height, p3_width = spatial_shapes[0]
            p3_height, p3_width = int(p3_height), int(p3_width)
            if tuple(stride8_scores.shape[-2:]) != (p3_height, p3_width):
                raise ValueError(
                    'P2 stride-8 score map {} does not match P3 {}.'.format(
                        tuple(stride8_scores.shape[-2:]),
                        (p3_height, p3_width)))
            p2_order = _local_maximum_indices(stride8_scores,
                                              args.p2_pool_size)
            # P3 is the first flattened encoder level, so its local indices
            # are also global memory indices.

            all_boxes = F.sigmoid(box_unact)[0].numpy()
            image_height, image_width = data['image'].shape[-2:]
            all_boxes = _cxcywh_to_xyxy(all_boxes, int(image_height),
                                        int(image_width))

            selections = {'base_top{}'.format(total_queries): base_order}
            base_set = set(int(index) for index in base_order)
            for quota in reserve_values:
                selected, p2_added = _reserve_selection(
                    base_order, p2_order, total_queries, quota)
                mode = 'reserve_{}'.format(quota)
                selections[mode] = selected

                considered = p2_order[:quota]
                overlap = sum(int(index) in base_set for index in considered)
                selected_new = sum(
                    int(index) not in base_set for index in selected)
                values = overlap_counts[quota]
                values['image_count'] += 1
                values['p2_candidates_considered'] += len(considered)
                values['p2_in_base_topk'] += overlap
                values['p2_queries_added'] += p2_added
                values['selected_not_in_base_topk'] += selected_new

            for mode, selected in selections.items():
                max_ious = _max_iou(resized_boxes, all_boxes[selected])
                for area, max_iou in zip(areas, max_ious):
                    bucket = _size_bucket(float(area))
                    if bucket is None:
                        continue
                    for threshold in thresholds:
                        _add_recall(recall_counts, mode, bucket, density,
                                    threshold, max_iou >= threshold)

            image_count += 1
            if args.progress_every and (image_index +
                                        1) % args.progress_every == 0:
                print('Processed {}/{} images'.format(image_index + 1,
                                                      len(loader)),
                      flush=True)

    recall_rows = _build_recall_rows(recall_counts)
    overlap_rows = _build_overlap_rows(overlap_counts)
    os.makedirs(args.output_dir, exist_ok=True)
    recall_path = os.path.join(args.output_dir, 'proposal_recall.csv')
    overlap_path = os.path.join(args.output_dir, 'candidate_overlap.csv')
    summary_path = os.path.join(args.output_dir, 'summary.json')
    _write_csv(recall_path, recall_rows)
    _write_csv(overlap_path, overlap_rows)

    summary = {
        'config': args.config,
        'weights': os.path.abspath(args.weights),
        'image_count': image_count,
        'total_queries': total_queries,
        'reserve_queries': reserve_values,
        'p2_pool_size': args.p2_pool_size,
        'iou_thresholds': thresholds,
        'proposal_recall': recall_rows,
        'candidate_overlap': overlap_rows,
    }
    with open(summary_path, 'w', encoding='utf-8') as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write('\n')

    _print_focus_table(recall_rows)
    print('\nCandidate overlap')
    for row in overlap_rows:
        print(row)
    print('\nProposal recall CSV: {}'.format(recall_path))
    print('Candidate overlap CSV: {}'.format(overlap_path))
    print('Summary JSON: {}'.format(summary_path))


if __name__ == '__main__':
    main()
