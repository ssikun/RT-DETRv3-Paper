# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import math
import os
import random
import sys
from collections import Counter, OrderedDict, defaultdict

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

COMPOSITION_GROUPS = [
    'small_only',
    'small_dominant',
    'mixed',
    'medium_large_dominant',
    'no_small',
]

PARAMETER_GROUPS = OrderedDict([
    ('backbone_stem', ('backbone.conv1.', )),
    ('backbone_res2', ('backbone.res2.', )),
    ('backbone_res3', ('backbone.res3.', )),
    ('backbone_res4', ('backbone.res4.', )),
    ('backbone_res5', ('backbone.res5.', )),
    ('neck_p2_enhancer', ('neck.p2_enhancer.', )),
    ('neck_input_proj', ('neck.input_proj.', )),
    ('neck_encoder', ('neck.encoder.', )),
    ('neck_fpn_pan', (
        'neck.lateral_convs.',
        'neck.fpn_blocks.',
        'neck.downsample_convs.',
        'neck.pan_blocks.',
    )),
    ('transformer', ('transformer.', )),
    ('detr_head', ('detr_head.', )),
    ('full_scale_aux_head', ('aux_o2m_head.', )),
    ('p2_center_head', ('p2_center_aux_head.', )),
])

SHARED_AUDIT_GROUPS = {'backbone_stem', 'backbone_res2'}
EXPECTED_ZERO_AUX_GROUPS = {
    'backbone_res3',
    'backbone_res4',
    'backbone_res5',
    'neck_p2_enhancer',
    'neck_input_proj',
    'neck_encoder',
    'neck_fpn_pan',
    'transformer',
    'detr_head',
    'full_scale_aux_head',
}
EXPECTED_ZERO_MAIN_GROUPS = {'p2_center_head'}


class GradientAuditArgsParser(ArgsParser):

    def __init__(self):
        super().__init__()
        self.add_argument('--weights',
                          required=True,
                          help='P2-center checkpoint path.')
        self.add_argument('--output-dir',
                          default='results/p2_center_gradient_audit',
                          help='Directory for CSV and JSON audit outputs.')
        self.add_argument('--samples-per-group', type=int, default=6)
        self.add_argument('--max-batches', type=int, default=1000)
        self.add_argument('--epoch-id', type=int, default=39)
        self.add_argument('--seed', type=int, default=2026)
        self.add_argument('--worker-num', type=int, default=0)
        self.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')


def _parameter_group(name):
    for group, prefixes in PARAMETER_GROUPS.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            return group
    return 'other'


def _parameter_inventory(model):
    counts = Counter()
    other_names = []
    for name, _ in model.named_parameters():
        group = _parameter_group(name)
        counts[group] += 1
        if group == 'other':
            other_names.append(name)
    required = [
        'backbone_stem',
        'backbone_res2',
        'backbone_res3',
        'backbone_res4',
        'backbone_res5',
        'neck_p2_enhancer',
        'p2_center_head',
    ]
    missing = [group for group in required if counts[group] == 0]
    if missing:
        raise RuntimeError(
            'Parameter grouping failed; no parameters found for: {}.'.format(
                ', '.join(missing)))
    print('Parameter groups: {}'.format(dict(counts)))
    if other_names:
        print('Unclassified parameters (first 20): {}'.format(
            other_names[:20]))
    return {'counts': dict(counts), 'other_names': other_names}


def _set_batch_norm_eval(model):
    for _, layer in model.named_sublayers():
        class_name = layer.__class__.__name__
        if 'BatchNorm' in class_name or class_name == 'SyncBatchNorm':
            layer.eval()


def _clear_gradients(model):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.clear_gradient()


def _tensor_scalar(value):
    array = value.detach().numpy().reshape([-1])
    return float(array[0])


def _target_composition(data, small_max_size):
    boxes = data['origin_gt_bbox'].numpy()[0]
    valid = data['pad_origin_gt_mask'].numpy()[0, :, 0] > 0
    boxes = boxes[valid]
    if boxes.shape[0] == 0:
        return 'no_small', {'small': 0, 'medium': 0, 'large': 0, 'total': 0}

    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0., a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0., a_max=None)
    areas = widths * heights
    small = int(np.sum(np.logical_and(areas > 0., areas <= small_max_size**2)))
    medium = int(
        np.sum(np.logical_and(areas > small_max_size**2, areas <= 96.**2)))
    large = int(np.sum(areas > 96.**2))
    medium_large = medium + large

    if small == 0:
        group = 'no_small'
    elif medium_large == 0:
        group = 'small_only'
    elif small >= 2 * medium_large:
        group = 'small_dominant'
    elif medium_large >= 2 * small:
        group = 'medium_large_dominant'
    else:
        group = 'mixed'
    return group, {
        'small': small,
        'medium': medium,
        'large': large,
        'total': small + medium + large,
    }


def _gradient_snapshot(model, keep_arrays=False):
    squared_norms = defaultdict(float)
    nonzero_parameters = Counter()
    arrays = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().numpy().astype('float32', copy=False)
        group = _parameter_group(name)
        squared_norm = float(np.sum(grad * grad, dtype=np.float64))
        squared_norms[group] += squared_norm
        if squared_norm > 0.:
            nonzero_parameters[group] += 1
        if keep_arrays and group in SHARED_AUDIT_GROUPS:
            arrays[name] = grad.copy()
    norms = {
        group: math.sqrt(squared_norms.get(group, 0.))
        for group in list(PARAMETER_GROUPS.keys()) + ['other']
    }
    return norms, dict(nonzero_parameters), arrays


def _gradient_comparison(model, main_norms, main_arrays):
    aux_squared_norms = defaultdict(float)
    dots = defaultdict(float)
    nonzero_parameters = Counter()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().numpy().astype('float32', copy=False)
        group = _parameter_group(name)
        squared_norm = float(np.sum(grad * grad, dtype=np.float64))
        aux_squared_norms[group] += squared_norm
        if squared_norm > 0.:
            nonzero_parameters[group] += 1
        if name in main_arrays:
            dots[group] += float(
                np.sum(main_arrays[name] * grad, dtype=np.float64))

    aux_norms = {
        group: math.sqrt(aux_squared_norms.get(group, 0.))
        for group in list(PARAMETER_GROUPS.keys()) + ['other']
    }
    comparisons = {}
    for group in list(PARAMETER_GROUPS.keys()) + ['other']:
        main_norm = main_norms.get(group, 0.)
        aux_norm = aux_norms.get(group, 0.)
        denominator = main_norm * aux_norm
        cosine = dots.get(group,
                          0.) / denominator if denominator > 0. else None
        ratio = aux_norm / main_norm if main_norm > 0. else None
        comparisons[group] = {
            'main_norm': main_norm,
            'aux_norm': aux_norm,
            'aux_main_norm_ratio': ratio,
            'cosine': cosine,
            'aux_nonzero_parameters': nonzero_parameters.get(group, 0),
        }
    return comparisons


def _image_id(data):
    if 'im_id' not in data:
        return None
    values = data['im_id'].numpy().reshape([-1])
    return int(values[0]) if values.size else None


def _audit_batch(model, data, center_loss_weight, epoch_id):
    data['epoch_id'] = epoch_id
    _clear_gradients(model)
    outputs = model(data)
    if 'loss_p2_center_aux' not in outputs:
        raise KeyError(
            'The configured model did not return loss_p2_center_aux.')
    center_loss_from_total = outputs['loss_p2_center_aux']
    main_loss = outputs['loss'] - center_loss_from_total
    main_loss.backward()
    main_norms, main_nonzero, main_arrays = _gradient_snapshot(
        model, keep_arrays=True)
    main_loss_value = _tensor_scalar(main_loss)
    _clear_gradients(model)
    del outputs

    backbone_feats = model.backbone(data)
    center_outputs = model.p2_center_aux_head(backbone_feats[0], data)
    aux_loss = center_outputs['loss'] * center_loss_weight
    aux_loss.backward()
    comparisons = _gradient_comparison(model, main_norms, main_arrays)
    aux_loss_value = _tensor_scalar(aux_loss)
    _clear_gradients(model)

    return {
        'main_loss': main_loss_value,
        'center_aux_loss': aux_loss_value,
        'main_nonzero_parameters': main_nonzero,
        'comparisons': comparisons,
    }


def _describe(values):
    finite = np.asarray([
        value for value in values if value is not None and np.isfinite(value)
    ],
                        dtype='float64')
    if finite.size == 0:
        return {'count': 0}
    return {
        'count': int(finite.size),
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'q25': float(np.percentile(finite, 25)),
        'q75': float(np.percentile(finite, 75)),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'negative_fraction': float(np.mean(finite < 0.)),
        'strong_negative_fraction': float(np.mean(finite <= -0.1)),
    }


def _summarize(records):
    summary = {'sample_count': len(records), 'by_composition': {}}
    record_sets = {'overall': records}
    for composition in COMPOSITION_GROUPS:
        record_sets[composition] = [
            record for record in records
            if record['composition'] == composition
        ]

    for composition, subset in record_sets.items():
        group_summary = {'sample_count': len(subset), 'parameters': {}}
        for parameter_group in list(PARAMETER_GROUPS.keys()) + ['other']:
            main_norms = [
                record['comparisons'][parameter_group]['main_norm']
                for record in subset
            ]
            aux_norms = [
                record['comparisons'][parameter_group]['aux_norm']
                for record in subset
            ]
            ratios = [
                record['comparisons'][parameter_group]['aux_main_norm_ratio']
                for record in subset
            ]
            cosines = [
                record['comparisons'][parameter_group]['cosine']
                for record in subset
            ]
            group_summary['parameters'][parameter_group] = {
                'main_norm': _describe(main_norms),
                'aux_norm': _describe(aux_norms),
                'aux_main_norm_ratio': _describe(ratios),
                'cosine': _describe(cosines),
            }
        summary['by_composition'][composition] = group_summary

    violations = []
    for record in records:
        for group in EXPECTED_ZERO_AUX_GROUPS:
            aux_norm = record['comparisons'][group]['aux_norm']
            if aux_norm > 1e-12:
                violations.append({
                    'image_id': record['image_id'],
                    'composition': record['composition'],
                    'parameter_group': group,
                    'aux_norm': aux_norm,
                })
    summary['expected_zero_aux_gradient_violations'] = violations

    main_violations = []
    for record in records:
        for group in EXPECTED_ZERO_MAIN_GROUPS:
            main_norm = record['comparisons'][group]['main_norm']
            if main_norm > 1e-12:
                main_violations.append({
                    'image_id': record['image_id'],
                    'composition': record['composition'],
                    'parameter_group': group,
                    'main_norm': main_norm,
                })
    summary['expected_zero_main_gradient_violations'] = main_violations

    shared_signals = {}
    for group in sorted(SHARED_AUDIT_GROUPS):
        stats = summary['by_composition']['overall']['parameters'][group]
        cosine = stats['cosine']
        ratio = stats['aux_main_norm_ratio']
        shared_signals[group] = {
            'enough_samples':
            cosine.get('count', 0) >= 10,
            'stable_negative_direction':
            cosine.get('negative_fraction', 0.) >= 0.6
            and cosine.get('median', 0.) <= -0.1,
            'nontrivial_magnitude':
            ratio.get('median', 0.) >= 0.1,
        }
    summary['heuristic_conflict_signals'] = shared_signals
    return summary


def _flatten_record(record):
    row = {
        'sample_index': record['sample_index'],
        'image_id': record['image_id'],
        'composition': record['composition'],
        'small_gt': record['targets']['small'],
        'medium_gt': record['targets']['medium'],
        'large_gt': record['targets']['large'],
        'total_gt': record['targets']['total'],
        'main_loss': record['main_loss'],
        'center_aux_loss': record['center_aux_loss'],
    }
    for group, comparison in record['comparisons'].items():
        row[group + '_main_norm'] = comparison['main_norm']
        row[group + '_aux_norm'] = comparison['aux_norm']
        row[group + '_ratio'] = comparison['aux_main_norm_ratio']
        row[group + '_cosine'] = comparison['cosine']
        row[group +
            '_aux_nonzero_params'] = comparison['aux_nonzero_parameters']
    return row


def _write_outputs(output_dir, metadata, records, summary):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'gradient_audit_samples.csv')
    json_path = os.path.join(output_dir, 'gradient_audit_summary.json')
    rows = [_flatten_record(record) for record in records]
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump({
            'metadata': metadata,
            'summary': summary,
        },
                  handle,
                  indent=2,
                  sort_keys=True)
    return csv_path, json_path


def main():
    args = GradientAuditArgsParser().parse_args()
    if args.samples_per_group < 1:
        raise ValueError('--samples-per-group must be positive.')
    if args.max_batches < 1:
        raise ValueError('--max-batches must be positive.')

    paddle.set_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed)

    cfg = load_config(args.config)
    merge_config(args.opt)
    merge_config({
        'worker_num': args.worker_num,
        'TrainReader': {
            'batch_size': 1,
            'shuffle': False,
            'drop_last': False,
            'use_shared_memory': False,
        }
    })

    dataset = create('TrainDataset')()
    if hasattr(dataset, 'set_epoch'):
        dataset.set_epoch(args.epoch_id)
    loader = create('TrainReader')(dataset, args.worker_num)
    model = create(cfg.architecture)
    model.load_meanstd(cfg['TestReader']['sample_transforms'])
    load_pretrain_weight(model, args.weights)

    if getattr(model, 'p2_center_aux_head', None) is None:
        raise ValueError('The config must enable p2_center_aux_head.')
    center_loss_weight = float(model.p2_center_aux_loss_weight)
    small_max_size = float(model.p2_center_aux_head.small_object_max_size)
    parameter_inventory = _parameter_inventory(model)
    model.train()
    _set_batch_norm_eval(model)

    counts = Counter()
    records = []
    for batch_index, data in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        composition, target_counts = _target_composition(data, small_max_size)
        if counts[composition] >= args.samples_per_group:
            continue

        audit = _audit_batch(model, data, center_loss_weight, args.epoch_id)
        record = {
            'sample_index': len(records),
            'batch_index': batch_index,
            'image_id': _image_id(data),
            'composition': composition,
            'targets': target_counts,
            **audit,
        }
        records.append(record)
        counts[composition] += 1

        stem = audit['comparisons']['backbone_stem']
        res2 = audit['comparisons']['backbone_res2']
        print('[GradientAudit] sample={} image={} group={} gt={}/{}/{} '
              'stem_cos={} stem_ratio={} res2_cos={} res2_ratio={}'.format(
                  len(records), record['image_id'], composition,
                  target_counts['small'], target_counts['medium'],
                  target_counts['large'],
                  'nan' if stem['cosine'] is None else '{:.6f}'.format(
                      stem['cosine']), 'nan' if stem['aux_main_norm_ratio']
                  is None else '{:.6f}'.format(stem['aux_main_norm_ratio']),
                  'nan' if res2['cosine'] is None else '{:.6f}'.format(
                      res2['cosine']), 'nan' if res2['aux_main_norm_ratio']
                  is None else '{:.6f}'.format(res2['aux_main_norm_ratio'])))

        if all(counts[group] >= args.samples_per_group
               for group in COMPOSITION_GROUPS):
            break

    summary = _summarize(records)
    metadata = {
        'config':
        args.config,
        'weights':
        args.weights,
        'epoch_id':
        args.epoch_id,
        'seed':
        args.seed,
        'samples_per_group':
        args.samples_per_group,
        'max_batches':
        args.max_batches,
        'scanned_batches':
        min(batch_index +
            1, args.max_batches) if 'batch_index' in locals() else 0,
        'composition_counts':
        dict(counts),
        'center_loss_weight':
        center_loss_weight,
        'small_object_max_size':
        small_max_size,
        'parameter_inventory':
        parameter_inventory,
    }
    csv_path, json_path = _write_outputs(args.output_dir, metadata, records,
                                         summary)

    print('\nGradient audit complete.')
    print('Composition counts: {}'.format(dict(counts)))
    print('CSV: {}'.format(csv_path))
    print('JSON: {}'.format(json_path))
    violations = summary['expected_zero_aux_gradient_violations']
    print('Unexpected auxiliary-gradient violations: {}'.format(
        len(violations)))
    main_violations = summary['expected_zero_main_gradient_violations']
    print('Unexpected main-gradient violations: {}'.format(
        len(main_violations)))
    for group, signal in summary['heuristic_conflict_signals'].items():
        print('Conflict signal {}: {}'.format(group, signal))


if __name__ == '__main__':
    main()
