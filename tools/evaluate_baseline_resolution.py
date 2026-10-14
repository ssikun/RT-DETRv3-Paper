"""Evaluate a fixed baseline resolution on CPU, or time five smoke images."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

import cv2
import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, set_random_seed
from ppdet.utils.cli import ArgsParser
from ppdet.utils.logger import set_logger_output


def main():
    parser = ArgsParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    merge_config(args.opt)
    merge_config({'use_gpu': False, 'use_ema': False, 'worker_num': 0,
                  'amp': False, 'fleet': False,
                  'EvalReader': {'batch_size': 1, 'use_shared_memory': False},
                  'output_eval': str(args.output_dir)})
    if args.smoke_only:
        merge_config({'EvalDataset': {'sample_num': 5}})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    set_logger_output(str(args.output_dir / 'eval.log'))
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    paddle.set_device('cpu')
    set_random_seed(3407)
    trainer = Trainer(cfg, mode='eval')
    weights_path = Path(cfg.weights)
    state = paddle.load(str(weights_path))
    expected = trainer.model.state_dict()
    if set(state) != set(expected):
        raise ValueError('Baseline state keys do not match the model')
    if any(list(expected[name].shape) != list(state[name].shape) for name in expected):
        raise ValueError('Baseline state tensor shapes do not match the model')
    trainer.model.set_state_dict(state)
    del state
    trainer.model.eval()
    assert not getattr(trainer.model.neck, 'use_p2_guidance', False)
    assert not trainer.model.transformer.use_p2_query_detail
    assert trainer.model.p2_center_aux_head is None
    expected_size = list(cfg.eval_size)
    protocol = {'config': args.config, 'weights': str(weights_path.resolve()),
                'weights_sha256': hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                'annotations': trainer.dataset.get_anno(),
                'annotations_sha256': hashlib.sha256(
                    Path(trainer.dataset.get_anno()).read_bytes()).hexdigest(),
                'eval_size': expected_size, 'device': paddle.get_device(),
                'amp': False, 'batch_size': 1, 'seed': 3407,
                'num_queries': trainer.model.transformer.num_queries[0],
                'smoke_only': args.smoke_only, 'loaded_tensors': len(expected)}
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('[ResolutionProtocol] {}'.format(json.dumps(protocol)), flush=True)
    start = time.perf_counter()
    inference_seconds = 0.
    count = 0
    total = len(trainer.dataset)
    with paddle.no_grad():
        for data in trainer.loader:
            if list(data['image'].shape) != [1, 3] + expected_size:
                raise RuntimeError('Evaluation image size does not match eval_size')
            prediction_start = time.perf_counter()
            outputs = trainer.model(data)
            inference_seconds += time.perf_counter() - prediction_start
            if not bool(paddle.isfinite(outputs['bbox']).all()):
                raise RuntimeError('Non-finite prediction')
            if not args.smoke_only:
                for metric in trainer._metrics:
                    metric.update(data, outputs)
            count += 1
            if count <= 5 or count % 10 == 0 or count == total:
                elapsed = time.perf_counter() - start
                eta = elapsed / count * (total - count)
                print('[ResolutionEval] size={} image={}/{} avg_seconds={:.3f} '
                      'remaining_minutes={:.1f}'.format(
                          expected_size[0], count, total, elapsed / count, eta / 60),
                      flush=True)
    profile = {'image_count': count, 'inference_seconds': inference_seconds,
               'mean_inference_seconds': inference_seconds / count,
               'prediction_wall_seconds': time.perf_counter() - start,
               'estimated_548_prediction_minutes': inference_seconds / count * 548 / 60}
    if not args.smoke_only:
        for metric in trainer._metrics:
            metric.accumulate()
        statistics = trainer._metrics[0].get_results()['bbox']
        profile.update(dict(zip(('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL'),
                                [float(value) for value in statistics[:6]])))
        profile['all_coco_statistics'] = [float(value) for value in statistics]
        if count != 548:
            raise RuntimeError('Expected all 548 VisDrone validation images')
    profile['total_wall_seconds'] = time.perf_counter() - start
    (args.output_dir / ('smoke_profile.json' if args.smoke_only else 'metrics.json')).write_text(
        json.dumps(profile, indent=2))
    print('[ResolutionResult] {}'.format(json.dumps(profile)), flush=True)


if __name__ == '__main__':
    main()
