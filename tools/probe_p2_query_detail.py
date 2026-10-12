"""Test P2 query detail while keeping a trained baseline fixed."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

import paddle
import paddle.nn as nn

from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import set_random_seed
from ppdet.metrics import COCOMetric
from ppdet.utils.cli import ArgsParser
from ppdet.utils.logger import set_logger_output


DETAIL_PREFIX = 'transformer.p2_query_detail.'


def state_digest(model, include_detail=False):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if include_detail or not name.startswith(DETAIL_PREFIX):
            digest.update(name.encode('utf-8'))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_baseline(model, filename):
    state = paddle.load(str(filename))
    current = model.state_dict()
    original = {name for name in current if not name.startswith(DETAIL_PREFIX)}
    missing = sorted(original - set(state))
    extra = sorted(set(state) - original)
    mismatched = [name for name in original & set(state)
                  if list(current[name].shape) != list(state[name].shape)]
    if missing or extra or mismatched:
        raise ValueError('Baseline mismatch: missing={}, extra={}, shapes={}'
                         .format(missing, extra, mismatched))
    current.update(state)
    model.set_state_dict(current)
    return len(original)


class FrozenBaseline:
    def __init__(self, model):
        self.model = model
        self.parameters = []
        for name, parameter in model.named_parameters():
            parameter.stop_gradient = not name.startswith(DETAIL_PREFIX)
            if not parameter.stop_gradient:
                self.parameters.append(parameter)
        if not self.parameters:
            raise ValueError('No P2 query detail parameters found')
        self.norms = [layer for name, layer in model.named_sublayers()
                      if not name.startswith(DETAIL_PREFIX) and
                      isinstance(layer, (nn.BatchNorm2D, nn.SyncBatchNorm))]
        self.hook = model.register_forward_pre_hook(self._lock_norms)
        self.original_digest = state_digest(model)

    def _lock_norms(self, layer, inputs):
        for norm in self.norms:
            norm.eval()

    def verify(self):
        unexpected = [name for name, parameter in self.model.named_parameters()
                      if not name.startswith(DETAIL_PREFIX) and
                      (not parameter.stop_gradient or parameter.grad is not None)]
        if unexpected:
            raise RuntimeError('Original parameter gradient: {}'.format(unexpected))
        if state_digest(self.model) != self.original_digest:
            raise RuntimeError('Frozen baseline parameter or buffer changed')


def check_equivalence(model, data):
    model.eval()
    model.transformer.use_p2_query_detail = False
    try:
        with paddle.no_grad():
            original = model(data)
    finally:
        model.transformer.use_p2_query_detail = True
    with paddle.no_grad():
        candidate = model(data)
    if not paddle.equal_all(original['bbox_num'], candidate['bbox_num']):
        raise RuntimeError('Zero-init detection count changed')
    if not paddle.allclose(original['bbox'], candidate['bbox'], atol=1e-6,
                           rtol=1e-6):
        raise RuntimeError('Zero-init baseline prediction mismatch')
    return float((original['bbox'] - candidate['bbox']).abs().max())


def check_gradients(model, frozen, data):
    model.train()
    data['epoch_id'] = 40
    loss = model(data)['loss']
    if not bool(paddle.isfinite(loss)):
        raise RuntimeError('Non-finite audit loss')
    loss.backward()
    gradient = model.transformer.p2_query_detail.output_proj.weight.grad
    if gradient is None or not bool(paddle.isfinite(gradient).all()):
        raise RuntimeError('Invalid detail projection gradient')
    magnitude = float(gradient.abs().sum())
    if magnitude <= 0:
        raise RuntimeError('Detail projection receives no training signal')
    frozen.verify()
    model.clear_gradients()
    return magnitude


def evaluate(model, loader, dataset, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    metric = COCOMetric(anno_file=dataset.get_anno(),
                        clsid2catid={value: key for key, value in
                                     dataset.catid2clsid.items()},
                        output_eval=str(output_dir))
    model.eval()
    count = 0
    start = time.time()
    with paddle.no_grad():
        for data in loader:
            prediction = model(data)
            if not bool(paddle.isfinite(prediction['bbox']).all()):
                raise RuntimeError('Non-finite evaluation prediction')
            metric.update(data, prediction)
            count += int(data['image'].shape[0])
    metric.accumulate()
    stats = metric.get_results()['bbox']
    result = dict(zip(('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL'),
                      [float(value) for value in stats[:6]]))
    result.update(image_count=count, seconds=time.time() - start)
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    return result


def write_results(output_dir, rows, protocol):
    with (output_dir / 'epoch_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'protocol': protocol, 'epochs': rows}
    candidates = rows[1:]
    summary['continuation_signal'] = any(
        row['delta_APS'] >= 0.003 and row['delta_APL'] >= -0.005 and
        row['delta_AP'] >= 0 for row in candidates)
    summary['scope'] = ('Frozen-baseline screening only. Not a full-training '
                        'result or a calibrated probability of success.')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))


def main():
    parser = ArgsParser()
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--epochs', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    merge_config(args.opt)
    merge_config({'worker_num': 0, 'use_ema': False,
                  'TrainReader': {'batch_size': args.batch_size,
                                  'use_shared_memory': False},
                  'EvalReader': {'batch_size': 1, 'use_shared_memory': False}})
    if args.smoke_only:
        for transform in cfg.TrainReader['batch_transforms']:
            if 'BatchRandomResize' in transform:
                transform['BatchRandomResize'].update(
                    target_size=[800, 800], random_size=False)
    if args.batch_size < 1 or args.lr <= 0:
        raise ValueError('Batch size and learning rate must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    set_logger_output(str(args.output_dir / 'probe.log'))
    paddle.set_device(args.device)
    set_random_seed(args.seed)
    model = create(cfg.architecture)
    model.load_meanstd(cfg.TestReader['sample_transforms'])
    loaded = load_baseline(model, args.baseline)
    frozen = FrozenBaseline(model)
    dataset = create('TrainDataset')()
    train_loader = create('TrainReader')(dataset, 0)
    eval_dataset = create('EvalDataset')()
    eval_loader = create('EvalReader')(eval_dataset, 0)
    max_error = check_equivalence(model, next(eval_loader))
    eval_loader.loader = iter(eval_loader.dataloader)
    gradient = check_gradients(model, frozen, next(train_loader))
    protocol = {'baseline': str(args.baseline.resolve()),
                'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                'original_state_sha256': frozen.original_digest,
                'loaded_original_tensors': loaded,
                'trainable_parameters': sum(int(p.numel()) for p in frozen.parameters),
                'locked_batchnorm_layers': len(frozen.norms),
                'zero_init_max_prediction_error': max_error,
                'initial_projection_gradient_l1': gradient,
                'seed': args.seed, 'epochs': args.epochs,
                'batch_size': args.batch_size, 'lr': args.lr,
                'amp_training': args.amp, 'evaluation': 'FP32, Q300',
                'smoke_only': args.smoke_only}
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('[FrozenProbeAudit] {}'.format(json.dumps(protocol)), flush=True)
    if args.smoke_only:
        projection_before = model.transformer.p2_query_detail.output_proj.weight.clone()
        optimizer = paddle.optimizer.AdamW(
            learning_rate=args.lr, parameters=frozen.parameters)
        scaler = paddle.amp.GradScaler(
            enable=args.amp and args.device == 'gpu', init_loss_scaling=128.)
        for _ in range(2):
            model.train()
            data = next(train_loader)
            data['epoch_id'] = 40
            with paddle.amp.auto_cast(enable=args.amp and args.device == 'gpu'):
                loss = model(data)['loss']
            if not bool(paddle.isfinite(loss)):
                raise RuntimeError('Non-finite smoke loss')
            if args.amp and args.device == 'gpu':
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.minimize(optimizer, scaled_loss)
            else:
                loss.backward()
                optimizer.step()
            optimizer.clear_grad()
        frozen.verify()
        projection_delta = float((
            model.transformer.p2_query_detail.output_proj.weight -
            projection_before).abs().max())
        if projection_delta <= 0:
            raise RuntimeError('Smoke optimizer did not update the new projection')
        protocol.update(smoke_passed=True, projection_max_update=projection_delta,
                        frozen_state_verified=True,
                        max_allocated_mb=paddle.device.cuda.max_memory_allocated() / 1024**2
                        if args.device == 'gpu' else 0.)
        (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        print('Frozen probe smoke passed at 800 px after two optimizer updates.',
              flush=True)
        return
    set_random_seed(args.seed)
    train_loader.loader = iter(train_loader.dataloader)
    rows = []
    baseline = evaluate(model, eval_loader, eval_dataset, args.output_dir / 'eval_00')
    print('[FrozenProbeBaseline] {}'.format(json.dumps(baseline)), flush=True)
    rows.append(dict(epoch=0, **baseline, delta_AP=0., delta_APS=0., delta_APL=0.))
    write_results(args.output_dir, rows, protocol)
    warmup = paddle.optimizer.lr.LinearWarmup(
        learning_rate=args.lr, warmup_steps=200,
        start_lr=args.lr * 0.001, end_lr=args.lr)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=warmup, parameters=frozen.parameters,
        weight_decay=0.0001, grad_clip=nn.ClipGradByGlobalNorm(0.1))
    scaler = paddle.amp.GradScaler(enable=args.amp and args.device == 'gpu',
                                  init_loss_scaling=128.)
    best_ap = baseline['AP']
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(40 + epoch - 1)
        model.train()
        started = time.time()
        for step, data in enumerate(train_loader):
            data['epoch_id'] = 40 + epoch - 1
            with paddle.amp.auto_cast(enable=args.amp and args.device == 'gpu'):
                losses = model(data)
                loss = losses['loss']
            if not bool(paddle.isfinite(loss)):
                raise RuntimeError('Non-finite training loss')
            if args.amp and args.device == 'gpu':
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.minimize(optimizer, scaled_loss)
            else:
                loss.backward()
                optimizer.step()
            optimizer.clear_grad()
            warmup.step()
            if step % 100 == 0:
                elapsed = time.time() - started
                eta = elapsed / (step + 1) * (len(train_loader) - step - 1)
                print('[FrozenProbe] epoch={}/{} step={}/{} loss={:.6f} '
                      'residual_ratio={:.6f} mean_gate={:.6f} eta_seconds={:.0f} '
                      'max_allocated_mb={:.0f}'.format(
                          epoch, args.epochs, step, len(train_loader), float(loss),
                          float(losses['log_p2_detail_residual_ratio']),
                          float(losses['log_p2_detail_mean_gate']), eta,
                          paddle.device.cuda.max_memory_allocated() / 1024**2
                          if args.device == 'gpu' else 0.), flush=True)
        frozen.verify()
        paddle.save(model.state_dict(), str(args.output_dir / ('epoch_{}.pdparams'.format(epoch))))
        paddle.save(optimizer.state_dict(), str(args.output_dir / 'probe.pdopt'))
        metrics = evaluate(model, eval_loader, eval_dataset,
                           args.output_dir / ('eval_{:02d}'.format(epoch)))
        rows.append(dict(epoch=epoch, **metrics,
                         delta_AP=metrics['AP'] - baseline['AP'],
                         delta_APS=metrics['APS'] - baseline['APS'],
                         delta_APL=metrics['APL'] - baseline['APL']))
        write_results(args.output_dir, rows, protocol)
        if metrics['AP'] > best_ap:
            best_ap = metrics['AP']
            paddle.save(model.state_dict(), str(args.output_dir / 'best_model.pdparams'))
        print('[FrozenProbeEval] {}'.format(json.dumps(rows[-1])), flush=True)
    paddle.save(model.state_dict(), str(args.output_dir / 'model_final.pdparams'))
    frozen.verify()
    print('Frozen P2 detail probe completed; original state unchanged.', flush=True)


if __name__ == '__main__':
    main()
