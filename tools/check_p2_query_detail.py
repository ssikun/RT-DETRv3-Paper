import argparse
import copy
import os
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle
import paddle.nn as nn

from ppdet.core.workspace import create, load_config, merge_config
from ppdet.modeling.transformers.rtdetr_transformerv3 import (
    P2QueryDetailEnhancer, RTDETRTransformerv3)
from ppdet.modeling.transformers.utils import inverse_sigmoid


def check_local_detail():
    module = P2QueryDetailEnhancer(
        16, 8, detail_dim=8, num_heads=2, full_size=16., max_size=48.)
    query = paddle.randn([1, 3, 16])
    p2 = paddle.randn([1, 8, 16, 32])
    p2.stop_gradient = False
    boxes = paddle.to_tensor([[[0.5, 0.5, 16 / 128, 16 / 64],
                               [0.5, 0.5, 32 / 128, 32 / 64],
                               [0.5, 0.5, 64 / 128, 64 / 64]]])
    boxes.stop_gradient = False
    memory, shape = module.encode_p2(p2)
    gate = module.scale_gate(boxes, shape)
    assert paddle.allclose(gate.flatten(), paddle.to_tensor([1., 0.5, 0.]))
    output = module(query, boxes, memory, shape)
    assert paddle.equal_all(output, query)

    nn.initializer.XavierUniform()(module.output_proj.weight)
    output = module(query, boxes, memory, shape)
    assert float((output[:, :2] - query[:, :2]).abs().sum()) > 0.
    assert paddle.equal_all(output[:, 2], query[:, 2])
    output[:, :2].square().mean().backward()
    assert float(p2.grad.abs().sum()) > 0.
    assert boxes.grad is None

    p2.clear_gradient()
    memory, shape = module.encode_p2(p2)
    module(query, boxes, memory, shape)[:, 2].square().mean().backward()
    assert float(p2.grad.abs().sum()) == 0.
    print('Detail: identity initialization, scale gate, and gradients passed.')


def check_decoder():
    kwargs = dict(
        num_classes=3, hidden_dim=32, nhead=4, num_queries=5,
        backbone_feat_channels=[32, 32, 32], feat_strides=[8, 16, 32],
        num_decoder_layers=2, dim_feedforward=64, learnt_init_query=False,
        num_noises=0, num_noise_queries=[], o2m_branch=False, eval_size=None)
    baseline = RTDETRTransformerv3(**copy.deepcopy(kwargs))
    enhanced = RTDETRTransformerv3(
        **copy.deepcopy(kwargs), use_p2_query_detail=True,
        p2_feat_channels=8, p2_detail_dim=8, p2_detail_heads=2)
    state = enhanced.state_dict()
    state.update(baseline.state_dict())
    enhanced.set_state_dict(state)
    baseline.eval()
    enhanced.eval()
    feats = [paddle.randn([1, 32, size, size]) for size in (8, 4, 2)]
    p2 = paddle.randn([1, 8, 16, 16])
    with paddle.no_grad():
        base_outputs = baseline(feats)
        detail_outputs = enhanced(feats, p2_feat=p2)
    for original, candidate in zip(base_outputs[:4], detail_outputs[:4]):
        assert paddle.allclose(original, candidate, atol=1e-6, rtol=1e-6)

    nn.initializer.XavierUniform()(enhanced.p2_query_detail.output_proj.weight)
    baseline.decoder.train()
    enhanced.decoder.train()
    memory, shapes, starts = baseline._get_encoder_input(feats)
    detail_memory, detail_shape = enhanced.p2_query_detail.encode_p2(p2)
    query = paddle.randn([1, 2, 32])
    refs = paddle.to_tensor([[[0.5, 0.5, 0.15, 0.15],
                              [0.5, 0.5, 0.9, 0.9]]])
    args = (query, inverse_sigmoid(refs), memory, shapes, starts,
            baseline.dec_bbox_head, baseline.dec_score_head,
            baseline.query_pos_head)
    original = baseline.decoder(*args)
    candidate = enhanced.decoder(
        *args, p2_detail_enhancer=enhanced.p2_query_detail,
        p2_memory=detail_memory, p2_spatial_shape=detail_shape)
    assert paddle.allclose(original[1][0], candidate[1][0])
    assert paddle.allclose(original[1][-1, :, 1], candidate[1][-1, :, 1])
    assert float((original[1][-1, :, 0] - candidate[1][-1, :, 0]).abs().sum()) > 0.
    print('Decoder: original candidates and earlier/large-query outputs passed.')


def check_full_model():
    cfg = load_config(
        'configs/rtdetrv3/rtdetrv3_r18vd_visdrone_p2_query_detail.yml')
    merge_config({
        'eval_size': None, 'o2m_branch': False,
        'RTDETRTransformerv3': {
            'num_queries': 10, 'num_noises': 0, 'num_noise_queries': [],
            'num_denoising': 2
        },
        'DETRPostProcess': {'num_top_queries': 10}
    })
    model = create(cfg.architecture)
    assert model.exclude_p2_from_neck
    assert model.neck.p2_enhancer is None
    assert model.p2_center_aux_head is None
    assert [shape.stride for shape in model.neck.out_shape] == [8, 16, 32]
    inputs = {
        'image': paddle.randn([1, 3, 128, 128]),
        'im_shape': paddle.to_tensor([[128., 128.]]),
        'scale_factor': paddle.ones([1, 2]),
        'gt_bbox': [paddle.to_tensor([[0.25, 0.25, 0.125, 0.125]])],
        'gt_class': [paddle.to_tensor([[0]], dtype='int32')],
        'origin_gt_bbox': paddle.to_tensor([[[24., 24., 40., 40.]]]),
        'origin_gt_class': paddle.to_tensor([[[0]]], dtype='int32'),
        'pad_origin_gt_mask': paddle.ones([1, 1, 1]),
        'epoch_id': 0
    }
    model.train()
    losses = model(inputs)
    assert bool(paddle.isfinite(losses['loss']).all())
    losses['loss'].backward()
    grad = model.transformer.p2_query_detail.output_proj.weight.grad
    assert grad is not None and bool(paddle.isfinite(grad).all())
    assert float(grad.abs().sum()) > 0.
    model.eval()
    with paddle.no_grad():
        predictions = model(inputs)
    assert bool(paddle.isfinite(predictions['bbox']).all())
    print('Full model: training loss/backward and inference passed.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()
    paddle.set_device(args.device)
    paddle.seed(3407)
    check_local_detail()
    check_decoder()
    check_full_model()
    print('All P2 query detail checks passed.')


if __name__ == '__main__':
    main()
