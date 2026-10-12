import os
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle
import paddle.nn as nn

from ppdet.core.workspace import create, load_config


class FixedBackbone(nn.Layer):
    def __init__(self, feats):
        super().__init__()
        self.feats = feats

    def forward(self, inputs):
        return self.feats


class RecordingNeck(nn.Layer):
    def __init__(self):
        super().__init__()
        self.p2_enhancer = None
        self.input_count = None

    def forward(self, feats):
        self.input_count = len(feats)
        return feats


class DummyTransformer(nn.Layer):
    def forward(self, body_feats, pad_mask, inputs):
        return body_feats


class DummyDetrHead(nn.Layer):
    def forward(self, out_transformer, body_feats, inputs):
        return {'loss_main': body_feats[0].sum() * 0.}


def main():
    paddle.set_device('cpu')
    cfg = load_config(
        'configs/rtdetrv3/rtdetrv3_r18vd_visdrone_p2_center_only.yml')
    model = create(cfg.architecture)
    assert model.p2_center_aux_only
    assert model.p2_center_aux_head is not None
    assert not model.neck.use_p2_guidance
    assert model.neck.p2_enhancer is None
    assert [shape.stride for shape in model.backbone.out_shape] == [4, 8, 16, 32]
    assert [shape.stride for shape in model.neck.out_shape] == [8, 16, 32]

    p2_channels = model.backbone.out_shape[0].channels
    p2 = paddle.randn([1, p2_channels, 16, 16])
    p2.stop_gradient = False
    targets = {
        'origin_gt_bbox': paddle.to_tensor([[[8., 8., 16., 16.]]]),
        'origin_gt_class': paddle.to_tensor([[[0]]], dtype='int32'),
        'pad_origin_gt_mask': paddle.ones([1, 1, 1])
    }
    center_loss = model.p2_center_aux_head(p2, targets)['loss']
    center_loss.backward()
    assert float(p2.grad.abs().sum()) > 0.

    model.neck.eval()
    neck_feats = [
        paddle.randn([1, shape.channels, 64 // shape.stride,
                      64 // shape.stride])
        for shape in model.backbone.out_shape[1:]
    ]
    with paddle.no_grad():
        outs = model.neck(neck_feats, is_teacher=True)
    assert len(outs) == 3
    assert [out.shape[2] for out in outs] == [8, 4, 2]

    recorder = RecordingNeck()
    model.backbone = FixedBackbone([p2] + neck_feats)
    model.neck = recorder
    model.transformer = DummyTransformer()
    model.detr_head = DummyDetrHead()
    model.aux_o2m_head = None
    model.inputs = targets
    model.train()
    losses = model._forward()
    assert recorder.input_count == 3
    assert float(losses['loss_p2_center_aux']) > 0.
    print('P2 center-only routing and gradient checks passed.')


if __name__ == '__main__':
    main()
