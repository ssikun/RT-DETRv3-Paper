# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['RTDETRV3']
# Deformable DETR, DINO use the same architecture as DETR


@register
class RTDETRV3(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process', 'post_process_semi']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 aux_o2m_head=None,
                 small_object_aux_head=None,
                 small_object_aux_loss_weight=0.5,
                 small_object_max_size=32.0,
                 post_process='DETRPostProcess',
                 post_process_semi=None,
                 with_mask=False,
                 exclude_post_process=False):
        super(RTDETRV3, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.aux_o2m_head = aux_o2m_head
        self.small_object_aux_head = small_object_aux_head
        self.small_object_aux_loss_weight = small_object_aux_loss_weight
        self.small_object_max_size = small_object_max_size
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process
        self.post_process_semi = post_process_semi

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        aux_o2m_head = create(cfg['aux_o2m_head'], **kwargs)

        small_object_aux_head = None
        if cfg.get('small_object_aux_head', None):
            kwargs = {'input_shape': neck.out_shape[:1]}
            small_object_aux_head = create(
                cfg['small_object_aux_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck,
            "aux_o2m_head": aux_o2m_head,
            "small_object_aux_head": small_object_aux_head
        }

    def _get_small_object_targets(self):
        required = ['origin_gt_bbox', 'pad_origin_gt_mask']
        missing = [key for key in required if key not in self.inputs]
        if missing:
            raise KeyError(
                'Small-object auxiliary supervision requires {}. Enable '
                'NormalizeBox.retain_origin_box and PadGT.only_origin_box.'.
                format(', '.join(missing)))

        boxes = self.inputs['origin_gt_bbox']
        valid_mask = self.inputs['pad_origin_gt_mask']
        widths = paddle.clip(boxes[:, :, 2] - boxes[:, :, 0], min=0.)
        heights = paddle.clip(boxes[:, :, 3] - boxes[:, :, 1], min=0.)
        areas = widths * heights
        max_area = self.small_object_max_size**2
        valid_bool = valid_mask.squeeze(-1).astype('bool')
        small_mask = paddle.logical_and(areas > 0., areas <= max_area)
        small_mask = paddle.logical_and(small_mask, valid_bool)
        medium_large_mask = paddle.logical_and(areas > max_area, valid_bool)

        targets = dict(self.inputs)
        targets['pad_origin_gt_mask'] = small_mask.unsqueeze(-1).astype(
            valid_mask.dtype)
        targets['small_aux_ignore_gt_mask'] = \
            medium_large_mask.unsqueeze(-1).astype(valid_mask.dtype)
        return targets

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            if self.aux_o2m_head is not None:
                aux_o2m_losses = self.aux_o2m_head(body_feats, self.inputs)
                for k, v in aux_o2m_losses.items():
                    if k == 'loss':
                        detr_losses[k] += v
                    k = k + '_aux_o2m'
                    detr_losses[k] = v
            if self.small_object_aux_head is not None:
                small_targets = self._get_small_object_targets()
                small_aux_losses = self.small_object_aux_head(
                    body_feats[:1], small_targets)
                for k, v in small_aux_losses.items():
                    if k.startswith('log_'):
                        detr_losses[k + '_small_aux'] = v.detach()
                        continue
                    weighted_loss = v * self.small_object_aux_loss_weight
                    if k == 'loss':
                        detr_losses[k] += weighted_loss
                    detr_losses[k + '_small_aux'] = weighted_loss
            if self.neck is not None and getattr(
                    self.neck, 'p2_enhancer', None) is not None:
                detr_losses['log_p2_alpha'] = \
                    self.neck.p2_enhancer.alpha.detach()
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    self.inputs['image'][2:].shape)

                # aux_o2m_outs = self.aux_o2m_head(body_feats)
                # bbox, bbox_num, nms_keep_idx = self.aux_o2m_head.post_process(
                #         aux_o2m_outs, self.inputs['scale_factor'])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
