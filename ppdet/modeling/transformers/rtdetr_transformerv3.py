# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from ..heads.detr_head import MLP
from .deformable_transformer import MSDeformableAttention
from ..initializer import (linear_init_, constant_, xavier_uniform_, normal_,
                           bias_init_with_prob)
from .utils import (_get_clones, get_sine_pos_embed,
                    get_contrastive_denoising_training_group, inverse_sigmoid)

__all__ = ['RTDETRTransformerv3']


class PPMSDeformableAttention(MSDeformableAttention):
    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:
            value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
            value_level_start_index = paddle.to_tensor(value_level_start_index)
            output = self.ms_deformable_attn_core(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


class P2QueryDetailEnhancer(nn.Layer):
    """Read stride-4 detail for small queries after the final decoder layer."""

    def __init__(self, hidden_dim, p2_channels, detail_dim=64, num_heads=4,
                 num_points=4, stride=4, full_size=16., max_size=48.):
        super().__init__()
        if not 0 < full_size < max_size:
            raise ValueError('P2 detail sizes must satisfy 0 < full < max')
        if detail_dim <= 0 or num_heads <= 0 or num_points <= 0:
            raise ValueError('P2 detail dimensions and sampling must be positive')
        if detail_dim % num_heads:
            raise ValueError('P2 detail dimension must divide into heads')
        self.stride = stride
        self.full_size = full_size
        self.max_size = max_size
        groups = min(32, detail_dim)
        while detail_dim % groups:
            groups //= 2
        self.p2_proj = nn.Sequential(
            nn.Conv2D(p2_channels, detail_dim, 1, bias_attr=False),
            nn.GroupNorm(groups, detail_dim), nn.Silu())
        self.query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, detail_dim))
        self.attention = PPMSDeformableAttention(
            detail_dim, num_heads, 1, num_points, 1.0)
        zero = nn.initializer.Constant(0.)
        self.output_proj = nn.Linear(
            detail_dim, hidden_dim, weight_attr=ParamAttr(initializer=zero),
            bias_attr=ParamAttr(initializer=zero))

    def encode_p2(self, p2_feat):
        with paddle.amp.auto_cast(enable=False):
            detail = self.p2_proj(p2_feat.astype('float32'))
        return detail.flatten(2).transpose([0, 2, 1]), detail.shape[2:]

    def scale_gate(self, reference_boxes, spatial_shape):
        height, width = spatial_shape
        wh = paddle.clip(reference_boxes[..., 2:].astype('float32'), min=0.)
        size = paddle.sqrt(wh[..., 0] * wh[..., 1] *
                           (height * width * self.stride**2))
        return paddle.clip(
            (self.max_size - size) / (self.max_size - self.full_size),
            min=0., max=1.).unsqueeze(-1)

    def forward(self, query, reference_boxes, p2_memory, spatial_shape):
        # Geometry controls sampling and scale without optimizing the gate.
        reference_boxes = reference_boxes.detach().astype('float32')
        with paddle.amp.auto_cast(enable=False):
            query_fp32 = query.astype('float32')
            gate = self.scale_gate(reference_boxes, spatial_shape)
            detail = self.attention(
                self.query_proj(query_fp32), reference_boxes.unsqueeze(2),
                p2_memory, [list(spatial_shape)], [0])
            residual = gate * self.output_proj(detail)
            self.last_gate = gate.detach()
            self.last_residual_ratio = (
                residual.square().mean().sqrt() /
                (query_fp32.square().mean().sqrt() + 1e-6)).detach()
        return query + residual.astype(query.dtype)


class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = PPMSDeformableAttention(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False,
                p2_detail_enhancer=None,
                p2_memory=None,
                p2_spatial_shape=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(
                    inverse_sigmoid(ref_points_detach))

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            if p2_detail_enhancer is not None and i == self.num_layers - 1:
                detail_boxes = F.sigmoid(
                    bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
                output = p2_detail_enhancer(
                    output, detail_boxes, p2_memory, p2_spatial_shape)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)


@register
class RTDETRTransformerv3(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size',
                  'o2m_branch', 'num_queries_o2m']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 query_pos_head_inv_sig=False,
                 eval_size=None,
                 eval_idx=-1,
                 num_noises=0,
                 num_noise_queries=[],
                 num_noise_denoising=100,
                 o2m_branch=False,
                 num_queries_o2m=450,
                 eps=1e-2,
                 use_small_object_query_selection=False,
                 soqs_lambda=0.05,
                 soqs_tau=0.04,
                 soqs_level_weights=[1.0, 0.5, 0.25],
                 soqs_apply_all_groups=False,
                 use_p2_query_detail=False,
                 p2_feat_channels=None,
                 p2_detail_dim=64,
                 p2_detail_heads=4,
                 p2_detail_points=4,
                 p2_detail_stride=4,
                 p2_detail_full_size=16.,
                 p2_detail_max_size=48.):
        super(RTDETRTransformerv3, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        assert len(num_noise_queries) == num_noises
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = [num_queries]
        self.eps = eps
        self.use_small_object_query_selection = use_small_object_query_selection
        self.soqs_lambda = soqs_lambda
        self.soqs_tau = soqs_tau
        self.soqs_level_weights = soqs_level_weights
        self.soqs_apply_all_groups = soqs_apply_all_groups
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.use_p2_query_detail = use_p2_query_detail

        self.num_noises = num_noises
        self.num_noise_denoising = num_noise_denoising
        self.num_groups = 1
        if num_noises > 0:
            self.num_queries.extend(num_noise_queries)
            self.num_groups += num_noises
        
        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m
        if o2m_branch:
            self.num_queries.append(num_queries_o2m)
            self.num_groups += 1

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig

        # encoder head
        self.enc_output = nn.LayerList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(
                    hidden_dim,
                    weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            for _ in range(self.num_groups)
        ])
        self.enc_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(self.num_groups)
        ])
        self.enc_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(self.num_groups)
        ])

        self.map_memory = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
            )

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()
        self.p2_query_detail = None
        if self.use_p2_query_detail:
            if p2_feat_channels is None:
                raise ValueError('P2 query detail requires a stride-4 input')
            if self.decoder.eval_idx != num_decoder_layers - 1:
                raise ValueError('P2 query detail requires the final eval layer')
            self.p2_query_detail = P2QueryDetailEnhancer(
                hidden_dim, p2_feat_channels, p2_detail_dim, p2_detail_heads,
                p2_detail_points, p2_detail_stride, p2_detail_full_size,
                p2_detail_max_size)

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        for enc_score_head in self.enc_score_head:
            linear_init_(enc_score_head)
            constant_(enc_score_head.bias, bias_cls)
        for enc_bbox_head in self.enc_bbox_head:
            constant_(enc_bbox_head.layers[-1].weight)
            constant_(enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        for enc_output in self.enc_output:
            linear_init_(enc_output[0])
            xavier_uniform_(enc_output[0].weight)
        linear_init_(self.map_memory[0])
        xavier_uniform_(self.map_memory[0].weight)

        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape, p2_input_shape=None):
        kwargs = {'backbone_feat_channels': [i.channels for i in input_shape]}
        if cfg.get('use_p2_query_detail', False):
            if p2_input_shape is None or p2_input_shape.stride != 4:
                raise ValueError('P2 query detail requires a stride-4 input')
            kwargs.update({
                'p2_feat_channels': p2_input_shape.channels,
                'p2_detail_stride': p2_input_shape.stride
            })
        return kwargs

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def forward(self, feats, pad_mask=None, gt_meta=None, is_teacher=False,
                p2_feat=None):
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(feats)
        p2_memory, p2_spatial_shape = None, None
        if self.use_p2_query_detail:
            if p2_feat is None:
                raise ValueError('P2 query detail is enabled but P2 is missing')
            p2_memory, p2_spatial_shape = \
                self.p2_query_detail.encode_p2(p2_feat)

        # prepare denoising training
        if self.training:
            denoising_classes, denoising_bbox_unacts, attn_masks, dn_metas = [], [], [], []
            for g_id in range(self.num_noises + 1):
                if g_id == 0:
                    num_denoising = self.num_denoising
                else:
                    num_denoising = self.num_noise_denoising
                denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group(gt_meta,
                                                self.num_classes,
                                                self.num_queries[g_id],
                                                self.denoising_class_embed.weight,
                                                num_denoising,
                                                self.label_noise_ratio,
                                                self.box_noise_scale)
                denoising_classes.append(denoising_class)
                denoising_bbox_unacts.append(denoising_bbox_unact)
                attn_masks.append(attn_mask)
                dn_metas.append(dn_meta)
        else:
            denoising_classes, denoising_bbox_unacts, attn_masks, dn_metas = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
                memory, spatial_shapes, denoising_classes, denoising_bbox_unacts, is_teacher)

        # multi group noise attention
        if self.training:
            new_size = target.shape[1]
            new_attn_mask = paddle.ones([new_size, new_size]) < 0
            begin, end = 0, 0
            mask = None
            for g_id in range(self.num_groups):
                new_mask = paddle.rand([self.num_queries[g_id], self.num_queries[g_id]])
                if self.o2m_branch and g_id == self.num_groups - 1:
                    end = end + self.num_queries_o2m
                    new_mask = new_mask >= 0.0
                    new_attn_mask[begin: end, begin: end] = new_mask
                else:
                    end = end + attn_masks[g_id].shape[1]
                    dn_size, q_size = dn_metas[g_id]['dn_num_split']
                    if g_id > 0:
                        new_mask = new_mask > 0.1
                    else:
                        new_mask = new_mask >= 0.0
                    attn_masks[g_id][dn_size: dn_size + q_size, dn_size: dn_size + q_size] = new_mask
                    new_attn_mask[begin: end, begin: end] = attn_masks[g_id]
                begin = end
            attn_masks = new_attn_mask

        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_masks,
            memory_mask=None,
            query_pos_head_inv_sig=self.query_pos_head_inv_sig,
            p2_detail_enhancer=self.p2_query_detail
            if self.use_p2_query_detail else None,
            p2_memory=p2_memory,
            p2_spatial_shape=p2_spatial_shape)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_metas)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_small_object_selection_score(self,
                                          cls_score,
                                          bbox_unact,
                                          spatial_shapes,
                                          g_id):
        if not self.use_small_object_query_selection:
            return cls_score
        if not self.soqs_apply_all_groups and g_id != 0:
            return cls_score

        pred_bbox = F.sigmoid(bbox_unact.detach())
        pred_wh = paddle.clip(pred_bbox[..., 2:], min=1e-6)
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        small_prior = paddle.exp(-pred_area / self.soqs_tau)

        level_priors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            if lvl < len(self.soqs_level_weights):
                level_weight = self.soqs_level_weights[lvl]
            else:
                level_weight = self.soqs_level_weights[-1]
            level_priors.append(
                paddle.full(
                    [int(h) * int(w)], level_weight, dtype=cls_score.dtype))
        level_prior = paddle.concat(level_priors, axis=0).unsqueeze(0)
        small_score = small_prior * level_prior
        return cls_score + self.soqs_lambda * small_score

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_classes=None,
                           denoising_bbox_unacts=None,
                           is_teacher=False):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None or is_teacher:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        map_memory = self.map_memory(memory.detach())
        targets, reference_points_unacts, enc_topk_bboxes, enc_topk_logits = [], [], [], []

        for g_id in range(self.num_groups):
            output_memory = self.enc_output[g_id](memory)
            enc_outputs_class = self.enc_score_head[g_id](output_memory)
            enc_outputs_coord_unact = self.enc_bbox_head[g_id](output_memory) + anchors

            select_score = self._get_small_object_selection_score(
                enc_outputs_class.max(-1), enc_outputs_coord_unact,
                spatial_shapes, g_id)
            _, topk_ind = paddle.topk(
                select_score, self.num_queries[g_id], axis=1)
            # extract region proposal boxes
            batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
            batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries[g_id]])
            topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

            reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact, topk_ind)  # unsigmoided.
            enc_topk_bbox = F.sigmoid(reference_points_unact)
            enc_topk_logit = paddle.gather_nd(enc_outputs_class, topk_ind)

            if denoising_bbox_unacts is not None and not (self.o2m_branch and g_id == self.num_groups - 1):
                reference_points_unact = paddle.concat(
                    [denoising_bbox_unacts[g_id], reference_points_unact], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            # extract region features
            if self.learnt_init_query:
                target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
            else:
                if g_id == 0:
                    target = paddle.gather_nd(output_memory, topk_ind)
                    if self.training:
                        target = target.detach()
                else:
                    target = paddle.gather_nd(map_memory, topk_ind)
            if denoising_classes is not None and not (self.o2m_branch and g_id == self.num_groups - 1):
                target = paddle.concat([denoising_classes[g_id], target], 1)
            
            if not self.training:
                return target, reference_points_unact, enc_topk_bbox, enc_topk_logit
            
            targets.append(target)
            reference_points_unacts.append(reference_points_unact)
            enc_topk_bboxes.append(enc_topk_bbox)
            enc_topk_logits.append(enc_topk_logit)

        targets = paddle.concat(targets, 1)
        reference_points_unacts = paddle.concat(reference_points_unacts, 1)
        enc_topk_bboxes = paddle.concat(enc_topk_bboxes, 1)
        enc_topk_logits = paddle.concat(enc_topk_logits, 1)
        return targets, reference_points_unacts, enc_topk_bboxes, enc_topk_logits
