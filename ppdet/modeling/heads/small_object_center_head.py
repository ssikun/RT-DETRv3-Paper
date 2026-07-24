import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from ppdet.core.workspace import register

__all__ = ['P2SmallObjectCenterHead']


@register
class P2SmallObjectCenterHead(nn.Layer):
    """Training-only stride-4 center supervision for small objects."""

    __shared__ = ['num_classes']

    def __init__(self,
                 in_channels=64,
                 num_classes=80,
                 hidden_channels=64,
                 stride=4,
                 small_object_max_size=32.0,
                 focal_alpha=2.0,
                 use_localization=False,
                 localization_loss_weight=0.1):
        super().__init__()
        groups = min(32, hidden_channels)
        while hidden_channels % groups:
            groups //= 2
        self.num_classes = num_classes
        self.stride = stride
        self.small_object_max_size = small_object_max_size
        self.focal_alpha = focal_alpha
        self.use_localization = use_localization
        self.localization_loss_weight = localization_loss_weight
        self.stem = nn.Sequential(
            nn.Conv2D(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.Silu())
        prior_prob = 0.01
        bias = -math.log((1 - prior_prob) / prior_prob)
        self.pred = nn.Conv2D(
            hidden_channels,
            num_classes,
            kernel_size=1,
            bias_attr=ParamAttr(
                initializer=nn.initializer.Constant(bias)))
        if self.use_localization:
            zero = nn.initializer.Constant(0.)
            self.offset_pred = nn.Conv2D(
                hidden_channels,
                2,
                kernel_size=1,
                weight_attr=ParamAttr(initializer=zero),
                bias_attr=ParamAttr(initializer=zero))
            self.ltrb_pred = nn.Conv2D(
                hidden_channels,
                4,
                kernel_size=1,
                weight_attr=ParamAttr(initializer=zero),
                bias_attr=ParamAttr(initializer=zero))

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': input_shape[0].channels}

    @staticmethod
    def _build_targets(boxes, classes, valid_mask, height, width, stride,
                       small_object_max_size, num_classes):
        valid = valid_mask.squeeze(-1).astype('bool')
        box_widths = paddle.clip(boxes[..., 2] - boxes[..., 0], min=0.)
        box_heights = paddle.clip(boxes[..., 3] - boxes[..., 1], min=0.)
        areas = box_widths * box_heights
        max_area = small_object_max_size**2
        small_gt = paddle.logical_and(areas > 0., areas <= max_area)
        small_gt = paddle.logical_and(small_gt, valid)
        medium_large_gt = paddle.logical_and(areas > max_area, valid)

        center_x = paddle.floor(
            (boxes[..., 0] + boxes[..., 2]) * 0.5 / stride)
        center_y = paddle.floor(
            (boxes[..., 1] + boxes[..., 3]) * 0.5 / stride)
        center_x = paddle.clip(center_x, min=0, max=width - 1).astype('int64')
        center_y = paddle.clip(center_y, min=0, max=height - 1).astype('int64')

        batch_size = boxes.shape[0]
        heatmap = paddle.zeros(
            [batch_size, num_classes, height, width], dtype='float32')
        small_indices = paddle.nonzero(small_gt)
        if small_indices.shape[0] > 0:
            batch_indices = small_indices[:, 0]
            class_indices = paddle.gather_nd(
                classes.squeeze(-1), small_indices).astype('int64')
            x_indices = paddle.gather_nd(center_x, small_indices)
            y_indices = paddle.gather_nd(center_y, small_indices)
            scatter_indices = paddle.stack(
                [batch_indices, class_indices, y_indices, x_indices], axis=1)
            heatmap = paddle.scatter_nd(
                scatter_indices,
                paddle.ones([small_indices.shape[0]], dtype='float32'),
                heatmap.shape)
            heatmap = paddle.clip(heatmap, max=1.)

        shift_x = (paddle.arange(width, dtype='float32') + 0.5) * stride
        shift_y = (paddle.arange(height, dtype='float32') + 0.5) * stride
        shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
        points = paddle.stack([shift_x, shift_y], axis=-1).reshape(
            [1, 1, -1, 2])
        expanded_boxes = boxes.unsqueeze(2)
        inside = paddle.logical_and(
            points[..., 0] >= expanded_boxes[..., 0],
            points[..., 0] <= expanded_boxes[..., 2])
        inside = paddle.logical_and(
            inside,
            paddle.logical_and(
                points[..., 1] >= expanded_boxes[..., 1],
                points[..., 1] <= expanded_boxes[..., 3]))
        inside = paddle.logical_and(
            inside, medium_large_gt.unsqueeze(-1))
        raw_ignore = paddle.any(inside, axis=1).reshape(
            [batch_size, height, width])
        return heatmap, raw_ignore, small_gt, medium_large_gt

    @staticmethod
    def _center_focal_loss(pred, target, valid_locations, alpha=2.0):
        pred = paddle.clip(pred, min=1e-4, max=1 - 1e-4)
        positive = (target == 1).astype(pred.dtype)
        negative = (target < 1).astype(pred.dtype)
        valid = valid_locations.unsqueeze(1).astype(pred.dtype)
        positive_loss = -paddle.log(pred) * paddle.pow(
            1 - pred, alpha) * positive
        negative_loss = -paddle.log(1 - pred) * paddle.pow(
            pred, alpha) * negative * valid
        positive_count = paddle.clip(positive.sum(), min=1.)
        return (positive_loss.sum() + negative_loss.sum()) / positive_count

    @staticmethod
    def _build_localization_targets(boxes, small_gt, height, width, stride):
        batch_size = boxes.shape[0]
        targets = paddle.zeros(
            [batch_size, 6, height, width], dtype='float32')
        positive_mask = paddle.zeros(
            [batch_size, height, width], dtype='bool')
        small_indices = paddle.nonzero(small_gt)
        if small_indices.shape[0] == 0:
            collisions = small_gt.astype('float32').sum(axis=1)
            return targets, positive_mask, collisions

        selected_boxes = paddle.gather_nd(boxes, small_indices)
        center_x = (selected_boxes[:, 0] + selected_boxes[:, 2]) * 0.5
        center_y = (selected_boxes[:, 1] + selected_boxes[:, 3]) * 0.5
        grid_x = paddle.floor(center_x / stride)
        grid_y = paddle.floor(center_y / stride)
        grid_x = paddle.clip(
            grid_x, min=0, max=width - 1).astype('int64')
        grid_y = paddle.clip(
            grid_y, min=0, max=height - 1).astype('int64')

        cell_ids = (
            small_indices[:, 0] * height * width + grid_y * width + grid_x)
        order = paddle.argsort(cell_ids)
        sorted_ids = paddle.gather(cell_ids, order)
        first = paddle.concat([
            paddle.ones([1], dtype='bool'),
            sorted_ids[1:] != sorted_ids[:-1]
        ])
        selected_order = paddle.gather(
            order, paddle.nonzero(first).flatten())
        selected_indices = paddle.gather(small_indices, selected_order)
        selected_boxes = paddle.gather(selected_boxes, selected_order)
        grid_x = paddle.gather(grid_x, selected_order)
        grid_y = paddle.gather(grid_y, selected_order)

        center_x = (selected_boxes[:, 0] + selected_boxes[:, 2]) * 0.5
        center_y = (selected_boxes[:, 1] + selected_boxes[:, 3]) * 0.5
        center_x_feature = center_x / stride
        center_y_feature = center_y / stride
        offsets = paddle.stack([
            center_x_feature - paddle.floor(center_x_feature),
            center_y_feature - paddle.floor(center_y_feature)
        ],
                               axis=1)
        ltrb = paddle.stack([
            center_x - selected_boxes[:, 0],
            center_y - selected_boxes[:, 1],
            selected_boxes[:, 2] - center_x,
            selected_boxes[:, 3] - center_y
        ],
                            axis=1)
        ltrb = paddle.log1p(paddle.clip(ltrb / stride, min=0.))
        updates = paddle.concat([offsets, ltrb], axis=1).astype('float32')
        scatter_indices = paddle.stack(
            [selected_indices[:, 0], grid_y, grid_x], axis=1)
        targets = paddle.scatter_nd(
            scatter_indices, updates,
            [batch_size, height, width, 6]).transpose([0, 3, 1, 2])
        positive_mask = paddle.scatter_nd(
            scatter_indices,
            paddle.ones([scatter_indices.shape[0]], dtype='int32'),
            [batch_size, height, width]).astype('bool')
        small_count = small_gt.astype('float32').sum(axis=1)
        selected_count = positive_mask.astype('float32').sum(axis=[1, 2])
        collisions = paddle.clip(small_count - selected_count, min=0.)
        return targets, positive_mask, collisions

    @staticmethod
    def _localization_losses(offset_pred, ltrb_pred, targets,
                             positive_mask):
        mask = positive_mask.unsqueeze(1).astype(offset_pred.dtype)
        positive_count = paddle.clip(mask.sum(), min=1.)
        offset_loss = F.smooth_l1_loss(
            offset_pred, targets[:, :2], reduction='none')
        offset_loss = (
            offset_loss * mask).sum() / (positive_count * 2.)
        ltrb_loss = F.smooth_l1_loss(
            ltrb_pred, targets[:, 2:], reduction='none')
        ltrb_loss = (ltrb_loss * mask).sum() / (positive_count * 4.)
        return offset_loss, ltrb_loss

    def forward(self, p2_feat, targets):
        required = [
            'origin_gt_bbox', 'origin_gt_class', 'pad_origin_gt_mask'
        ]
        missing = [key for key in required if key not in targets]
        if missing:
            raise KeyError(
                'P2 center supervision requires {}.'.format(
                    ', '.join(missing)))

        features = self.stem(p2_feat)
        logits = self.pred(features)
        height, width = logits.shape[2:]
        heatmap, raw_ignore, small_gt, medium_large_gt = self._build_targets(
            targets['origin_gt_bbox'], targets['origin_gt_class'],
            targets['pad_origin_gt_mask'], height, width, self.stride,
            self.small_object_max_size, self.num_classes)
        positive_locations = heatmap.max(axis=1) > 0
        overlap = paddle.logical_and(positive_locations, raw_ignore)
        final_ignore = paddle.logical_and(raw_ignore, ~positive_locations)
        valid_locations = ~final_ignore
        valid_negative = paddle.logical_and(valid_locations,
                                            ~positive_locations)
        pred = F.sigmoid(logits)
        loss_cls = self._center_focal_loss(pred, heatmap, valid_locations,
                                           self.focal_alpha)
        loss = loss_cls
        localization_outputs = {}
        regression_collisions = None
        regression_locations = None
        if self.use_localization:
            localization_targets, regression_mask, regression_collisions = \
                self._build_localization_targets(
                    targets['origin_gt_bbox'], small_gt, height, width,
                    self.stride)
            offset_pred = F.sigmoid(self.offset_pred(features))
            ltrb_pred = self.ltrb_pred(features)
            raw_offset_loss, raw_ltrb_loss = self._localization_losses(
                offset_pred, ltrb_pred, localization_targets, regression_mask)
            loss_offset = (
                raw_offset_loss * self.localization_loss_weight * 0.5)
            loss_ltrb = raw_ltrb_loss * self.localization_loss_weight * 0.5
            loss = loss + loss_offset + loss_ltrb
            regression_locations = regression_mask.astype('float32').sum(
                axis=[1, 2])
            localization_outputs = {
                'loss_offset': loss_offset,
                'loss_ltrb': loss_ltrb,
                'log_raw_loss_offset': raw_offset_loss.detach(),
                'log_raw_loss_ltrb': raw_ltrb_loss.detach(),
                'log_regression_locations': regression_locations.sum(),
                'log_regression_collisions': regression_collisions.sum()
            }

        positive_centers = heatmap.sum(axis=[1, 2, 3])
        small_gt_count = small_gt.astype('float32').sum(axis=1)
        medium_large_gt_count = medium_large_gt.astype('float32').sum(axis=1)
        center_collisions = paddle.clip(
            small_gt_count - positive_centers, min=0.)
        all_locations = paddle.full(
            [p2_feat.shape[0]], height * width, dtype='int64')
        stats = {
            'small_gt': small_gt_count.detach(),
            'medium_large_gt': medium_large_gt_count.detach(),
            'positive_centers': positive_centers.detach(),
            'positive_locations':
            positive_locations.sum(axis=[1, 2]).detach(),
            'center_collisions': center_collisions.detach(),
            'ignore': final_ignore.sum(axis=[1, 2]).detach(),
            'valid_negative': valid_negative.sum(axis=[1, 2]).detach(),
            'raw_overlap': overlap.sum(axis=[1, 2]).detach(),
            'final_overlap': paddle.logical_and(
                positive_locations, final_ignore).sum(axis=[1, 2]).detach(),
            'all_locations': all_locations
        }
        if self.use_localization:
            stats.update({
                'regression_locations': regression_locations.detach(),
                'regression_collisions': regression_collisions.detach()
            })
        self.supervision_stats = stats
        self._print_stats_once(stats)

        outputs = {
            'loss': loss,
            'loss_cls': loss_cls,
            'log_small_gt': small_gt_count.sum(),
            'log_medium_large_gt': medium_large_gt_count.sum(),
            'log_positive_centers': positive_centers.sum(),
            'log_positive_per_small_gt':
            positive_centers.sum() / paddle.clip(small_gt_count.sum(), min=1.),
            'log_center_collisions': center_collisions.sum(),
            'log_ignore_locations': final_ignore.astype('float32').sum(),
            'log_valid_negative_locations':
            valid_negative.astype('float32').sum(),
            'log_positive_ignore_overlap': overlap.astype('float32').sum()
        }
        outputs.update(localization_outputs)
        return outputs

    def _print_stats_once(self, stats):
        if getattr(self, '_supervision_stats_printed', False):
            return
        values = {
            key: value.numpy().tolist()
            for key, value in stats.items()
        }
        batch_size = len(values['small_gt'])
        for image_idx in range(batch_size):
            positive = int(values['positive_locations'][image_idx])
            ignore = int(values['ignore'][image_idx])
            negative = int(values['valid_negative'][image_idx])
            all_locations = int(values['all_locations'][image_idx])
            assert positive + ignore + negative == all_locations
            assert int(values['final_overlap'][image_idx]) == 0
            localization = ''
            if 'regression_locations' in values:
                localization = (
                    ' regression_locations={} regression_collisions={}'.
                    format(
                        int(values['regression_locations'][image_idx]),
                        int(values['regression_collisions'][image_idx])))
            print(
                '[P2CenterStats] image={} small_gt={} medium_large_gt={} '
                'positive_centers={} positive_locations={} '
                'center_collisions={} ignore={} '
                'valid_negative={} all_locations={} raw_overlap={} '
                'final_overlap={}{}'.format(
                    image_idx, int(values['small_gt'][image_idx]),
                    int(values['medium_large_gt'][image_idx]),
                    int(values['positive_centers'][image_idx]), positive,
                    int(values['center_collisions'][image_idx]), ignore,
                    negative, all_locations,
                    int(values['raw_overlap'][image_idx]),
                    int(values['final_overlap'][image_idx]), localization))
        self._supervision_stats_printed = True
