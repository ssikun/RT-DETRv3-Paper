import os
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle

from ppdet.modeling.heads.small_object_center_head import (
    P2SmallObjectCenterHead, )


def main():
    boxes = paddle.to_tensor([
        [[0., 0., 1., 1.], [0., 0., 0., 0.]],
        [[0., 0., 12., 12.], [0., 0., 0., 0.]],
        [[0., 0., 1., 1.], [0., 0., 12., 12.]],
    ])
    classes = paddle.to_tensor([
        [[0], [0]],
        [[1], [0]],
        [[0], [1]],
    ], dtype='int32')
    valid_mask = paddle.to_tensor([
        [[1.], [0.]],
        [[1.], [0.]],
        [[1.], [1.]],
    ])
    heatmap, raw_ignore, small_gt, medium_large_gt = \
        P2SmallObjectCenterHead._build_targets(
            boxes, classes, valid_mask, 4, 4, 4, 4., 2)
    positive_locations = heatmap.max(axis=1) > 0
    final_ignore = paddle.logical_and(raw_ignore, ~positive_locations)
    valid_locations = ~final_ignore
    valid_negative = paddle.logical_and(valid_locations,
                                        ~positive_locations)

    expected = [
        (1, 0, 15),
        (0, 9, 7),
        (1, 8, 7),
    ]
    for image_idx, expected_counts in enumerate(expected):
        counts = (
            int(positive_locations[image_idx].astype('int64').sum()),
            int(final_ignore[image_idx].astype('int64').sum()),
            int(valid_negative[image_idx].astype('int64').sum()),
        )
        assert counts == expected_counts, (counts, expected_counts)
        assert sum(counts) == 16
        print('image={}: positive={}, ignore={}, negative={}'.format(
            image_idx, *counts))

    assert small_gt.astype('int64').sum(axis=1).tolist() == [1, 0, 1]
    assert medium_large_gt.astype('int64').sum(axis=1).tolist() == [0, 1, 1]

    pred = paddle.full(heatmap.shape, 0.2, dtype='float32')
    pred.stop_gradient = False
    loss = P2SmallObjectCenterHead._center_focal_loss(
        pred, heatmap, valid_locations)
    loss.backward()
    location_grad = pred.grad.abs().sum(axis=1)
    ignore_grad = float(
        (location_grad * final_ignore.astype(location_grad.dtype)).sum())
    positive_grad = float(
        (location_grad * positive_locations.astype(
            location_grad.dtype)).sum())
    negative_grad = float(
        (location_grad * valid_negative.astype(
            location_grad.dtype)).sum())
    assert ignore_grad == 0.0
    assert positive_grad > 0.0
    assert negative_grad > 0.0
    print('gradients: positive={:.6f}, ignore={:.6f}, negative={:.6f}'.
          format(positive_grad, ignore_grad, negative_grad))

    localization_targets, regression_mask, regression_collisions = \
        P2SmallObjectCenterHead._build_localization_targets(
            boxes, small_gt, 4, 4, 4)
    assert regression_mask.astype('int64').sum(axis=[1, 2]).tolist() == [
        1, 0, 1
    ]
    assert regression_collisions.tolist() == [0., 0., 0.]
    expected_offset = [0.125, 0.125]
    actual_offset = localization_targets[0, :2, 0, 0].tolist()
    for actual, target in zip(actual_offset, expected_offset):
        assert abs(actual - target) < 1e-6, (actual_offset, expected_offset)

    offset_pred = paddle.full([3, 2, 4, 4], 0.5, dtype='float32')
    ltrb_pred = paddle.zeros([3, 4, 4, 4], dtype='float32')
    offset_pred.stop_gradient = False
    ltrb_pred.stop_gradient = False
    offset_loss, ltrb_loss = \
        P2SmallObjectCenterHead._localization_losses(
            offset_pred, ltrb_pred, localization_targets, regression_mask)
    (offset_loss + ltrb_loss).backward()
    regression_grad = (
        offset_pred.grad.abs().sum(axis=1) +
        ltrb_pred.grad.abs().sum(axis=1))
    positive_regression_grad = float(
        (regression_grad * regression_mask.astype(
            regression_grad.dtype)).sum())
    non_positive_regression_grad = float(
        (regression_grad * (~regression_mask).astype(
            regression_grad.dtype)).sum())
    assert positive_regression_grad > 0.
    assert non_positive_regression_grad == 0.
    print(
        'localization: offset_loss={:.6f}, ltrb_loss={:.6f}, '
        'positive_grad={:.6f}, non_positive_grad={:.6f}'.format(
            float(offset_loss), float(ltrb_loss),
            positive_regression_grad, non_positive_regression_grad))

    collision_boxes = paddle.to_tensor([[
        [0., 0., 2., 2.],
        [0.5, 0.5, 2.5, 2.5],
    ]])
    collision_small_gt = paddle.ones([1, 2], dtype='bool')
    _, collision_mask, collision_count = \
        P2SmallObjectCenterHead._build_localization_targets(
            collision_boxes, collision_small_gt, 4, 4, 4)
    assert int(collision_mask.astype('int64').sum()) == 1
    assert collision_count.tolist() == [1.]
    print('All P2 center supervision checks passed.')


if __name__ == '__main__':
    main()
