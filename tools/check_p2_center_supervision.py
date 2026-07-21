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
    print('All P2 center supervision checks passed.')


if __name__ == '__main__':
    main()
