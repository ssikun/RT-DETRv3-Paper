import os
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle
import paddle.nn.functional as F

from ppdet.modeling.heads.ppyoloe_head import PPYOLOESmallObjectIgnoreHead


def check_case(name, assigned_scores, raw_ignore, expected):
    positive, ignore, negative, overlap, valid = \
        PPYOLOESmallObjectIgnoreHead._build_classification_masks(
            assigned_scores, raw_ignore)
    counts = {
        'positive': int(positive.astype('int64').sum()),
        'ignore': int(ignore.astype('int64').sum()),
        'negative': int(negative.astype('int64').sum()),
        'overlap': int(overlap.astype('int64').sum()),
        'all': int(positive.shape[0] * positive.shape[1])
    }
    assert counts == expected, '{}: {} != {}'.format(name, counts, expected)
    assert not bool(paddle.logical_and(positive, ignore).any())
    assert counts['positive'] + counts['ignore'] + counts['negative'] == \
        counts['all']

    labels = paddle.full(positive.shape, 2, dtype='int64')
    labels = paddle.where(positive, paddle.zeros_like(labels), labels)
    one_hot = F.one_hot(labels, 3)[..., :-1]
    pred_scores = paddle.full(assigned_scores.shape, 0.2, dtype='float32')
    pred_scores.stop_gradient = False
    loss = PPYOLOESmallObjectIgnoreHead._masked_varifocal_loss(
        pred_scores, assigned_scores, one_hot, valid)
    loss.backward()
    grad_per_location = pred_scores.grad.abs().sum(axis=-1)

    ignore_grad = float(
        (grad_per_location * ignore.astype(grad_per_location.dtype)).sum())
    positive_grad = float(
        (grad_per_location * positive.astype(grad_per_location.dtype)).sum())
    negative_grad = float(
        (grad_per_location * negative.astype(grad_per_location.dtype)).sum())
    assert ignore_grad == 0.0, '{} ignore gradient: {}'.format(
        name, ignore_grad)
    if counts['positive']:
        assert positive_grad > 0.0
    if counts['negative']:
        assert negative_grad > 0.0

    print('{}: counts={}, gradients=(positive={:.6f}, ignore={:.6f}, '
          'negative={:.6f})'.format(name, counts, positive_grad, ignore_grad,
                                    negative_grad))


def main():
    points = paddle.to_tensor(
        [[4., 4.], [12., 4.], [20., 4.], [4., 12.], [12., 12.],
         [20., 12.], [4., 20.], [12., 20.], [20., 20.]]) / 8.
    strides = paddle.full([9, 1], 8., dtype='float32')
    large_box = paddle.to_tensor([[[0., 0., 16., 16.]]])
    large_gt_mask = paddle.ones([1, 1, 1], dtype='float32')
    raw_ignore = PPYOLOESmallObjectIgnoreHead \
        ._build_medium_large_ignore_mask(points, strides, large_box,
                                         large_gt_mask)
    no_ignore = paddle.zeros_like(raw_ignore)

    no_positive = paddle.zeros([1, 9, 2], dtype='float32')
    one_positive = no_positive.clone()
    one_positive[0, 0, 0] = 0.8

    check_case('only_small', one_positive, no_ignore, {
        'positive': 1,
        'ignore': 0,
        'negative': 8,
        'overlap': 0,
        'all': 9
    })
    check_case('only_large', no_positive, raw_ignore, {
        'positive': 0,
        'ignore': 4,
        'negative': 5,
        'overlap': 0,
        'all': 9
    })
    check_case('small_and_large', one_positive, raw_ignore, {
        'positive': 1,
        'ignore': 3,
        'negative': 5,
        'overlap': 1,
        'all': 9
    })
    print('All small-object three-state supervision checks passed.')


if __name__ == '__main__':
    main()
