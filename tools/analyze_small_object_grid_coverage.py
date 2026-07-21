import argparse
import json
import math
from collections import Counter, defaultdict


DEFAULT_TARGET_SIZES = [
    480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze grid coverage for resized small-object boxes.')
    parser.add_argument(
        '--annotation',
        default='dataset/visdrone2019/annotations/train.json')
    parser.add_argument(
        '--target-sizes',
        nargs='+',
        type=int,
        default=DEFAULT_TARGET_SIZES)
    parser.add_argument('--strides', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--small-max-size', type=float, default=32.0)
    parser.add_argument('--grid-offset', type=float, default=0.5)
    return parser.parse_args()


def contains_grid_center(box, stride, offset, target_size):
    x1, y1, x2, y2 = box
    min_x = math.ceil(x1 / stride - offset)
    max_x = math.floor(x2 / stride - offset)
    min_y = math.ceil(y1 / stride - offset)
    max_y = math.floor(y2 / stride - offset)
    grid_size = int(math.ceil(target_size / stride))
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, grid_size - 1)
    max_y = min(max_y, grid_size - 1)
    return min_x <= max_x and min_y <= max_y


def percentage(value, total):
    return 100.0 * value / total if total else 0.0


def new_stats(strides):
    return {
        stride: {
            'small': 0,
            'covered': 0,
            'collisions': 0
        }
        for stride in strides
    }


def analyze(data, target_sizes, strides, small_max_size, grid_offset):
    images = {image['id']: image for image in data['images']}
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        if annotation.get('ignore', 0):
            continue
        x, y, width, height = annotation['bbox']
        if width > 0 and height > 0:
            annotations[annotation['image_id']].append((x, y, width, height))

    size_weights = Counter(target_sizes)
    per_size = {}
    overall = new_stats(strides)
    max_area = small_max_size**2

    for target_size, sample_weight in sorted(size_weights.items()):
        stats = new_stats(strides)
        for image_id, image in images.items():
            image_width = image['width']
            image_height = image['height']
            scale_x = target_size / image_width
            scale_y = target_size / image_height
            small_boxes = []
            for x, y, width, height in annotations.get(image_id, []):
                x1 = x * scale_x
                y1 = y * scale_y
                x2 = (x + width) * scale_x
                y2 = (y + height) * scale_y
                if 0 < (x2 - x1) * (y2 - y1) <= max_area:
                    small_boxes.append((x1, y1, x2, y2))

            for stride in strides:
                cells = set()
                stats[stride]['small'] += len(small_boxes)
                for box in small_boxes:
                    if contains_grid_center(box, stride, grid_offset,
                                            target_size):
                        stats[stride]['covered'] += 1
                    center_x = (box[0] + box[2]) * 0.5
                    center_y = (box[1] + box[3]) * 0.5
                    cell_x = min(int(center_x // stride),
                                 int(math.ceil(target_size / stride)) - 1)
                    cell_y = min(int(center_y // stride),
                                 int(math.ceil(target_size / stride)) - 1)
                    cells.add((cell_x, cell_y))
                stats[stride]['collisions'] += len(small_boxes) - len(cells)

        per_size[target_size] = stats
        for stride in strides:
            for key in overall[stride]:
                overall[stride][key] += stats[stride][key] * sample_weight

    return per_size, overall, size_weights


def print_stats(label, stats, strides):
    print(label)
    print('stride  small_gt  center_covered  coverage  center_collisions  '
          'collision_rate')
    for stride in strides:
        values = stats[stride]
        print('{:>6}  {:>8}  {:>14}  {:>7.2f}%  {:>17}  {:>13.2f}%'.format(
            stride, values['small'], values['covered'],
            percentage(values['covered'], values['small']),
            values['collisions'],
            percentage(values['collisions'], values['small'])))


def main():
    args = parse_args()
    with open(args.annotation, 'r', encoding='utf-8') as annotation_file:
        data = json.load(annotation_file)

    per_size, overall, size_weights = analyze(
        data, args.target_sizes, args.strides, args.small_max_size,
        args.grid_offset)
    for target_size, stats in per_size.items():
        print_stats(
            '\nTarget size {} (sampling weight {})'.format(
                target_size, size_weights[target_size]), stats, args.strides)
    print_stats('\nWeighted overall', overall, args.strides)


if __name__ == '__main__':
    main()
